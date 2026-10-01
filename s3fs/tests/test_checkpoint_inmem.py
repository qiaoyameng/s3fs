"""Resumable-upload tests against a zero-network in-memory S3 stub.

The stub emulates the small slice of the S3 API that
:mod:`s3fs.checkpoint` needs (create/list/upload/complete/head) and lets us
deterministically inject "the request succeeded on the server but the
response was lost" failures, without binding a network port.
"""

import asyncio
import hashlib
import os
import socket
import uuid

import pytest
from botocore.exceptions import ClientError

from s3fs.checkpoint import (
    UploadCheckpoint,
    CheckpointMismatchError,
    UploadExpiredError,
    put_file_with_checkpoint,
    resume_put_file,
    create_and_save_checkpoint,
)


def _not_found(op):
    return ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"},
         "ResponseMetadata": {"HTTPStatusCode": 404}},
        op,
    )


class InMemoryS3Stub:
    """Minimal async S3 surface with optional per-call fault injection."""

    def __init__(self):
        self.uploads = {}   # upload_id -> {"bucket", "key", "parts": {n: {...}}}
        self.objects = {}   # (bucket, key) -> bytes
        self.calls = []
        self.reset_faults()

    def reset_faults(self):
        self.fault_on = None
        self.fault_exc = None
        self.fault_fired = False
        self.head_missing_once_after = None
        self._complete_response_lost = False
        self._head_hidden_used = False

    def _maybe_fault(self, method, **kwargs):
        if (
            not self.fault_fired
            and self.fault_on is not None
            and self.fault_on(method, kwargs)
        ):
            self.fault_fired = True
            raise self.fault_exc

    async def _call_s3(self, method, *akwarglist, **kwargs):
        self.calls.append(method)
        handler = getattr(self, method)
        return await handler(*akwarglist, **kwargs)

    async def create_multipart_upload(self, *a, Bucket, Key, **kw):
        upload_id = str(uuid.uuid4())
        self.uploads[upload_id] = {"bucket": Bucket, "key": Key, "parts": {}}
        return {"UploadId": upload_id}

    async def upload_part(self, *a, Bucket, Key, PartNumber, UploadId, Body=b"",
                          **kw):
        self._maybe_fault(
            "upload_part", PartNumber=PartNumber, UploadId=UploadId
        )
        data = bytes(Body)
        upload = self.uploads[UploadId]
        etag = '"%s"' % hashlib.md5(data).hexdigest()
        upload["parts"][PartNumber] = {
            "PartNumber": PartNumber, "ETag": etag, "Size": len(data),
            "_data": data,
        }
        return {"ETag": etag}

    async def list_parts(self, *a, Bucket, Key, UploadId, **kw):
        if UploadId not in self.uploads:
            raise ClientError(
                {"Error": {"Code": "NoSuchUpload", "Message": "no such upload"},
                 "ResponseMetadata": {"HTTPStatusCode": 404}},
                "ListParts",
            )
        parts = sorted(
            self.uploads[UploadId]["parts"].values(),
            key=lambda p: p["PartNumber"],
        )
        return {"Parts": parts, "IsTruncated": False}

    async def complete_multipart_upload(self, *a, Bucket, Key, UploadId,
                                        MultipartUpload, **kw):
        upload = self.uploads.get(UploadId)
        if upload is None:
            raise ClientError(
                {"Error": {"Code": "NoSuchUpload", "Message": "gone"},
                 "ResponseMetadata": {"HTTPStatusCode": 404}},
                "CompleteMultipartUpload",
            )
        ordered = sorted(MultipartUpload["Parts"], key=lambda p: p["PartNumber"])
        blob = b"".join(
            upload["parts"][p["PartNumber"]]["_data"] for p in ordered
        )
        # Raise the response-loss fault *after* materialising the object.
        # If the follow-up head_object is emulated to miss, the upload is
        # kept open so the driver can observe it via ListParts and re-issue
        # complete; otherwise it is closed as a real S3 server would.
        self.objects[(Bucket, Key)] = blob
        before = self.fault_fired
        self._maybe_fault("complete_multipart_upload")
        self._complete_response_lost = (not before) and self.fault_fired
        if not (self._complete_response_lost and self.head_missing_once_after):
            del self.uploads[UploadId]
        return {"ETag": '"completed"', "Location": "http://stub/x"}

    async def head_object(self, *a, Bucket, Key, **kw):
        if (
            getattr(self, "head_missing_once_after", False)
            and self.fault_fired
            and not getattr(self, "_head_hidden_used", False)
        ):
            self._head_hidden_used = True
            raise _not_found("HeadObject")
        if (Bucket, Key) not in self.objects:
            raise _not_found("HeadObject")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}


class StubFS(InMemoryS3Stub):
    """Adds just enough of the real S3FileSystem surface for the driver."""

    max_concurrency = 4

    def split_path(self, path):
        bucket, key = path.split("/", 1)
        return bucket, key, None

    def invalidate_cache(self, *a, **kw):
        pass


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def fs():
    f = StubFS()
    f.reset_faults()
    return f


def _pattern_data(size, seed=7):
    # fast deterministic non-repeating-ish pattern via a repeated 4 KiB tile
    tile = bytes((i * 31 + seed) % 251 for i in range(4096))
    repeats, rem = divmod(size, len(tile))
    return tile * repeats + tile[:rem]


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.bin"
    data = _pattern_data(12 * 2**20 + 7)
    path.write_bytes(data)
    return str(path), data


def _fault(fs, predicate, exc=None):
    fs.reset_faults()
    fs.fault_on = predicate
    fs.fault_exc = exc or socket.timeout("simulated response loss")


def test_resume_after_mid_upload_crash(fs, source, tmp_path):
    lpath, data = source
    cp_path = str(tmp_path / "cp.json")

    _fault(fs, lambda m, kw: m == "upload_part" and kw.get("PartNumber") == 3)
    with pytest.raises(socket.timeout):
        run(put_file_with_checkpoint(
            fs, lpath, "bkt/o", cp_path, chunksize=5 * 2**20))

    cp = UploadCheckpoint.load(cp_path)
    assert set(cp.parts) == {1, 2} and cp.total_parts == 3
    uploads_before = dict(fs.uploads)
    assert cp.upload_id in uploads_before

    # restart = new stub-backed instance sharing the same server state
    fs2 = StubFS()
    fs2.uploads = fs.uploads
    fs2.objects = fs.objects
    uploaded = []

    async def track(method, *a, **kw):
        if method == "upload_part":
            uploaded.append(kw["PartNumber"])
        return await StubFS._call_s3(fs2, method, *a, **kw)

    fs2._call_s3 = track
    run(resume_put_file(fs2, cp_path))

    assert uploaded == [3]  # only the missing part
    assert fs2.objects[("bkt", "o")] == data
    assert UploadCheckpoint.load(cp_path).state == "completed"


def test_complete_succeeds_but_response_lost(fs, source, tmp_path):
    # The response to complete_multipart_upload is lost *after* the server
    # materialised the object. The driver must query status (head_object)
    # and must not re-submit any part data; it then marks the checkpoint
    # completed itself.
    lpath, _ = source
    data = _pattern_data(5 * 2**20 + 99)
    with open(lpath, "wb") as f:
        f.write(data)
    cp_path = str(tmp_path / "cp.json")

    _fault(fs, lambda m, kw: m == "complete_multipart_upload")
    # does not raise: the ambiguous result is resolved via a status query
    run(put_file_with_checkpoint(
        fs, lpath, "bkt/o", cp_path, chunksize=5 * 2**20))

    assert fs.objects[("bkt", "o")] == data
    assert UploadCheckpoint.load(cp_path).state == "completed"
    assert fs.calls.count("upload_part") == 2  # one per part, no re-send
    assert "head_object" in fs.calls        # status was queried, not re-put
    assert fs.calls.count("complete_multipart_upload") == 1


def test_uncertain_completion_missing_then_recomplete(fs, source, tmp_path):
    lpath, _ = source
    data = _pattern_data(5 * 2**20 + 5)
    with open(lpath, "wb") as f:
        f.write(data)
    cp_path = str(tmp_path / "cp.json")

    # first complete succeeds server-side, response lost; first head says
    # "missing" to emulate read-after-write lag; the driver must re-complete.
    _fault(fs, lambda m, kw: m == "complete_multipart_upload")
    fs.head_missing_once_after = True

    run(put_file_with_checkpoint(
        fs, lpath, "bkt/o", cp_path, chunksize=5 * 2**20))

    # internal retry sequence: complete(fault) -> head(404) -> list_parts
    # -> complete(ok) -> head/cp completed
    methods = fs.calls
    first = methods.index("complete_multipart_upload")
    second = methods.index("complete_multipart_upload", first + 1)
    between = methods[first + 1:second]
    assert "head_object" in between and "list_parts" in between
    assert methods.count("upload_part") == 2  # two parts, each sent once
    assert fs.objects[("bkt", "o")] == data
    assert UploadCheckpoint.load(cp_path).state == "completed"


def test_reject_changed_content(fs, source, tmp_path):
    lpath, data = source
    cp_path = str(tmp_path / "cp.json")
    _fault(fs, lambda m, kw: m == "upload_part" and kw.get("PartNumber") == 2)
    with pytest.raises(socket.timeout):
        run(put_file_with_checkpoint(
            fs, lpath, "bkt/o", cp_path, chunksize=5 * 2**20))

    with open(lpath, "r+b") as f:
        f.seek(100)
        f.write(b"\x00" * 10)  # same length, different bytes

    with pytest.raises(CheckpointMismatchError, match="SHA256"):
        run(resume_put_file(fs, cp_path))


def test_reject_changed_chunksize(fs, source, tmp_path):
    lpath, data = source
    cp_path = str(tmp_path / "cp.json")
    _fault(fs, lambda m, kw: m == "upload_part" and kw.get("PartNumber") == 2)
    with pytest.raises(socket.timeout):
        run(put_file_with_checkpoint(
            fs, lpath, "bkt/o", cp_path, chunksize=5 * 2**20))

    with pytest.raises(CheckpointMismatchError, match="Chunksize"):
        run(resume_put_file(fs, cp_path, chunksize=6 * 2**20))


def test_reject_expired_upload(fs, source, tmp_path):
    lpath, data = source
    cp_path = str(tmp_path / "cp.json")
    cp = run(create_and_save_checkpoint(
        fs, lpath, "bkt/o", cp_path, chunksize=5 * 2**20))
    del fs.uploads[cp.upload_id]  # expired / aborted server-side
    with pytest.raises(UploadExpiredError):
        run(resume_put_file(fs, cp_path))


def test_export_contains_identity_scheme_uploadid_and_parts(fs, source,
                                                            tmp_path):
    lpath, data = source
    cp_path = str(tmp_path / "cp.json")
    cp = run(create_and_save_checkpoint(
        fs, lpath, "bkt/o", cp_path, chunksize=5 * 2**20,
        ContentType="application/octet-stream"))

    raw = open(cp_path).read()
    assert cp.upload_id and cp.parts == {}
    assert '"source_sha256"' in raw
    assert cp.chunksize == 5 * 2**20 and cp.total_parts == 3
    assert cp.kwargs["ContentType"] == "application/octet-stream"
    assert ("bkt", "o") not in fs.objects  # export sends no data

    run(resume_put_file(fs, cp_path))
    assert fs.objects[("bkt", "o")] == data

    # completed checkpoint is idempotent and must not re-upload
    observed = []

    async def track(method, *a, **kw):
        observed.append(method)
        return await StubFS._call_s3(fs, method, *a, **kw)

    fs._call_s3 = track
    run(resume_put_file(fs, cp_path))
    assert "upload_part" not in observed


def test_put_with_checkpoint_idempotent_after_restart(fs, source, tmp_path):
    # Re-issuing the same put_file call resumes the same upload id.
    lpath, data = source
    cp_path = str(tmp_path / "cp.json")
    _fault(fs, lambda m, kw: m == "upload_part" and kw.get("PartNumber") == 2)
    with pytest.raises(socket.timeout):
        run(put_file_with_checkpoint(
            fs, lpath, "bkt/o", cp_path, chunksize=5 * 2**20))
    upload_id = UploadCheckpoint.load(cp_path).upload_id

    run(put_file_with_checkpoint(
        fs, lpath, "bkt/o", cp_path, chunksize=5 * 2**20))
    assert fs.objects[("bkt", "o")] == data
    assert UploadCheckpoint.load(cp_path).upload_id == upload_id
