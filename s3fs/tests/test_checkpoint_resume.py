"""Tests for resumable multipart uploads driven by on-disk checkpoints.

The "local S3 stub" is a threaded moto server. Several tests inject faults
into the async S3 call dispatch to emulate a process crash mid-upload and -
most importantly - a ``complete_multipart_upload`` that succeeds on the
server but whose response is lost on the client.
"""

import os

import pytest
import socket
import requests
from moto.moto_server.threaded_moto_server import ThreadedMotoServer

import s3fs
from s3fs.core import S3FileSystem
from s3fs.checkpoint import (
    UploadCheckpoint,
    CheckpointMismatchError,
    UploadExpiredError,
)

PORT = 5566
ENDPOINT_URI = f"http://127.0.0.1:{PORT}/"
BUCKET = "resume-test"


def _can_bind(host, port):
    import contextlib
    import socket as _socket

    try:
        ctx = contextlib.closing(
            _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        )
    except PermissionError:
        return False
    with ctx as probe:
        try:
            probe.bind((host, port))
        except (PermissionError, OSError):
            return False
    return True


@pytest.fixture(scope="module")
def s3_base():
    if not _can_bind("127.0.0.1", PORT):
        pytest.skip("sandbox forbids binding the local S3 stub port")
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=PORT, verbose=False)
    server.start()
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "foo")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "foo")
    os.environ.pop("AWS_PROFILE", None)
    yield server
    server.stop()


@pytest.fixture(autouse=True)
def reset_state(s3_base):
    requests.post(f"{ENDPOINT_URI}/moto-api/reset")


def make_fs(retries=1):
    S3FileSystem.clear_instance_cache()
    fs = S3FileSystem(
        anon=False,
        client_kwargs={"endpoint_url": ENDPOINT_URI},
        retries=retries,
    )
    try:
        get_client().create_bucket(Bucket=BUCKET)
    except Exception:
        pass
    return fs


def get_client():
    from botocore.session import Session

    return Session().create_client("s3", endpoint_url=ENDPOINT_URI)


def make_source(tmpdir, size=5 * 2**20 + 123, seed=7):
    # deterministic non-trivial pattern, fast even for tens of MiB
    tile = bytes((i * 31 + seed) % 251 for i in range(4096))
    repeats, rem = divmod(size, len(tile))
    data = tile * repeats + tile[:rem]
    path = os.path.join(str(tmpdir), "source.bin")
    with open(path, "wb") as f:
        f.write(data)
    return path, data


def inject_call_failure(fs, monkeypatch, predicate, exc):
    """Make ``fs._call_s3`` raise ``exc`` once when ``predicate(kwargs)``."""
    real = fs._call_s3
    state = {"fired": False}

    async def flaky(method, *akwarglist, **kwargs):
        if not state["fired"] and predicate(method, kwargs):
            state["fired"] = True
            raise exc
        return await real(method, *akwarglist, **kwargs)

    monkeypatch.setattr(fs, "_call_s3", flaky)
    return state


def object_bytes(key):
    resp = get_client().get_object(Bucket=BUCKET, Key=key)
    return resp["Body"].read()


def test_resume_after_crash_uploads_only_missing_parts(tmpdir, monkeypatch):
    lpath, data = make_source(tmpdir, size=12 * 2**20 + 7)
    rpath = f"{BUCKET}/crash.bin"
    cp_path = os.path.join(str(tmpdir), "cp.json")

    # retries=1 so the injected error propagates immediately, like a hard crash
    fs1 = make_fs(retries=1)
    seen = {"upload_part": 0}
    real = fs1._call_s3

    async def fail_on_third_part(method, *akwarglist, **kwargs):
        if method == "upload_part":
            seen["upload_part"] += 1
            if seen["upload_part"] == 3:
                raise socket.timeout("simulated crash while uploading part 3")
        return await real(method, *akwarglist, **kwargs)

    monkeypatch.setattr(fs1, "_call_s3", fail_on_third_part)
    with pytest.raises(socket.timeout):
        fs1.put_file(
            lpath, rpath, chunksize=5 * 2**20, checkpoint_path=cp_path
        )

    cp = UploadCheckpoint.load(cp_path)
    assert cp.upload_id
    assert set(cp.parts) == {1, 2}
    assert cp.total_parts == 3

    # "restart": a fresh process means a brand new filesystem instance
    fs2 = make_fs(retries=1)
    uploaded = []
    real2 = fs2._call_s3

    async def track(method, *akwarglist, **kwargs):
        if method == "upload_part":
            uploaded.append(kwargs["PartNumber"])
        return await real2(method, *akwarglist, **kwargs)

    monkeypatch.setattr(fs2, "_call_s3", track)
    fs2.resume_put_file(cp_path)

    # parts 1..2 confirmed on the server, only the missing part 3 is resent
    assert uploaded == [3]
    assert object_bytes("crash.bin") == data
    assert UploadCheckpoint.load(cp_path).state == "completed"


def test_completion_succeeds_but_response_is_lost(tmpdir, monkeypatch):
    lpath, data = make_source(tmpdir, size=5 * 2**20 + 99)
    rpath = f"{BUCKET}/lost-response.bin"
    cp_path = os.path.join(str(tmpdir), "cp.json")

    fs1 = make_fs(retries=1)
    methods1 = []
    real1 = fs1._call_s3
    exc = socket.timeout("response lost after complete_multipart_upload")
    state = {"fired": False}

    async def flaky(method, *akwarglist, **kwargs):
        methods1.append(method)
        if (
            not state["fired"]
            and method == "complete_multipart_upload"
        ):
            state["fired"] = True
            raise exc
        return await real1(method, *akwarglist, **kwargs)

    monkeypatch.setattr(fs1, "_call_s3", flaky)
    # put_file does not raise: the uncertain result is resolved by querying
    # the server before re-issuing anything.
    fs1.put_file(
        lpath, rpath, chunksize=5 * 2**20, checkpoint_path=cp_path
    )
    assert state["fired"]
    assert object_bytes("lost-response.bin") == data
    # part data was submitted exactly as often as required, never re-put
    assert methods1.count("upload_part") == 2
    assert "head_object" in methods1
    assert UploadCheckpoint.load(cp_path).state == "completed"

    # A later resume is an idempotent no-op that only checks object status.
    fs2 = make_fs()
    methods2 = []
    real2 = fs2._call_s3

    async def track(method, *akwarglist, **kwargs):
        methods2.append(method)
        return await real2(method, *akwarglist, **kwargs)

    monkeypatch.setattr(fs2, "_call_s3", track)
    fs2.resume_put_file(cp_path)

    assert "upload_part" not in methods2
    assert "complete_multipart_upload" not in methods2
    assert "head_object" in methods2
    assert object_bytes("lost-response.bin") == data


def test_uncertain_completion_retried_after_status_query(tmpdir, monkeypatch):
    # complete_multipart_upload succeeds on the server, but the first
    # response is lost AND the very first head_object still says "missing";
    # the code must query state, find the upload open, and re-issue
    # complete (not resubmit any part data).
    from botocore.exceptions import ClientError

    lpath, data = make_source(tmpdir, size=5 * 2**20 + 5)
    rpath = f"{BUCKET}/uncertain.bin"
    cp_path = os.path.join(str(tmpdir), "cp.json")

    fs = make_fs(retries=1)
    real = fs._call_s3
    complete_calls = {"n": 0}
    upload_calls = {"n": 0}
    phase = {"head_after_lost": 0}

    async def flaky(method, *akwarglist, **kwargs):
        if method == "upload_part":
            upload_calls["n"] += 1
            return await real(method, *akwarglist, **kwargs)
        if method == "complete_multipart_upload":
            complete_calls["n"] += 1
            result = await real(method, *akwarglist, **kwargs)
            if complete_calls["n"] == 1:
                raise socket.timeout("response lost")
            return result
        if method == "head_object" and complete_calls["n"] >= 1:
            phase["head_after_lost"] += 1
            if phase["head_after_lost"] == 1:
                raise ClientError(
                    {"Error": {"Code": "404", "Message": "Not Found"},
                     "ResponseMetadata": {"HTTPStatusCode": 404}},
                    "HeadObject",
                )
        return await real(method, *akwarglist, **kwargs)

    monkeypatch.setattr(fs, "_call_s3", flaky)
    fs.put_file(lpath, rpath, chunksize=5 * 2**20, checkpoint_path=cp_path)

    assert complete_calls["n"] == 2  # queried first, then completed again
    assert upload_calls["n"] == 1  # part data submitted exactly once
    assert object_bytes("uncertain.bin") == data
    assert UploadCheckpoint.load(cp_path).state == "completed"


def test_reject_resume_when_source_content_changes(tmpdir, monkeypatch):
    lpath, data = make_source(tmpdir, size=12 * 2**20 + 1)
    rpath = f"{BUCKET}/changed.bin"
    cp_path = os.path.join(str(tmpdir), "cp.json")

    fs1 = make_fs(retries=1)
    inject_call_failure(
        fs1,
        monkeypatch,
        lambda method, kw: method == "upload_part" and kw.get("PartNumber") == 2,
        socket.timeout("crash"),
    )
    with pytest.raises(socket.timeout):
        fs1.put_file(
            lpath, rpath, chunksize=5 * 2**20, checkpoint_path=cp_path
        )

    # same size, different bytes -> must be detected via SHA256
    with open(lpath, "r+b") as f:
        f.seek(100)
        f.write(b"\x00" * 10)

    fs2 = make_fs()
    with pytest.raises(CheckpointMismatchError, match="SHA256"):
        fs2.resume_put_file(cp_path)


def test_reject_resume_when_chunksize_changes(tmpdir, monkeypatch):
    lpath, data = make_source(tmpdir, size=12 * 2**20 + 1)
    rpath = f"{BUCKET}/rechunk.bin"
    cp_path = os.path.join(str(tmpdir), "cp.json")

    fs1 = make_fs(retries=1)
    inject_call_failure(
        fs1,
        monkeypatch,
        lambda method, kw: method == "upload_part" and kw.get("PartNumber") == 2,
        socket.timeout("crash"),
    )
    with pytest.raises(socket.timeout):
        fs1.put_file(
            lpath, rpath, chunksize=5 * 2**20, checkpoint_path=cp_path
        )

    fs2 = make_fs()
    with pytest.raises(CheckpointMismatchError, match="Chunksize"):
        fs2.resume_put_file(cp_path, chunksize=6 * 2**20)


def test_reject_resume_when_upload_expired(tmpdir):
    lpath, data = make_source(tmpdir, size=5 * 2**20 + 1)
    rpath = f"{BUCKET}/expired.bin"
    cp_path = os.path.join(str(tmpdir), "cp.json")

    fs1 = make_fs()
    fs1.export_upload_checkpoint(lpath, rpath, cp_path, chunksize=5 * 2**20)
    upload_id = UploadCheckpoint.load(cp_path).upload_id
    get_client().abort_multipart_upload(
        Bucket=BUCKET, Key="expired.bin", UploadId=upload_id
    )

    fs2 = make_fs()
    with pytest.raises(UploadExpiredError):
        fs2.resume_put_file(cp_path)


def test_export_then_resume_creates_exact_object(tmpdir):
    lpath, data = make_source(tmpdir, size=7 * 2**20 + 33)
    rpath = f"{BUCKET}/exported.bin"
    cp_path = os.path.join(str(tmpdir), "ckpt.json")

    fs1 = make_fs()
    cp = fs1.export_upload_checkpoint(lpath, rpath, cp_path, chunksize=5 * 2**20)
    assert cp.upload_id and not cp.parts
    # export must not have uploaded data
    assert not fs1.exists(rpath)

    fs2 = make_fs()  # restart
    fs2.resume_put_file(cp_path)
    assert object_bytes("exported.bin") == data

    # resuming a completed checkpoint is an idempotent no-op
    fs3 = make_fs()
    fs3.resume_put_file(cp_path)
    assert object_bytes("exported.bin") == data
