import asyncio
import hashlib
import os
import tempfile

import botocore.session
import pytest

from s3fs.core import (
    S3FileSystem,
    _expected_multipart_etag,
    export_multipart_checkpoint,
)

SERVICE_MODEL = botocore.session.Session().get_service_model("s3")


class FakeS3:
    def __init__(self, fail=()):
        class Meta:
            service_model = SERVICE_MODEL

        self.meta = Meta()
        self.fail = set(fail)
        self.uploads = {}
        self.objects = {}
        self.calls = {}

    def count(self, method):
        self.calls[method] = self.calls.get(method, 0) + 1

    async def create_multipart_upload(self, **kwargs):
        self.count("create_multipart_upload")
        upload_id = f"upload-{len(self.uploads) + 1}"
        self.uploads[upload_id] = {
            "key": (kwargs["Bucket"], kwargs["Key"]),
            "parts": {},
        }
        return {"UploadId": upload_id}

    async def upload_part(self, **kwargs):
        self.count("upload_part")
        body = kwargs["Body"]
        etag = hashlib.md5(body).hexdigest()
        self.uploads[kwargs["UploadId"]]["parts"][kwargs["PartNumber"]] = {
            "ETag": etag,
            "Size": len(body),
            "Body": body,
        }
        if "part" in self.fail and kwargs["PartNumber"] == 1:
            raise ConnectionError("part uploaded but response was lost")
        return {"ETag": etag}

    async def list_parts(self, **kwargs):
        self.count("list_parts")
        if kwargs["UploadId"] not in self.uploads:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchUpload"}}, "ListParts")
        parts = self.uploads[kwargs["UploadId"]]["parts"]
        return {
            "Parts": [
                {"PartNumber": number, "ETag": part["ETag"], "Size": part["Size"]}
                for number, part in sorted(parts.items())
            ],
            "IsTruncated": False,
        }

    async def complete_multipart_upload(self, **kwargs):
        self.count("complete_multipart_upload")
        upload = self.uploads.pop(kwargs["UploadId"])
        parts = kwargs["MultipartUpload"]["Parts"]
        data = b"".join(
            upload["parts"][part["PartNumber"]]["Body"] for part in parts
        )
        etag = _expected_multipart_etag(parts)
        self.objects[upload["key"]] = (data, etag)
        if "complete" in self.fail:
            raise ConnectionError("complete succeeded but response was lost")
        return {"ETag": etag}

    async def head_object(self, **kwargs):
        self.count("head_object")
        data, etag = self.objects[(kwargs["Bucket"], kwargs["Key"])]
        return {"ContentLength": len(data), "ETag": etag}

    async def abort_multipart_upload(self, **kwargs):
        self.count("abort_multipart_upload")
        self.uploads.pop(kwargs["UploadId"])
        return {}


def make_filesystem(fake):
    fs = S3FileSystem(asynchronous=True)
    fs._s3 = fake

    async def call_s3(method, *args, **kwargs):
        return await getattr(fake, method)(**kwargs)

    fs._call_s3 = call_s3
    return fs


async def resume_after_lost_part_response(directory):
    source = os.path.join(directory, "source")
    checkpoint = os.path.join(directory, "checkpoint.json")
    chunksize = 4
    data = b"abcdefghijkl"
    with open(source, "wb") as f:
        f.write(data)

    fake = FakeS3(fail={"part"})
    original = make_filesystem(fake)
    with pytest.raises(ConnectionError):
        await original._put_file(
            source,
            "bucket/key",
            chunksize=chunksize,
            max_concurrency=1,
            checkpoint_file=checkpoint,
        )

    assert export_multipart_checkpoint(checkpoint)["upload"]["parts"] == []
    resumed = make_filesystem(fake)
    await resumed._resume_put_file(checkpoint, checkpoint)

    assert fake.calls["upload_part"] == 3
    assert fake.objects[("bucket", "key")][0] == data
    assert export_multipart_checkpoint(checkpoint)["status"] == "completed"


async def resume_after_lost_complete_response(directory):
    source = os.path.join(directory, "source")
    checkpoint = os.path.join(directory, "checkpoint.json")
    data = b"abcdefghijkl"
    with open(source, "wb") as f:
        f.write(data)

    fake = FakeS3(fail={"complete"})
    original = make_filesystem(fake)
    await original._put_file(
        source,
        "bucket/key",
        chunksize=4,
        max_concurrency=1,
        checkpoint_file=checkpoint,
    )

    uploaded_parts = fake.calls["upload_part"]
    resumed = make_filesystem(fake)
    await resumed._resume_put_file(checkpoint, checkpoint)

    assert fake.calls["upload_part"] == uploaded_parts
    assert fake.calls["head_object"] == 1
    assert fake.calls["complete_multipart_upload"] == 1
    assert fake.objects[("bucket", "key")][0] == data


async def reject_unsafe_resume(directory):
    source = os.path.join(directory, "source")
    checkpoint = os.path.join(directory, "checkpoint.json")
    data = b"abcdefghijkl"
    with open(source, "wb") as f:
        f.write(data)

    fake = FakeS3(fail={"part"})
    fs = make_filesystem(fake)
    with pytest.raises(ConnectionError):
        await fs._put_file(
            source,
            "bucket/key",
            chunksize=4,
            max_concurrency=1,
            checkpoint_file=checkpoint,
        )

    upload_id = export_multipart_checkpoint(checkpoint)["upload"]["upload_id"]
    with open(source, "r+b") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="content changed"):
        await fs._resume_put_file(checkpoint, checkpoint)

    with open(source, "wb") as f:
        f.write(data)
    with pytest.raises(ValueError, match="chunksize changed"):
        await fs._resume_put_file(checkpoint, checkpoint, chunksize=5)
    with pytest.raises(ValueError, match="parameters changed"):
        await fs._resume_put_file(
            checkpoint, checkpoint, Metadata={"checkpoint": "changed"}
        )

    await fs._abort_mpu("bucket", "key", upload_id)
    with pytest.raises(ValueError, match="no longer valid"):
        await fs._resume_put_file(checkpoint, checkpoint)


def test_multipart_checkpoint_resume():
    with tempfile.TemporaryDirectory() as directory:
        asyncio.run(resume_after_lost_part_response(directory))


def test_multipart_checkpoint_lost_complete_response():
    with tempfile.TemporaryDirectory() as directory:
        asyncio.run(resume_after_lost_complete_response(directory))


def test_multipart_checkpoint_rejects_unsafe_resume():
    with tempfile.TemporaryDirectory() as directory:
        asyncio.run(reject_unsafe_resume(directory))
