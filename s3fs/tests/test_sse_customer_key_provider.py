import json
import pickle

import pytest

from s3fs.core import S3FileSystem, S3File, SSECustomerKeyContext
from s3fs.utils import SSEParams


class RecordingS3FileSystem(S3FileSystem):
    def __init__(self, key_provider, *args, **kwargs):
        super().__init__(
            sse_customer_key_provider=key_provider,
            skip_instance_cache=True,
            version_aware=True,
            *args,
            **kwargs,
        )
        self.calls = []
        self._set_session_lock = None

    async def set_session(self, refresh=False, kwargs={}):
        return None

    async def _call_s3(self, method, *akwarglist, **kwargs):
        call = {"method": method, "kwargs": kwargs.copy()}
        context = call["kwargs"].pop("sse_key_context", None)
        call["sse_key_context"] = context
        if context is not None:
            call["kwargs"].update(context.kwargs())
        source_context = call["kwargs"].pop("source_sse_key_context", None)
        call["source_sse_key_context"] = source_context
        if source_context is not None:
            call["kwargs"].update(source_context.copy_source_kwargs())
        for akwargs in akwarglist:
            call["kwargs"].update(akwargs)
        self.calls.append(call)

        if method == "head_object":
            return {
                "ContentLength": 1,
                "ETag": '"etag"',
                "VersionId": "source-version",
            }
        if method == "get_object":
            class Body:
                async def read(self):
                    return b"object-data"

                def close(self):
                    pass

            return {"Body": Body()}
        if method == "create_multipart_upload":
            return {"UploadId": "upload-id"}
        if method in {"upload_part", "upload_part_copy"}:
            return {"ETag": '"part-etag"', "CopyPartResult": {"ETag": '"part-etag"'}}
        if method == "complete_multipart_upload":
            return {"VersionId": "written-version"}
        if method == "copy_object":
            return {}
        raise AssertionError(method)


def key_for(operation, bucket, key, version_id=None):
    if operation == "read":
        value = f"source-key:{bucket}:{key}:{version_id or ''}"
    else:
        value = f"target-key:{bucket}:{key}"
    return {
        "SSECustomerAlgorithm": "AES256",
        "SSECustomerKey": value.encode().ljust(32, b"x")[:32],
    }


@pytest.fixture
def provider():
    calls = []

    def get_key(operation, bucket, key, version_id=None):
        calls.append((operation, bucket, key, version_id))
        return key_for(operation, bucket, key, version_id)

    get_key.calls = calls
    return get_key


@pytest.fixture
def make_fs():
    def make(provider):
        return RecordingS3FileSystem(provider)

    return make


def call_kwargs(fs, method):
    calls = [call for call in fs.calls if call["method"] == method]
    assert len(calls) == 1
    return calls[0]["kwargs"]


def test_context_repr_and_serialization_do_not_contain_key():
    context = SSECustomerKeyContext(
        SSECustomerAlgorithm="AES256", SSECustomerKey=b"0123456789abcdef0123456789abcdef"
    )
    assert "0123456789abcdef" not in repr(context)
    with pytest.raises(RuntimeError):
        pickle.dumps(context)


def test_filesystem_serialization_omits_provider(provider):
    fs = S3FileSystem(
        sse_customer_key_provider=provider,
        skip_instance_cache=True,
        version_aware=True,
    )
    encoded = json.dumps(fs.to_dict())
    assert "sse_customer_key_provider" not in encoded
    assert "source-key" not in encoded
    assert "source-key" not in repr(fs)


def test_sync_key_provider_entry_is_available():
    assert callable(S3FileSystem.resolve_sse_customer_key)
    assert callable(S3FileSystem._resolve_sse_customer_key)


def test_provider_can_return_sse_params():
    context = SSECustomerKeyContext.from_value(
        SSEParams(
            sse_customer_algorithm="AES256",
            sse_customer_key=b"k" * 32,
            server_side_encryption="AES256",
        )
    )

    assert context.kwargs()["SSECustomerAlgorithm"] == "AES256"


async def test_async_read_sends_object_key(make_fs, provider):
    fs = make_fs(provider)
    await fs._cat_file("bucket/source", version_id="version-1")

    assert provider.calls[0] == ("read", "bucket", "source", "version-1")
    kwargs = call_kwargs(fs, "get_object")
    assert kwargs["SSECustomerKey"].startswith(b"source-key:bucket:source")
    assert "source-key" not in str(fs.dircache.items())


async def test_async_key_provider_is_supported():
    async def provider(operation, bucket, key, version_id=None):
        return key_for(operation, bucket, key, version_id)

    fs = RecordingS3FileSystem(provider)
    await fs._cat_file("bucket/source")

    assert call_kwargs(fs, "get_object")["SSECustomerKey"].startswith(b"source-key")


async def test_multipart_write_uses_one_context(make_fs, provider):
    fs = make_fs(provider)
    data = b"a" * (11 * 2**20)
    await fs._pipe_file(
        "bucket/target", data, chunksize=5 * 2**20, max_concurrency=1
    )

    methods = [call["method"] for call in fs.calls]
    assert methods == [
        "create_multipart_upload",
        "upload_part",
        "upload_part",
        "upload_part",
        "complete_multipart_upload",
    ]
    keys = {
        call["kwargs"]["SSECustomerKey"]
        for call in fs.calls
        if "SSECustomerKey" in call["kwargs"]
    }
    assert keys == {key_for("write", "bucket", "target")["SSECustomerKey"]}
    multipart_calls = [
        call for call in fs.calls if call["method"] != "put_object"
    ]
    assert len({call["sse_key_context"] for call in multipart_calls}) == 1
    assert provider.calls == [("write", "bucket", "target", None)]


def test_s3file_keyword_arguments_remain_clean():
    import inspect

    signature = inspect.signature(S3File)
    assert "write_sse_key_context" in signature.parameters
    assert "SSECustomerKey" not in signature.parameters


async def test_cross_key_copy_sends_independent_source_and_target_contexts(
    make_fs, provider
):
    fs = make_fs(provider)
    await fs._cp_file("bucket/source?versionId=source-version", "other/target")

    kwargs = call_kwargs(fs, "copy_object")
    source_key = key_for("read", "bucket", "source", "source-version")["SSECustomerKey"]
    target_key = key_for("write", "other", "target")["SSECustomerKey"]
    assert kwargs["CopySource"] == {
        "Bucket": "bucket",
        "Key": "source",
        "VersionId": "source-version",
    }
    assert kwargs["CopySourceSSECustomerKey"] == source_key
    assert kwargs["SSECustomerKey"] == target_key
    assert source_key != target_key


async def test_multipart_cross_key_copy_sends_independent_contexts(
    make_fs, provider, monkeypatch
):
    fs = make_fs(provider)
    monkeypatch.setattr("s3fs.core.MANAGED_COPY_THRESHOLD", 0)
    await fs._cp_file("bucket/source", "other/target")

    create_kwargs = call_kwargs(fs, "create_multipart_upload")
    part_kwargs = call_kwargs(fs, "upload_part_copy")
    complete_kwargs = call_kwargs(fs, "complete_multipart_upload")
    source_key = key_for("read", "bucket", "source")["SSECustomerKey"]
    target_key = key_for("write", "other", "target")["SSECustomerKey"]
    assert create_kwargs["SSECustomerKey"] == target_key
    assert part_kwargs["CopySourceSSECustomerKey"] == source_key
    assert part_kwargs["SSECustomerKey"] == target_key
    assert complete_kwargs["SSECustomerKey"] == target_key
    target_ids = {
        call["sse_key_context"]
        for call in fs.calls
        if call["method"]
        in {
            "create_multipart_upload",
            "upload_part_copy",
            "complete_multipart_upload",
        }
    }
    source_ids = {
        call["source_sse_key_context"]
        for call in fs.calls
        if call["method"] == "upload_part_copy"
    }
    assert len(target_ids) == 1
    assert len(source_ids) == 1
    assert target_ids != source_ids


async def test_provider_failure_prevents_multipart_upload():
    def failing_provider(operation, bucket, key, version_id=None):
        if operation == "write":
            raise RuntimeError("no key")
        return key_for(operation, bucket, key, version_id)

    fs = RecordingS3FileSystem(failing_provider)
    with pytest.raises(RuntimeError, match="no key"):
        await fs._pipe_file(
            "bucket/target", b"a" * (11 * 2**20), chunksize=5 * 2**20
        )

    assert not [
        call for call in fs.calls if call["method"] == "create_multipart_upload"
    ]
