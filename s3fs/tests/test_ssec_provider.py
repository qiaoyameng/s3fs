"""Tests for the object-level SSE-C key provider (``ssec_key_provider``).

These tests use ``botocore.stub.Stubber`` plus a ``before-parameter-build``
hook to inspect the exact parameters s3fs sends for reads, writes and
cross-key copies, without any network access.

Note: fsspec's ``sync()`` relies on waking an event loop running in a
different thread, which is not permitted in every environment. The
``same_thread_sync`` fixture below runs the coroutine in the current thread
instead, so the synchronous entry points can be exercised regardless.
"""

import asyncio
import io
import pickle

import aiobotocore.session
import fsspec.asyn
import pytest
from botocore.stub import Stubber

import s3fs.core
from s3fs import S3FileSystem, SSECustomerKeyContext, SSECustomerKeyProvider
from s3fs.utils import ParamKwargsHelper, resolve_ssec_provider

KEY_A = b"customer-a-key-0123456789abcdef"
KEY_B = b"customer-b-key-0123456789abcdef"


class AsyncBody:
    """Minimal async stand-in for an aiobotocore streaming body."""

    def __init__(self, data):
        self._buf = io.BytesIO(data)

    async def read(self, n=-1):
        return self._buf.read(n)

    def close(self):
        pass


class DictKeyProvider(SSECustomerKeyProvider):
    """Resolves contexts from a {(bucket, key): raw_key} mapping."""

    def __init__(self, keys):
        self.keys = keys
        self.calls = []

    def resolve(self, operation, bucket, key, version_id=None):
        self.calls.append((operation, bucket, key, version_id))
        raw = self.keys.get((bucket, key))
        if raw is None:
            return None
        return SSECustomerKeyContext(raw)


class RotatingKeyProvider(SSECustomerKeyProvider):
    """Returns a *different* key on every resolve; used to prove pinning."""

    def __init__(self):
        self.calls = []

    def resolve(self, operation, bucket, key, version_id=None):
        self.calls.append(operation)
        return SSECustomerKeyContext(bytes([len(self.calls) % 256]) * 32)


def sse(raw_key):
    return {"SSECustomerAlgorithm": "AES256", "SSECustomerKey": raw_key}


def copy_source_sse(raw_key):
    return {
        "CopySourceSSECustomerAlgorithm": "AES256",
        "CopySourceSSECustomerKey": raw_key,
    }


def make_stubbed_fs(provider=None, asynchronous=False, **kwargs):
    """An S3FileSystem whose client is replaced by a stubbed one.

    Returns (fs, stubber, captured) where ``captured`` is a list of
    ``(operation_name, params)`` for every API call attempted.
    """
    fs = S3FileSystem(
        anon=True,
        skip_instance_cache=True,
        asynchronous=asynchronous,
        ssec_key_provider=provider,
        **kwargs,
    )

    async def _make_client():
        session = aiobotocore.session.AioSession()
        cm = session.create_client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        return await cm.__aenter__()

    client = asyncio.run(_make_client())
    fs._s3 = client
    fs._kwargs_helper = ParamKwargsHelper(client)

    captured = []

    def _capture(params, model, **kw):
        captured.append((model.name, dict(params)))

    client.meta.events.register("before-parameter-build.s3.*", _capture)
    stubber = Stubber(client)
    stubber.activate()
    return fs, stubber, captured


@pytest.fixture()
def same_thread_sync(monkeypatch):
    """Run sync-wrapper coroutines on a fresh loop in the current thread."""

    def _sync(loop, func, *args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    monkeypatch.setattr(fsspec.asyn, "sync", _sync)
    monkeypatch.setattr(s3fs.core, "sync", _sync)


def params_of(captured, operation):
    return [params for name, params in captured if name == operation]


# ---------------------------------------------------------------------------
# Context object safety properties
# ---------------------------------------------------------------------------


def test_context_repr_redacts_key():
    ctx = SSECustomerKeyContext(KEY_A)
    assert KEY_A.decode() not in repr(ctx)
    assert KEY_A.decode() not in str(ctx)
    assert "<redacted>" in repr(ctx)
    assert ctx.sse_kwargs == sse(KEY_A)
    assert ctx.copy_source_kwargs == copy_source_sse(KEY_A)


def test_context_is_not_serializable():
    ctx = SSECustomerKeyContext(KEY_A)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(ctx)


def test_context_requires_key():
    with pytest.raises(ValueError):
        SSECustomerKeyContext(None)


def test_invalid_provider_rejected():
    with pytest.raises(TypeError):
        S3FileSystem(anon=True, skip_instance_cache=True, ssec_key_provider=42)
    with pytest.raises(TypeError):
        resolve_ssec_provider(lambda *a: object(), "read", "b", "k")


def test_provider_kept_out_of_fs_serialization():
    provider = DictKeyProvider({("bucketa", "file"): KEY_A})
    fs = S3FileSystem(anon=True, skip_instance_cache=True, ssec_key_provider=provider)
    assert "ssec_key_provider" not in fs.storage_options
    blob = pickle.dumps(fs)
    assert KEY_A not in blob


# ---------------------------------------------------------------------------
# Synchronous entry points
# ---------------------------------------------------------------------------


def test_sync_read_sends_object_key(same_thread_sync):
    provider = DictKeyProvider({("bucketa", "file"): KEY_A})
    fs, stubber, captured = make_stubbed_fs(provider)
    stubber.add_response(
        "head_object",
        {"ContentLength": 5, "ETag": '"etag"'},
        {"Bucket": "bucketa", "Key": "file", **sse(KEY_A)},
    )
    stubber.add_response(
        "get_object",
        {"Body": AsyncBody(b"hello")},
        {"Bucket": "bucketa", "Key": "file", **sse(KEY_A)},
    )

    assert fs.cat("bucketa/file") == b"hello"
    assert ("read", "bucketa", "file", None) in provider.calls
    stubber.assert_no_pending_responses()


def test_sync_write_sends_object_key(same_thread_sync):
    provider = DictKeyProvider({("bucketb", "out"): KEY_B})
    fs, stubber, captured = make_stubbed_fs(provider)
    stubber.add_response(
        "put_object",
        {},
        {"Bucket": "bucketb", "Key": "out", "Body": b"data", **sse(KEY_B)},
    )

    fs.pipe_file("bucketb/out", b"data")
    assert ("write", "bucketb", "out", None) in provider.calls
    stubber.assert_no_pending_responses()


def test_sync_multipart_write_pins_single_context(same_thread_sync):
    provider = RotatingKeyProvider()
    fs, stubber, captured = make_stubbed_fs(provider)
    stubber.add_response("create_multipart_upload", {"UploadId": "upload-1"})
    stubber.add_response("upload_part", {"ETag": '"etag-1"'})
    stubber.add_response("upload_part", {"ETag": '"etag-2"'})
    stubber.add_response("complete_multipart_upload", {})

    with fs.open("bucketb/big", "wb", block_size=5 * 2**20) as f:
        f.write(b"x" * 5 * 2**20)
        f.write(b"y" * 2 * 2**20)

    # the write context was resolved exactly once for the whole upload
    assert provider.calls == ["write"]
    (create,) = params_of(captured, "CreateMultipartUpload")
    parts = params_of(captured, "UploadPart")
    assert len(parts) == 2
    keys = {create["SSECustomerKey"]} | {p["SSECustomerKey"] for p in parts}
    assert len(keys) == 1
    (complete,) = params_of(captured, "CompleteMultipartUpload")
    assert "SSECustomerKey" not in complete
    stubber.assert_no_pending_responses()


def test_sync_copy_uses_independent_source_and_dest_keys(same_thread_sync):
    provider = DictKeyProvider(
        {("bucketa", "src"): KEY_A, ("bucketb", "dst"): KEY_B}
    )
    fs, stubber, captured = make_stubbed_fs(provider)
    stubber.add_response(
        "head_object",
        {"ContentLength": 10, "ETag": '"etag"'},
        {"Bucket": "bucketa", "Key": "src", **sse(KEY_A)},
    )
    stubber.add_response(
        "copy_object",
        {},
        {
            "Bucket": "bucketb",
            "Key": "dst",
            "CopySource": {"Bucket": "bucketa", "Key": "src"},
            **sse(KEY_B),
            **copy_source_sse(KEY_A),
        },
    )

    fs.cp_file("bucketa/src", "bucketb/dst")
    assert ("copy_source", "bucketa", "src", None) in provider.calls
    assert ("copy_dest", "bucketb", "dst", None) in provider.calls
    stubber.assert_no_pending_responses()


def test_provider_failure_prevents_upload(same_thread_sync):
    class FailingProvider(SSECustomerKeyProvider):
        def resolve(self, operation, bucket, key, version_id=None):
            raise RuntimeError("no key available")

    fs, stubber, captured = make_stubbed_fs(FailingProvider())

    with pytest.raises(RuntimeError, match="no key available"):
        fs.pipe_file("bucketb/out", b"data")

    f = fs.open("bucketb/big", "wb", block_size=5 * 2**20)
    with pytest.raises(RuntimeError, match="no key available"):
        f.write(b"x" * 6 * 2**20)
    f.discard()

    # no upload was ever started
    assert captured == []


# ---------------------------------------------------------------------------
# Asynchronous entry points
# ---------------------------------------------------------------------------


def test_async_read_write_and_copy():
    provider = DictKeyProvider(
        {("bucketa", "src"): KEY_A, ("bucketb", "dst"): KEY_B}
    )
    fs, stubber, captured = make_stubbed_fs(provider, asynchronous=True)
    stubber.add_response(
        "head_object",
        {"ContentLength": 5, "ETag": '"etag"'},
        {"Bucket": "bucketa", "Key": "src", **sse(KEY_A)},
    )
    stubber.add_response(
        "get_object",
        {"Body": AsyncBody(b"hello")},
        {"Bucket": "bucketa", "Key": "src", **sse(KEY_A)},
    )
    stubber.add_response(
        "put_object",
        {},
        {"Bucket": "bucketb", "Key": "dst", "Body": b"data", **sse(KEY_B)},
    )
    stubber.add_response(
        "head_object",
        {"ContentLength": 5, "ETag": '"etag"'},
        {"Bucket": "bucketa", "Key": "src", **sse(KEY_A)},
    )
    stubber.add_response(
        "copy_object",
        {},
        {
            "Bucket": "bucketb",
            "Key": "dst",
            "CopySource": {"Bucket": "bucketa", "Key": "src"},
            **sse(KEY_B),
            **copy_source_sse(KEY_A),
        },
    )

    async def run():
        assert await fs._cat_file("bucketa/src") == b"hello"
        await fs._pipe_file("bucketb/dst", b"data")
        await fs._cp_file("bucketa/src", "bucketb/dst")

    asyncio.run(run())
    stubber.assert_no_pending_responses()


def test_async_callable_provider_returning_dict():
    calls = []

    def provider(operation, bucket, key, version_id=None):
        calls.append((operation, bucket, key, version_id))
        return sse(KEY_A)

    fs, stubber, captured = make_stubbed_fs(provider, asynchronous=True)
    stubber.add_response(
        "put_object",
        {},
        {"Bucket": "bucketa", "Key": "out", "Body": b"d", **sse(KEY_A)},
    )

    async def run():
        await fs._pipe_file("bucketa/out", b"d")

    asyncio.run(run())
    assert ("write", "bucketa", "out", None) in calls
    stubber.assert_no_pending_responses()


def test_async_provider_failure_prevents_upload():
    def provider(operation, bucket, key, version_id=None):
        raise RuntimeError("no key available")

    fs, stubber, captured = make_stubbed_fs(provider, asynchronous=True)

    async def run():
        with pytest.raises(RuntimeError, match="no key available"):
            await fs._pipe_file("bucketa/out", b"data")

    asyncio.run(run())
    assert captured == []
