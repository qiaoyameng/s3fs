"""Tests for read-only as_of views on versioned buckets.

These tests use a stubbed multi-page ``list_object_versions`` response (not
moto) so that the page boundary deliberately splits the version history of a
single key: the file ``dir/recreated`` is written (v1), deleted (delete
marker) and then recreated (v2), and its oldest version only appears on the
second page. A naive implementation that finalizes a key at a page boundary
would draw the wrong conclusion about its visibility.
"""

import asyncio
import datetime
import errno

import fsspec.asyn
import pytest

import s3fs.core
from s3fs.core import S3FileSystem

UTC = datetime.timezone.utc

T0 = datetime.datetime(2024, 1, 1, tzinfo=UTC)  # "kept" and "nested" written
T1 = datetime.datetime(2024, 1, 2, tzinfo=UTC)  # recreated v1, gone v1
T2 = datetime.datetime(2024, 1, 3, tzinfo=UTC)  # delete markers
T3 = datetime.datetime(2024, 1, 4, tzinfo=UTC)  # recreated v2

# the three points in time under test
BEFORE_DELETE = datetime.datetime(2024, 1, 2, 12, tzinfo=UTC)
WHILE_DELETED = datetime.datetime(2024, 1, 3, 12, tzinfo=UTC)
AFTER_RECREATE = datetime.datetime(2024, 1, 4, 12, tzinfo=UTC)

BUCKET = "test-bucket"

CONTENTS = {
    ("dir/kept", "v-kept"): b"kept-contents",
    ("dir/recreated", "v1-rec"): b"first-version",
    ("dir/recreated", "v2-rec"): b"second-version",
    ("dir/gone", "v-gone"): b"gone-contents",
    ("dir/sub/nested", "v-nested"): b"nested-contents",
}

LAST_MODIFIED = {
    ("dir/kept", "v-kept"): T0,
    ("dir/recreated", "v1-rec"): T1,
    ("dir/recreated", "v2-rec"): T3,
    ("dir/gone", "v-gone"): T1,
    ("dir/sub/nested", "v-nested"): T0,
}


def _version(key, vid, ts):
    return {
        "Key": key,
        "VersionId": vid,
        "IsLatest": False,
        "LastModified": ts,
        "Size": len(CONTENTS[(key, vid)]),
        "ETag": '"etag-%s"' % vid,
        "StorageClass": "STANDARD",
    }


def _marker(key, vid, ts):
    return {
        "Key": key,
        "VersionId": vid,
        "IsLatest": False,
        "LastModified": ts,
    }


# The full version-listing stream in server order: key ascending, then
# LastModified descending within each key. Note that the history of
# "dir/recreated" (v2, delete marker, v1) straddles the page boundary when
# PAGE_SIZE is 5: page 1 ends right after its delete marker.
STREAM = [
    ("marker", _marker("dir/gone", "d-gone", T2)),
    ("version", _version("dir/gone", "v-gone", T1)),
    ("version", _version("dir/kept", "v-kept", T0)),
    ("version", _version("dir/recreated", "v2-rec", T3)),
    ("marker", _marker("dir/recreated", "d-rec", T2)),
    ("version", _version("dir/recreated", "v1-rec", T1)),
    ("version", _version("dir/sub/nested", "v-nested", T0)),
]

PAGE_SIZE = 5


def _rollup(stream, prefix, delimiter):
    """Apply prefix/delimiter semantics like S3 does."""
    out = []
    seen_prefixes = set()
    for kind, entry in stream:
        key = entry["Key"]
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        if delimiter and delimiter in rest:
            common = prefix + rest.split(delimiter, 1)[0] + delimiter
            if common not in seen_prefixes:
                seen_prefixes.add(common)
                out.append(("prefix", {"Prefix": common}))
            continue
        out.append((kind, entry))
    return out


def _list_object_versions(Prefix="", Delimiter="", KeyMarker="",
                          VersionIdMarker="", **kwargs):
    stream = _rollup(STREAM, Prefix, Delimiter)
    if KeyMarker:
        pos = next(
            i
            for i, (kind, entry) in enumerate(stream)
            if entry.get("Key", entry.get("Prefix")) == KeyMarker
            and (not VersionIdMarker
                 or entry.get("VersionId") == VersionIdMarker)
        )
        stream = stream[pos + 1:]
    page, rest = stream[:PAGE_SIZE], stream[PAGE_SIZE:]
    out = {
        "Versions": [e for kind, e in page if kind == "version"],
        "DeleteMarkers": [e for kind, e in page if kind == "marker"],
        "CommonPrefixes": [e for kind, e in page if kind == "prefix"],
        "IsTruncated": bool(rest),
    }
    if rest:
        last = page[-1][1]
        out["NextKeyMarker"] = last.get("Key", last.get("Prefix"))
        out["NextVersionIdMarker"] = last.get("VersionId", "")
    return out


class _FakeBody:
    def __init__(self, data):
        self._data = data

    async def read(self, n=-1):
        return self._data

    def close(self):
        pass


def make_stubbed_fs():
    """A version-aware filesystem whose S3 API is served by the stub above."""
    fs = S3FileSystem(version_aware=True, anon=True)

    async def fake_call_s3(method, *args, **kwargs):
        if method == "list_object_versions":
            return _list_object_versions(**kwargs)
        if method == "head_object":
            ident = (kwargs["Key"], kwargs.get("VersionId"))
            data = CONTENTS[ident]
            return {
                "ContentLength": len(data),
                "ETag": '"etag-%s"' % ident[1],
                "LastModified": LAST_MODIFIED[ident],
                "VersionId": ident[1],
                "ContentType": "application/octet-stream",
            }
        if method == "get_object":
            data = CONTENTS[(kwargs["Key"], kwargs.get("VersionId"))]
            return {"Body": _FakeBody(data), "ETag": '"etag"'}
        raise AssertionError("unexpected S3 call: %s %s" % (method, kwargs))

    fs._call_s3 = fake_call_s3
    return fs


@pytest.fixture(autouse=True)
def same_thread_sync(monkeypatch):
    """Run coroutines on a fresh same-thread loop.

    The stub does not care which loop it runs on, and this keeps the tests
    independent of how the fsspec IO loop thread is scheduled.
    """

    def _sync(loop, func, *args, timeout=None, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    monkeypatch.setattr(fsspec.asyn, "sync", _sync)
    monkeypatch.setattr(s3fs.core, "sync", _sync)


@pytest.fixture
def fs():
    return make_stubbed_fs()


def test_as_of_requires_version_aware():
    plain = S3FileSystem(anon=True)
    with pytest.raises(ValueError, match="version"):
        plain.as_of(BEFORE_DELETE)


def test_as_of_ls_three_points_in_time(fs):
    before = fs.as_of(BEFORE_DELETE)
    during = fs.as_of(WHILE_DELETED)
    after = fs.as_of(AFTER_RECREATE)

    # before the delete: the original version is listed
    assert before.ls(f"{BUCKET}/dir") == [
        f"{BUCKET}/dir/gone",
        f"{BUCKET}/dir/kept",
        f"{BUCKET}/dir/recreated",
        f"{BUCKET}/dir/sub",
    ]
    # while deleted: both deleted keys disappear, even though the delete
    # marker and the older version sit on different response pages
    assert during.ls(f"{BUCKET}/dir") == [
        f"{BUCKET}/dir/kept",
        f"{BUCKET}/dir/sub",
    ]
    # after the recreate: the file is back, "gone" is not
    assert after.ls(f"{BUCKET}/dir") == [
        f"{BUCKET}/dir/kept",
        f"{BUCKET}/dir/recreated",
        f"{BUCKET}/dir/sub",
    ]


def test_as_of_ls_detail_selects_version(fs):
    before = fs.as_of(BEFORE_DELETE)
    after = fs.as_of(AFTER_RECREATE)

    (entry,) = [e for e in before.ls(f"{BUCKET}/dir", detail=True)
                if e["name"] == f"{BUCKET}/dir/recreated"]
    assert entry["VersionId"] == "v1-rec"
    assert entry["size"] == len(b"first-version")

    (entry,) = [e for e in after.ls(f"{BUCKET}/dir", detail=True)
                if e["name"] == f"{BUCKET}/dir/recreated"]
    assert entry["VersionId"] == "v2-rec"
    assert entry["size"] == len(b"second-version")


def test_as_of_info(fs):
    before = fs.as_of(BEFORE_DELETE)
    during = fs.as_of(WHILE_DELETED)
    after = fs.as_of(AFTER_RECREATE)

    info = before.info(f"{BUCKET}/dir/recreated")
    assert info["type"] == "file"
    assert info["VersionId"] == "v1-rec"
    assert info["size"] == len(b"first-version")

    info = after.info(f"{BUCKET}/dir/recreated")
    assert info["VersionId"] == "v2-rec"
    assert info["size"] == len(b"second-version")

    with pytest.raises(FileNotFoundError):
        during.info(f"{BUCKET}/dir/recreated")
    assert not during.exists(f"{BUCKET}/dir/recreated")
    assert not during.exists(f"{BUCKET}/dir/gone")
    assert before.exists(f"{BUCKET}/dir/gone")

    # prefixes are still directories at every point in time
    for view in (before, during, after):
        assert view.info(f"{BUCKET}/dir/sub")["type"] == "directory"
        assert view.isdir(f"{BUCKET}/dir/sub")


def test_as_of_open_reads_selected_version(fs):
    before = fs.as_of(BEFORE_DELETE)
    during = fs.as_of(WHILE_DELETED)
    after = fs.as_of(AFTER_RECREATE)

    with before.open(f"{BUCKET}/dir/recreated") as f:
        assert f.read() == b"first-version"
    with after.open(f"{BUCKET}/dir/recreated") as f:
        assert f.read() == b"second-version"
    with pytest.raises(FileNotFoundError):
        during.open(f"{BUCKET}/dir/recreated")

    # cat follows the same selection (this is what FSMap reads use)
    assert before.cat(f"{BUCKET}/dir/recreated") == b"first-version"
    assert after.cat(f"{BUCKET}/dir/recreated") == b"second-version"
    with pytest.raises(FileNotFoundError):
        during.cat(f"{BUCKET}/dir/recreated")


def test_as_of_mapper(fs):
    before = fs.as_of(BEFORE_DELETE)
    during = fs.as_of(WHILE_DELETED)
    after = fs.as_of(AFTER_RECREATE)

    mapper_before = before.get_mapper(f"{BUCKET}/dir")
    mapper_during = during.get_mapper(f"{BUCKET}/dir")
    mapper_after = after.get_mapper(f"{BUCKET}/dir")

    assert mapper_before["recreated"] == b"first-version"
    assert mapper_after["recreated"] == b"second-version"
    with pytest.raises(KeyError):
        mapper_during["recreated"]

    assert "recreated" in mapper_before
    assert "recreated" not in mapper_during
    assert "gone" not in mapper_after
    assert set(mapper_after) == {"kept", "recreated", "sub/nested"}

    # the view is read-only through the mapping as well
    with pytest.raises(OSError, match="read-only"):
        mapper_after["new-key"] = b"nope"
    # FSMap.__delitem__ wraps all errors in KeyError; the read-only
    # error is the cause
    with pytest.raises(KeyError) as excinfo:
        del mapper_after["kept"]
    assert isinstance(excinfo.value.__cause__, OSError)
    assert excinfo.value.__cause__.errno == errno.EROFS


def test_as_of_view_is_read_only(fs):
    view = fs.as_of(AFTER_RECREATE)
    path = f"{BUCKET}/dir/kept"

    for call in (
        lambda: view.touch(path),
        lambda: view.rm(path),
        lambda: view.rm_file(path),
        lambda: view.pipe_file(path, b"data"),
        lambda: view.put_file(__file__, path),
        lambda: view.mkdir(f"{BUCKET}/newdir"),
        lambda: view.makedirs(f"{BUCKET}/newdir"),
        lambda: view.rmdir(BUCKET),
        lambda: view.cp_file(path, f"{BUCKET}/dir/copy"),
        lambda: view.setxattr(path, attr="value"),
        lambda: view.chmod(path, "private"),
    ):
        with pytest.raises(OSError) as excinfo:
            call()
        assert excinfo.value.errno == errno.EROFS

    with pytest.raises(OSError) as excinfo:
        view.open(path, "wb")
    assert excinfo.value.errno == errno.EROFS

    # the live filesystem is not affected by the view
    assert fs._as_of is None
    fs._check_writable()  # does not raise


def test_as_of_timestamp_forms(fs):
    as_datetime = fs.as_of(BEFORE_DELETE)
    as_string = fs.as_of("2024-01-02T12:00:00+00:00")
    as_naive = fs.as_of(datetime.datetime(2024, 1, 2, 12, 0, 0))
    assert as_datetime._as_of == as_string._as_of == as_naive._as_of
    with pytest.raises(TypeError):
        fs.as_of(12345)


def test_as_of_same_timestamp_tiebreak(fs):
    # a version and a delete marker sharing an identical (key, LastModified):
    # the server does not expose their relative order, and the documented
    # tie-break treats the version as newer, so the file stays visible
    view = fs.as_of(T2)
    stream_backup = STREAM[:]
    # "dir/tied" sorts after "dir/sub/nested", so it goes at the very end;
    # the version and the delete marker share the exact same timestamp
    CONTENTS[("dir/tied", "v-tied")] = b"tied"
    LAST_MODIFIED[("dir/tied", "v-tied")] = T2
    STREAM.append(("version", _version("dir/tied", "v-tied", T2)))
    STREAM.append(("marker", _marker("dir/tied", "d-tied", T2)))
    try:
        assert view.exists(f"{BUCKET}/dir/tied")
    finally:
        STREAM[:] = stream_backup
        del CONTENTS[("dir/tied", "v-tied")]
        del LAST_MODIFIED[("dir/tied", "v-tied")]


def test_as_of_listing_cache_isolated(fs):
    view = fs.as_of(WHILE_DELETED)
    # listing through the view must not populate the live fs cache
    assert view.ls(f"{BUCKET}/dir") == [f"{BUCKET}/dir/kept", f"{BUCKET}/dir/sub"]
    assert f"{BUCKET}/dir" not in fs.dircache
    assert f"{BUCKET}/dir" in view.dircache
