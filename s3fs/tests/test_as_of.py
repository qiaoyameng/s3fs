"""Tests for the read-only as-of view (``S3FileSystem(as_of=...)``).

These tests stub ``_call_s3`` with a multi-page ``list_object_versions``
result, so no S3/moto server is needed.  The stubbed history contains a
file that was created, deleted and recreated; the page boundary splits
that key's versions, so a listing that drew conclusions from a single
page would get the first time point wrong.
"""

import asyncio
import copy
import datetime

import fsspec.asyn
import pytest

import s3fs.core
from s3fs import S3FileSystem


def _dt(hhmm):
    return datetime.datetime.fromisoformat("2024-01-01T%s:00+00:00" % hhmm)


# History of bucket "bucket" (all times UTC):
#   other.txt  created 09:00          (version o1, content b"other")
#   data.txt   created 10:00          (version v1, content b"first")
#   data.txt   deleted 11:00          (delete marker d1)
#   data.txt   recreated 12:00        (version v2, content b"second")
#   tie.txt    written twice at 13:00 (versions tv1/tv2, identical timestamps,
#                                      server returns tv2 first)
#
# Page 1 ends in the middle of data.txt's history: the recreate and the
# delete marker are on page 1, the original version only appears on page 2.
PAGE_1 = {
    "IsTruncated": True,
    "NextKeyMarker": "data.txt",
    "NextVersionIdMarker": "v2",
    "Versions": [
        {
            "Key": "data.txt",
            "VersionId": "v2",
            "LastModified": _dt("12:00"),
            "Size": 6,
            "ETag": '"e-v2"',
            "StorageClass": "STANDARD",
            "IsLatest": True,
        },
    ],
    "DeleteMarkers": [
        {
            "Key": "data.txt",
            "VersionId": "d1",
            "LastModified": _dt("11:00"),
            "IsLatest": False,
        },
    ],
}
PAGE_2 = {
    "IsTruncated": False,
    "Versions": [
        {
            "Key": "data.txt",
            "VersionId": "v1",
            "LastModified": _dt("10:00"),
            "Size": 5,
            "ETag": '"e-v1"',
            "StorageClass": "STANDARD",
            "IsLatest": False,
        },
        {
            "Key": "other.txt",
            "VersionId": "o1",
            "LastModified": _dt("09:00"),
            "Size": 5,
            "ETag": '"e-o1"',
            "StorageClass": "STANDARD",
            "IsLatest": True,
        },
        {
            "Key": "tie.txt",
            "VersionId": "tv2",
            "LastModified": _dt("13:00"),
            "Size": 3,
            "ETag": '"e-tv2"',
            "StorageClass": "STANDARD",
            "IsLatest": True,
        },
        {
            "Key": "tie.txt",
            "VersionId": "tv1",
            "LastModified": _dt("13:00"),
            "Size": 3,
            "ETag": '"e-tv1"',
            "StorageClass": "STANDARD",
            "IsLatest": False,
        },
    ],
    "DeleteMarkers": [],
}
PAGES = {(None, None): PAGE_1, ("data.txt", "v2"): PAGE_2}

CONTENTS = {
    "v1": b"first", "v2": b"second", "o1": b"other", "tv1": b"tie", "tv2": b"TIE"
}

T1 = "2024-01-01T10:30:00Z"  # after the first write, before the delete
T2 = "2024-01-01T11:30:00Z"  # after the delete, before the recreate
T3 = "2024-01-01T12:30:00Z"  # after the recreate


@pytest.fixture(autouse=True)
def in_thread_sync(monkeypatch):
    """Run coroutines on the calling thread instead of the fsspec IO thread.

    The sandboxed test environment cannot wake an event loop running in
    another thread, which is what fsspec's ``sync`` normally relies on.
    Running the coroutine directly is equivalent for these stubbed tests.
    """

    def fake_sync(loop, func, *args, timeout=None, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    monkeypatch.setattr(fsspec.asyn, "sync", fake_sync)
    monkeypatch.setattr(s3fs.core, "sync", fake_sync)


class _FakeBody:
    def __init__(self, data):
        self._data = data

    async def read(self):
        return self._data

    def close(self):
        pass


def make_fs(as_of):
    fs = S3FileSystem(
        anon=True, version_aware=True, as_of=as_of, skip_instance_cache=True
    )

    async def fake_call_s3(method, *args, **kwargs):
        if method == "list_object_versions":
            marker = (
                kwargs.get("KeyMarker") or None,
                kwargs.get("VersionIdMarker") or None,
            )
            page = copy.deepcopy(PAGES[marker])
            # emulate the server-side prefix filter
            prefix = kwargs.get("Prefix", "")
            for field in ("Versions", "DeleteMarkers"):
                page[field] = [
                    e for e in page.get(field, []) if e["Key"].startswith(prefix)
                ]
            return page
        if method == "get_object":
            version_id = kwargs["VersionId"]
            return {
                "Body": _FakeBody(CONTENTS[version_id]),
                "ETag": '"e-%s"' % version_id,
            }
        raise AssertionError("unexpected call: %s %r" % (method, kwargs))

    fs._call_s3 = fake_call_s3
    return fs


def test_as_of_requires_version_aware():
    with pytest.raises(ValueError, match="version_aware"):
        S3FileSystem(anon=True, as_of=T1, skip_instance_cache=True)


def test_as_of_normalization():
    assert make_fs(T1).as_of == _dt("10:30")
    # naive datetimes are assumed to be UTC
    assert make_fs(datetime.datetime(2024, 1, 1, 10, 30)).as_of == _dt("10:30")
    # other timezones are converted to UTC
    assert make_fs("2024-01-01T18:30:00+08:00").as_of == _dt("10:30")
    with pytest.raises((TypeError, ValueError)):
        make_fs("not-a-time")


def test_ls_at_three_moments():
    # t1: page 1 alone shows nothing usable for data.txt (the recreate and
    # the delete marker are both after the cutoff); only v1 on page 2 makes
    # the key visible.
    assert make_fs(T1).ls("bucket") == ["bucket/data.txt", "bucket/other.txt"]
    # t2: the delete marker is the newest entry at or before the cutoff
    assert make_fs(T2).ls("bucket") == ["bucket/other.txt"]
    # t3: the recreate is visible again
    assert make_fs(T3).ls("bucket") == ["bucket/data.txt", "bucket/other.txt"]


def test_info_and_exists_at_three_moments():
    fs1 = make_fs(T1)
    info = fs1.info("bucket/data.txt")
    assert info["VersionId"] == "v1"
    assert info["size"] == 5
    assert fs1.exists("bucket/data.txt")

    fs2 = make_fs(T2)
    with pytest.raises(FileNotFoundError):
        fs2.info("bucket/data.txt")
    assert not fs2.exists("bucket/data.txt")

    fs3 = make_fs(T3)
    assert fs3.info("bucket/data.txt")["VersionId"] == "v2"
    assert fs3.exists("bucket/data.txt")


def test_open_and_cat_at_three_moments():
    fs1 = make_fs(T1)
    assert fs1.cat_file("bucket/data.txt") == b"first"
    with fs1.open("bucket/data.txt", "rb") as f:
        assert f.read() == b"first"

    fs2 = make_fs(T2)
    with pytest.raises(FileNotFoundError):
        fs2.cat_file("bucket/data.txt")
    with pytest.raises(FileNotFoundError):
        fs2.open("bucket/data.txt", "rb")

    fs3 = make_fs(T3)
    assert fs3.cat_file("bucket/data.txt") == b"second"
    with fs3.open("bucket/data.txt", "rb") as f:
        assert f.read() == b"second"


def test_mapper_follows_view():
    mapper2 = make_fs(T2).get_mapper("bucket")
    assert set(mapper2) == {"other.txt"}
    assert "data.txt" not in mapper2
    assert mapper2["other.txt"] == b"other"
    with pytest.raises(KeyError):
        mapper2["data.txt"]

    mapper3 = make_fs(T3).get_mapper("bucket")
    assert set(mapper3) == {"data.txt", "other.txt"}
    assert mapper3["data.txt"] == b"second"


def test_view_is_read_only():
    fs = make_fs(T3)
    with pytest.raises(ValueError, match="read-only"):
        fs.open("bucket/new.txt", "wb")
    with pytest.raises(ValueError, match="read-only"):
        fs.pipe_file("bucket/new.txt", b"x")
    with pytest.raises(ValueError, match="read-only"):
        fs.touch("bucket/new.txt")
    with pytest.raises(ValueError, match="read-only"):
        fs.mkdir("bucket/newdir")
    with pytest.raises(ValueError, match="read-only"):
        fs.rm("bucket/other.txt")

    mapper = fs.get_mapper("bucket")
    with pytest.raises(ValueError, match="read-only"):
        mapper["new.txt"] = b"x"
    with pytest.raises(KeyError):
        # FSMap.__delitem__ re-raises backend errors as KeyError
        del mapper["other.txt"]


def test_same_timestamp_uses_server_order():
    # tv1 and tv2 share a LastModified; the server returns tv2 first
    # (newest), so the view resolves the key to tv2.
    fs = make_fs("2024-01-01T13:30:00Z")
    assert fs.info("bucket/tie.txt")["VersionId"] == "tv2"
    assert fs.cat_file("bucket/tie.txt") == b"TIE"


def test_raw_version_listing_still_available():
    # versions=True bypasses the as-of view and shows every version
    names = make_fs(T2).ls("bucket", versions=True)
    assert "bucket/data.txt?versionId=v1" in names
    assert "bucket/data.txt?versionId=v2" in names
    assert "bucket/other.txt?versionId=o1" in names
