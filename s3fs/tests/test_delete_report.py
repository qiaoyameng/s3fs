import asyncio
import socket
from types import SimpleNamespace

import pytest

from s3fs.core import (
    DeleteReport,
    DeleteResult,
    DeleteStatus,
    S3FileSystem,
)


@pytest.fixture
def fs(monkeypatch):
    filesystem = S3FileSystem(
        anon=True, version_aware=True, asynchronous=True, loop=asyncio.new_event_loop()
    )
    filesystem._s3 = SimpleNamespace(delete_objects=None)

    async def set_session():
        return filesystem._s3

    monkeypatch.setattr(filesystem, "set_session", set_session)
    monkeypatch.setattr(
        filesystem,
        "_get_s3_method_kwargs",
        lambda method, kwargs, **request_kwargs: request_kwargs,
    )
    return filesystem


def test_report_over_limit_with_mixed_responses_and_retries(fs, monkeypatch):
    paths = [f"bucket/key/{index:04}?versionId=v{index}" for index in range(1003)]
    calls = []

    async def delete_objects(**kwargs):
        objects = list(kwargs["Delete"]["Objects"])
        calls.append(objects)
        assert kwargs["Delete"]["Quiet"] is False

        if len(calls) == 1:
            return {
                "Deleted": [
                    {
                        "Key": obj["Key"],
                        "VersionId": obj["VersionId"],
                    }
                    for index, obj in enumerate(objects)
                    if index not in {1, 2, 3}
                ],
                "Errors": [
                    {
                        "Key": objects[1]["Key"],
                        "VersionId": objects[1]["VersionId"],
                        "Code": "AccessDenied",
                        "Message": "Access denied",
                    },
                    {
                        "Key": objects[2]["Key"],
                        "VersionId": objects[2]["VersionId"],
                        "Code": "SlowDown",
                        "Message": "Slow down",
                    },
                    {
                        "Key": objects[3]["Key"],
                        "VersionId": objects[3]["VersionId"],
                        "Code": "Mystery",
                        "Message": "Unknown",
                    },
                ],
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }

        if len(calls) == 2:
            assert objects == [{"Key": "key/0002", "VersionId": "v2"}]
            return {
                "Deleted": [
                    {"Key": objects[0]["Key"], "VersionId": objects[0]["VersionId"]}
                ]
            }

        if len(calls) == 3:
            assert len(objects) == 3
            raise socket.timeout()

        if len(calls) == 4:
            return {
                "Deleted": [
                    {"Key": objects[0]["Key"], "VersionId": objects[0]["VersionId"]}
                ],
                "Errors": [
                    {
                        "Key": objects[1]["Key"],
                        "VersionId": objects[1]["VersionId"],
                        "Code": "AccessDenied",
                        "Message": "Access denied",
                    },
                    {
                        "Key": objects[2]["Key"],
                        "VersionId": objects[2]["VersionId"],
                        "Code": "SlowDown",
                        "Message": "Slow down",
                    },
                ],
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }

        assert objects == [{"Key": "key/1002", "VersionId": "v1002"}]
        raise socket.timeout()

    monkeypatch.setattr(fs._s3, "delete_objects", delete_objects)
    report = asyncio.run(fs._delete_paths_with_report(paths, retries=3))

    assert [len(objects) for objects in calls] == [1000, 1, 3, 3, 1]
    assert len(report.succeeded) == 999
    assert {
        result.key.rsplit("/", 1)[-1]
        for result in report.permanently_rejected
    } == {"0001", "1001"}
    assert [result.key for result in report.retryable_failures] == [
        "key/1002"
    ]
    assert [result.key.rsplit("/", 1)[-1] for result in report.unknown] == [
        "0003"
    ]
    assert report.retryable_paths == ["bucket/key/1002?versionId=v1002"]


@pytest.mark.parametrize("network_failure", [True, False])
def test_report_does_not_retry_unversioned_unknowns(
    fs, monkeypatch, network_failure
):
    calls = []

    async def delete_objects(**kwargs):
        calls.append(list(kwargs["Delete"]["Objects"]))
        if network_failure:
            raise socket.timeout()
        return {
            "Errors": [
                {
                    "Key": obj["Key"],
                    "Code": "SlowDown",
                    "Message": "slow",
                }
                for obj in kwargs["Delete"]["Objects"]
            ],
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

    monkeypatch.setattr(fs._s3, "delete_objects", delete_objects)
    report = asyncio.run(
        fs._delete_paths_with_report(["bucket/a", "bucket/b"], retries=5)
    )

    assert len(calls) == 1
    assert {result.key for result in report.unknown} == {"a", "b"}
    assert report.retryable_failures == []
    assert report.retryable_paths == []


def test_retryable_only_filters_versions(fs, monkeypatch):
    paths = ["bucket/old?versionId=v1", "bucket/new?versionId=v2"]
    calls = []

    async def delete_objects(**kwargs):
        calls.append(list(kwargs["Delete"]["Objects"]))
        if len(calls) == 1:
            return {
                "Errors": [
                    {
                        "Key": "old",
                        "VersionId": "v1",
                        "Code": "SlowDown",
                        "Message": "slow",
                    }
                ],
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }
        return {"Deleted": [{"Key": "old", "VersionId": "v1"}]}

    monkeypatch.setattr(fs._s3, "delete_objects", delete_objects)
    first = asyncio.run(fs._delete_paths_with_report(paths, retries=1))
    assert [result.key for result in first.retryable_failures] == ["old"]

    retry_report = DeleteReport()
    retry_report.add(
        DeleteResult(
            "bucket",
            "old",
            "v1",
            DeleteStatus.RETRYABLE_FAILURE,
            error_code="SlowDown",
        )
    )
    returned = asyncio.run(
        fs._delete_paths_with_report(
            paths,
            report=retry_report,
            retryable_only=True,
            retries=2,
        )
    )

    assert returned is retry_report
    assert calls[1] == [{"Key": "old", "VersionId": "v1"}]
    assert [result.key for result in returned.succeeded] == ["old"]
    assert returned.retryable_failures == []


def test_duplicate_identities_get_one_result(fs, monkeypatch):
    calls = []

    async def delete_objects(**kwargs):
        calls.append(list(kwargs["Delete"]["Objects"]))
        return {"Deleted": [{"Key": "same", "VersionId": "v1"}]}

    monkeypatch.setattr(fs._s3, "delete_objects", delete_objects)
    report = asyncio.run(
        fs._delete_paths_with_report(
            ["bucket/same?versionId=v1", "bucket/same?versionId=v1"]
        )
    )

    assert calls == [[{"Key": "same", "VersionId": "v1"}]]
    assert [result.key for result in report.succeeded] == ["same"]
    assert report.failed == []


def test_report_records_delete_marker_fields(fs, monkeypatch):
    async def delete_objects(**kwargs):
        return {
            "Deleted": [
                {"Key": "object", "VersionId": "v1"},
                {
                    "Key": "object",
                    "VersionId": "marker-v",
                    "DeleteMarker": True,
                    "DeleteMarkerVersionId": "marker-v",
                },
            ]
        }

    monkeypatch.setattr(fs._s3, "delete_objects", delete_objects)
    report = asyncio.run(
        fs._delete_paths_with_report(
            [
                "bucket/object?versionId=v1",
                "bucket/object?versionId=marker-v",
            ]
        )
    )

    assert [result.version_id for result in report.succeeded] == [
        "v1",
        "marker-v",
    ]
    assert [result.version_id for result in report.delete_markers] == [
        "marker-v"
    ]
    assert report.delete_markers[0].delete_marker_version_id == "marker-v"


def test_versioned_listing_interrupt_keeps_confirmed_results(
    fs, monkeypatch
):
    pages = [
        {
            "Versions": [{"Key": "confirmed", "VersionId": "v1"}],
            "DeleteMarkers": [],
            "IsTruncated": True,
            "NextKeyMarker": "next",
            "NextVersionIdMarker": "next-version",
        }
    ]
    deleted = []

    async def list_object_versions(**kwargs):
        if pages:
            return pages.pop(0)
        raise socket.timeout()

    async def delete_objects(**kwargs):
        deleted.extend(kwargs["Delete"]["Objects"])
        return {
            "Deleted": [
                {
                    "Key": obj["Key"],
                    "VersionId": obj["VersionId"],
                }
                for obj in kwargs["Delete"]["Objects"]
            ]
        }

    async def list_pages(method, **kwargs):
        yield await list_object_versions()
        yield await list_object_versions()

    monkeypatch.setattr(fs, "_list_pages", list_pages)
    monkeypatch.setattr(fs._s3, "delete_objects", delete_objects)

    async def run():
        return await fs._rm_versioned_bucket_contents(
            "bucket", report=DeleteReport()
        )

    report = asyncio.run(run())

    assert report.complete is False
    assert report.listing_complete is False
    assert isinstance(report.errors[0], socket.timeout)
    assert [result.path for result in report.succeeded] == [
        "bucket/confirmed?versionId=v1"
    ]
    assert deleted == [{"Key": "confirmed", "VersionId": "v1"}]
