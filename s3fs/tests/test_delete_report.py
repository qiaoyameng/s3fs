import asyncio

import pytest
from botocore.exceptions import ClientError

from s3fs import S3FileSystem, S3ObjectIdentifier


class FakeS3Client:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    async def delete_objects(self, **kwargs):
        self.calls.append(kwargs)
        return await self.handler(kwargs)


@pytest.fixture
def fs(monkeypatch):
    filesystem = S3FileSystem(
        anon=False,
        asynchronous=True,
        client_kwargs={"endpoint_url": "http://localhost"},
    )

    async def set_session(refresh=False, kwargs=None):
        return None

    async def get_s3(bucket=None):
        return client

    monkeypatch.setattr(filesystem, "set_session", set_session)
    monkeypatch.setattr(filesystem, "get_s3", get_s3)
    client = FakeS3Client(None)
    monkeypatch.setattr(filesystem, "_s3", client)
    monkeypatch.setattr(
        filesystem,
        "_get_s3_method_kwargs",
        lambda method, *args, **kwargs: kwargs,
    )
    filesystem._fake_client = client
    return filesystem


def run(coro):
    return asyncio.run(coro)


def test_delete_report_splits_large_batch_and_classifies_mixed_errors(fs):
    total = 1001
    identifiers = [
        S3ObjectIdentifier(bucket="bucket", key=f"key-{index:04}")
        for index in range(total)
    ]

    async def handler(kwargs):
        objects = kwargs["Delete"]["Objects"]
        assert kwargs["Delete"]["Quiet"] is False
        assert len(objects) <= 1000
        deleted = [
            {"Key": obj["Key"], "VersionId": "null"}
            for obj in objects
            if not (len(objects) == 1 and obj["Key"] == "key-1000")
        ]
        if len(objects) == 1:
            errors = [
                {
                    "Key": objects[0]["Key"],
                    "Code": "AccessDenied",
                    "Message": "Access denied",
                }
            ]
        else:
            errors = []
        return {
            "Deleted": deleted,
            "Errors": errors,
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

    fs._fake_client.handler = handler
    report = run(fs._delete_identifiers(identifiers))

    assert len(fs._fake_client.calls) == 2
    assert sorted(len(call["Delete"]["Objects"]) for call in fs._fake_client.calls) == [
        1,
        1000,
    ]
    assert len(report.succeeded) == total - 1
    assert [entry.key for entry in report.permanently_rejected] == ["key-1000"]
    assert report.retryable_failures == []
    assert report.unknown == []
    assert report.ok is False


def test_delete_report_keeps_first_batch_after_network_failure(fs, monkeypatch):
    monkeypatch.setattr(type(fs), "retries", 1)
    identifiers = [
        S3ObjectIdentifier(bucket="bucket", key=f"key-{index:04}")
        for index in range(1001)
    ]

    async def handler(kwargs):
        objects = kwargs["Delete"]["Objects"]
        if len(objects) == 1000:
            return {
                "Deleted": [
                    {"Key": obj["Key"], "VersionId": "null"} for obj in objects
                ],
                "Errors": [],
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }
        raise TimeoutError("connection lost")

    fs._fake_client.handler = handler
    report = run(fs._delete_identifiers(identifiers))

    assert len(report.succeeded) == 1000
    assert [entry.key for entry in report.unknown] == ["key-1000"]
    assert report.retryable_failures == []
    assert report.permanently_rejected == []
    assert [entry.code for entry in report.unknown] == ["TimeoutError"]


def test_delete_report_keeps_confirmed_results_after_retryable_failure(fs):
    identifiers = [
        S3ObjectIdentifier(bucket="bucket", key="good"),
        S3ObjectIdentifier(bucket="bucket", key="denied"),
        S3ObjectIdentifier(bucket="bucket", key="version", version_id="v1"),
        S3ObjectIdentifier(bucket="bucket", key="ambiguous"),
    ]
    attempts = {"count": 0}

    async def handler(kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return {
                "Deleted": [{"Key": "good", "VersionId": "null"}],
                "Errors": [
                    {
                        "Key": "denied",
                        "Code": "AccessDenied",
                        "Message": "Access denied",
                    },
                    {
                        "Key": "version",
                        "VersionId": "v1",
                        "Code": "SlowDown",
                        "Message": "Slow down",
                    },
                    {
                        "Key": "ambiguous",
                        "Code": "SlowDown",
                        "Message": "Slow down",
                    },
                ],
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }
        objects = kwargs["Delete"]["Objects"]
        assert objects == [{"Key": "version", "VersionId": "v1"}]
        assert all("VersionId" in obj for obj in objects)
        return {
            "Deleted": [{"Key": "version", "VersionId": "v1"}],
            "Errors": [],
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

    fs._fake_client.handler = handler
    report = run(fs._delete_identifiers(identifiers))

    assert {entry.key for entry in report.succeeded} == {"good"}
    assert [entry.key for entry in report.permanently_rejected] == ["denied"]
    assert [entry.key for entry in report.retryable_failures] == ["version"]
    assert [entry.key for entry in report.unknown] == ["ambiguous"]
    assert [(obj.key, obj.version_id) for obj in report.retryable_identifiers] == [
        ("version", "v1")
    ]

    retry_report = run(fs._retry_deletions(report))
    assert {entry.key for entry in report.succeeded} == {"good", "version"}
    assert [entry.key for entry in report.retryable_failures] == []
    assert [entry.key for entry in retry_report.succeeded] == ["version"]
    assert len(fs._fake_client.calls) == 2


def test_retry_can_move_version_failure_to_permanent_rejection(fs):
    async def handler(kwargs):
        if len(fs._fake_client.calls) == 1:
            return {
                "Errors": [
                    {
                        "Key": "version",
                        "VersionId": "v1",
                        "Code": "SlowDown",
                        "Message": "Slow down",
                    }
                ],
                "Deleted": [],
                "ResponseMetadata": {"HTTPStatusCode": 200},
            }
        return {
            "Errors": [
                {
                    "Key": "version",
                    "VersionId": "v1",
                    "Code": "AccessDenied",
                    "Message": "Access denied",
                }
            ],
            "Deleted": [],
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

    fs._fake_client.handler = handler
    identifiers = [S3ObjectIdentifier("bucket", "version", "v1")]
    report = run(fs._delete_identifiers(identifiers))
    assert [entry.key for entry in report.retryable_failures] == ["version"]

    run(fs._retry_deletions(report))
    assert report.retryable_failures == []
    assert [entry.key for entry in report.permanently_rejected] == ["version"]


def test_delete_report_separates_delete_markers(fs):
    identifiers = [
        S3ObjectIdentifier(bucket="bucket", key="object"),
        S3ObjectIdentifier(
            bucket="bucket", key="marker", version_id="marker-v", is_delete_marker=True
        ),
    ]

    async def handler(kwargs):
        return {
            "Deleted": [
                {
                    "Key": "object",
                    "DeleteMarker": "true",
                    "DeleteMarkerVersionId": "new-marker-v",
                },
                {
                    "Key": "marker",
                    "VersionId": "marker-v",
                    "DeleteMarker": "true",
                },
            ],
            "Errors": [],
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

    fs._fake_client.handler = handler
    report = run(fs._delete_identifiers(identifiers))

    assert {entry.key for entry in report.succeeded} == {"object", "marker"}
    assert {entry.key for entry in report.delete_markers} == {"object", "marker"}
    assert report.delete_markers[0].delete_marker_version_id == "new-marker-v"
    assert report.delete_markers[1].identifier.is_delete_marker is True


def test_versioned_bucket_report_keeps_deleted_pages_on_listing_error(fs, monkeypatch):
    pages = [
        {
            "Versions": [{"Key": "old", "VersionId": "v1"}],
            "DeleteMarkers": [{"Key": "old", "VersionId": "marker-v1"}],
            "IsTruncated": True,
            "NextKeyMarker": "old",
            "NextVersionIdMarker": "marker-v1",
        }
    ]

    async def fake_list_pages(method, **kwargs):
        assert method == "list_object_versions"
        for page in pages:
            yield page
        raise TimeoutError("listing interrupted")

    async def delete_handler(kwargs):
        objects = kwargs["Delete"]["Objects"]
        return {
            "Deleted": [
                {"Key": obj["Key"], "VersionId": obj["VersionId"]}
                for obj in objects
            ],
            "Errors": [],
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

    fs._fake_client.handler = delete_handler
    monkeypatch.setattr(fs, "_list_pages", fake_list_pages)

    report = run(fs._rm_versioned_bucket_contents_report("bucket"))

    assert {entry.key for entry in report.succeeded} == {"old"}
    assert len(report.succeeded) == 2
    assert {entry.identifier.version_id for entry in report.succeeded} == {
        "v1",
        "marker-v1",
    }
    assert {entry.identifier.version_id for entry in report.delete_markers} == {
        "marker-v1"
    }
    assert report.operation_errors[0].stage == "list_object_versions"
    assert report.operation_errors[0].code == "TimeoutError"


def test_request_failure_classifies_unversioned_items_as_unknown(fs):
    identifiers = [
        S3ObjectIdentifier(bucket="bucket", key="unversioned"),
        S3ObjectIdentifier(bucket="bucket", key="versioned", version_id="v1"),
    ]

    async def handler(kwargs):
        raise ClientError(
            {
                "Error": {"Code": "SlowDown", "Message": "Slow down"},
                "ResponseMetadata": {"HTTPStatusCode": 503},
            },
            "DeleteObjects",
        )

    fs._fake_client.handler = handler
    report = run(fs._delete_identifiers(identifiers))

    assert [entry.key for entry in report.retryable_failures] == ["versioned"]
    assert [entry.key for entry in report.unknown] == ["unversioned"]
    assert report.retryable_identifiers[0].version_id == "v1"


def test_rm_returns_structured_report(fs, monkeypatch):
    paths = ["bucket/a", "bucket/b"]

    async def expand_path(path, recursive=False):
        return paths

    async def handler(kwargs):
        return {
            "Deleted": [
                {"Key": "a", "VersionId": "null"},
                {"Key": "b", "VersionId": "null"},
            ],
            "Errors": [],
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

    async def rmdir(path):
        return None

    monkeypatch.setattr(fs, "_expand_path", expand_path)
    monkeypatch.setattr(fs, "_rmdir", rmdir)
    fs._fake_client.handler = handler

    report = asyncio.run(fs._rm(paths, return_report=True))

    assert report.ok
    assert report.deleted_paths == paths
    assert fs._fake_client.calls[0]["Delete"]["Quiet"] is False
