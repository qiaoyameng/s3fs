"""Explicit checkpoint/restore support for resumable S3 multipart uploads.

A checkpoint (:class:`UploadCheckpoint`) records everything required to resume
the *same* S3 multipart upload after a process restart:

* the identity of the source file (absolute path, size and SHA256 digest);
* the partitioning scheme (``chunksize`` and the derived number of parts);
* the S3 destination (bucket/key), the S3 ``UploadId`` and the keyword
  arguments used to initiate the upload;
* the parts that have been confirmed on the client side (part number and
  ETag), as known at the time the checkpoint was last written.

On resume, the server is the source of truth: ``ListParts`` is consulted and
cross-checked against the checkpoint. Only parts that are genuinely missing
on the server are uploaded again. If the source content or the partitioning
parameters have changed, or the multipart upload no longer exists on the
server, resumption is refused with :class:`ResumableUploadError` rather than
risking assembling a mixed/hybrid object.
"""

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone

from botocore.exceptions import ClientError

from fsspec.callbacks import _DEFAULT_CALLBACK

logger = logging.getLogger("s3fs.checkpoint")

CHECKPOINT_VERSION = 1

MIN_PART_SIZE = 5 * 2**20  # every MPU part but the last must be >= 5 MiB
_READ_DIGEST_CHUNKSIZE = 2**20


class ResumableUploadError(Exception):
    """Base class for errors encountered while resuming an S3 upload."""


class CheckpointMismatchError(ResumableUploadError):
    """The checkpoint does not match the current source or upload parameters."""


class UploadExpiredError(ResumableUploadError):
    """The multipart upload referenced by a checkpoint no longer exists."""


class IncompleteUploadError(ResumableUploadError):
    """Completion could not be confirmed and the object is still missing."""


@dataclasses.dataclass
class UploadCheckpoint:
    """Persisted state for a single S3 multipart upload.

    Parameters
    ----------
    bucket, key:
        Destination S3 bucket and key.
    upload_id:
        The S3 multipart upload id.
    source_path:
        Absolute path of the local source file.
    source_size:
        Size of the source file in bytes.
    source_sha256:
        Hex SHA256 digest of the full source file.
    chunksize:
        Part size in bytes (the last part may be smaller).
    total_parts:
        Number of parts derived from ``source_size`` and ``chunksize``.
    parts:
        Mapping ``{part_number: etag}`` of parts the client has confirmed.
    kwargs:
        JSON-serialisable keyword arguments used to create the upload and
        required to complete it (e.g. ``ContentType``, ``ACL``).
    mode:
        Put mode, ``"overwrite"`` or ``"create"``.
    state:
        ``"pending"`` while uploading, ``"completed"`` after the object exists.
    created_at, updated_at:
        UTC ISO8601 timestamps.
    """

    bucket: str
    key: str
    upload_id: str
    source_path: str
    source_size: int
    source_sha256: str
    chunksize: int
    total_parts: int
    parts: dict
    kwargs: dict = dataclasses.field(default_factory=dict)
    mode: str = "overwrite"
    state: str = "pending"
    created_at: str = ""
    updated_at: str = ""
    version: int = CHECKPOINT_VERSION

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json(self, indent=None):
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data):
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise CheckpointMismatchError(
                f"Unknown checkpoint fields: {sorted(unknown)}"
            )
        if data.get("version") != CHECKPOINT_VERSION:
            raise CheckpointMismatchError(
                "Unsupported checkpoint version: %r" % data.get("version")
            )
        data["parts"] = {int(k): v for k, v in data.get("parts", {}).items()}
        return cls(**data)

    @classmethod
    def from_json(cls, text):
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            raise CheckpointMismatchError(f"Malformed checkpoint: {e}") from e
        return cls.from_dict(data)

    def save(self, path):
        """Atomically write this checkpoint to ``path``."""
        self.updated_at = _utc_now()
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".s3cp-", dir=directory)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(self.to_json())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            with _ignoring_oserror():
                os.remove(tmp)
            raise

    @classmethod
    def load(cls, path):
        try:
            with open(path, "r") as f:
                return cls.from_json(f.read())
        except FileNotFoundError:
            raise
        except OSError as e:
            raise CheckpointMismatchError(f"Cannot read checkpoint {path}: {e}") from e


class _ignoring_oserror:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, OSError)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    """Return the hex SHA256 digest of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_READ_DIGEST_CHUNKSIZE), b""):
            h.update(block)
    return h.hexdigest()


def expected_part_count(size, chunksize):
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    return max(1, (size + chunksize - 1) // chunksize)


def expected_part_size(size, chunksize, part_number):
    """Byte length that part ``part_number`` (1-based) must contain."""
    total_parts = expected_part_count(size, chunksize)
    if part_number < 1 or part_number > total_parts:
        raise ValueError(f"part_number out of range: {part_number}")
    start = (part_number - 1) * chunksize
    return min(chunksize, size - start)


def _ensure_jsonable(value, what="kwargs"):
    try:
        json.dumps(value)
    except (TypeError, ValueError) as e:
        raise CheckpointMismatchError(
            f"Checkpoint {what} must be JSON-serialisable: {e}"
        ) from e
    return value


def _describe_source(lpath):
    lpath = os.path.abspath(lpath)
    size = os.path.getsize(lpath)
    digest = file_sha256(lpath)
    return lpath, size, digest


def build_checkpoint(bucket, key, upload_id, lpath, chunksize, kwargs, mode, size=None,
                     digest=None, parts=None):
    """Construct an :class:`UploadCheckpoint` for a local source file."""
    lpath = os.path.abspath(lpath)
    if size is None or digest is None:
        size = os.path.getsize(lpath)
        digest = file_sha256(lpath)
    total_parts = expected_part_count(size, chunksize)
    now = _utc_now()
    return UploadCheckpoint(
        bucket=bucket,
        key=key,
        upload_id=upload_id,
        source_path=lpath,
        source_size=size,
        source_sha256=digest,
        chunksize=chunksize,
        total_parts=total_parts,
        parts=dict(parts or {}),
        kwargs=dict(_ensure_jsonable(kwargs or {})),
        mode=mode,
        created_at=now,
        updated_at=now,
    )


def validate_checkpoint(cp, lpath, bucket, key, chunksize=None, mode=None, kwargs=None):
    """Verify a checkpoint still describes ``lpath`` and the requested upload.

    Raises :class:`CheckpointMismatchError` if the source identity, chunking
    scheme, destination or upload parameters differ.
    """
    lpath = os.path.abspath(lpath)
    if cp.bucket != bucket or cp.key != key:
        raise CheckpointMismatchError(
            f"Checkpoint destination s3://{cp.bucket}/{cp.key} does not match "
            f"s3://{bucket}/{key}"
        )
    if cp.source_path != lpath:
        raise CheckpointMismatchError(
            f"Checkpoint source {cp.source_path!r} does not match {lpath!r}"
        )
    try:
        stat = os.stat(lpath)
    except OSError as e:
        raise CheckpointMismatchError(f"Cannot stat source {lpath}: {e}") from e
    if stat.st_size != cp.source_size:
        raise CheckpointMismatchError(
            f"Source size changed: {stat.st_size} != checkpoint {cp.source_size}"
        )
    if chunksize is not None and chunksize != cp.chunksize:
        raise CheckpointMismatchError(
            f"Chunksize changed: {chunksize} != checkpoint {cp.chunksize}"
        )
    if mode is not None and mode != cp.mode:
        raise CheckpointMismatchError(
            f"Upload mode changed: {mode!r} != checkpoint {cp.mode!r}"
        )
    kwargs = kwargs or {}
    for name, value in kwargs.items():
        if cp.kwargs.get(name) != value:
            raise CheckpointMismatchError(
                f"Upload parameter {name!r} differs from the checkpointed upload"
            )
    if expected_part_count(cp.source_size, cp.chunksize) != cp.total_parts:
        raise CheckpointMismatchError(
            "Checkpoint part count is inconsistent with size and chunksize"
        )
    digest = file_sha256(lpath)
    if digest != cp.source_sha256:
        raise CheckpointMismatchError(
            "Source content changed: SHA256 no longer matches the checkpoint"
        )


def _client_error_code(exc):
    if not isinstance(exc, ClientError):
        return None
    return exc.response.get("Error", {}).get("Code")


def _is_no_such_upload(exc):
    return _client_error_code(exc) in ("NoSuchUpload", "NoSuchUploadException")


def _is_not_found(exc):
    return _client_error_code(exc) in ("404", "NotFound", "NoSuchKey")


async def _list_all_parts(fs, bucket, key, upload_id):
    """Return ``{part_number: part_dict}`` from the server via ListParts."""
    server_parts = {}
    marker = None
    while True:
        kw = {"MaxParts": 1000}
        if marker is not None:
            kw["PartNumberMarker"] = marker
        try:
            resp = await fs._call_s3(
                "list_parts",
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                **kw,
            )
        except ClientError as e:
            if _is_no_such_upload(e):
                raise UploadExpiredError(
                    f"Multipart upload {upload_id} for s3://{bucket}/{key} "
                    "no longer exists"
                ) from e
            raise
        for part in resp.get("Parts", []) or []:
            server_parts[part["PartNumber"]] = part
        if resp.get("IsTruncated"):
            marker = resp.get("NextPartNumberMarker")
            if marker is None:
                break
        else:
            break
    return server_parts


def reconcile_parts(cp, server_parts):
    """Cross-check server ListParts output against the checkpoint.

    Returns ``{part_number: etag}`` for the parts that are present on the
    server with the exact expected size (and, when the client knew an ETag,
    with a matching ETag). Any inconsistency that could produce a hybrid
    object raises :class:`CheckpointMismatchError`.
    """
    confirmed = {}
    expected_numbers = set(range(1, cp.total_parts + 1))
    for number in sorted(expected_numbers):
        expected_size = expected_part_size(cp.source_size, cp.chunksize, number)
        server_part = server_parts.get(number)
        if server_part is None:
            continue
        if server_part.get("Size") != expected_size:
            raise CheckpointMismatchError(
                f"Server part {number} has size {server_part.get('Size')}, "
                f"expected {expected_size} for the checkpointed source"
            )
        client_etag = cp.parts.get(number)
        if client_etag is not None and server_part.get("ETag") != client_etag:
            raise CheckpointMismatchError(
                f"ETag mismatch for part {number}: server "
                f"{server_part.get('ETag')!r} != checkpoint {client_etag!r}"
            )
        confirmed[number] = server_part["ETag"]
    unexpected = sorted(set(server_parts) - expected_numbers)
    if unexpected:
        raise CheckpointMismatchError(
            f"Server reports unexpected part numbers {unexpected} for the "
            "checkpointed chunking scheme"
        )
    return confirmed


def _is_ambiguous_completion_error(exc):
    """True when a complete-MPU call may or may not have succeeded."""
    if isinstance(exc, ClientError):
        code = _client_error_code(exc)
        return code in (
            "InternalError",
            "ServiceUnavailable",
            "SlowDown",
            "RequestTimeout",
            "500",
            "503",
        )
    # Imported lazily to avoid a s3fs.core <-> s3fs.checkpoint import cycle.
    from s3fs.core import S3_RETRYABLE_ERRORS

    return isinstance(exc, S3_RETRYABLE_ERRORS)


async def _object_exists(fs, bucket, key):
    try:
        await fs._call_s3("head_object", Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if _is_not_found(e) or e.response.get("ResponseMetadata", {}).get(
            "HTTPStatusCode"
        ) == 404:
            return False
        raise


async def _already_completed(fs, cp, checkpoint_path):
    """Pre-flight status query for a pending checkpoint.

    A previous completion may have succeeded even though its response (and
    therefore the "completed" marker) was lost. Query status before doing
    any work; if the object exists the upload is done.
    """
    if await _object_exists(fs, cp.bucket, cp.key):
        cp.state = "completed"
        if checkpoint_path:
            cp.save(checkpoint_path)
        return True
    return False


async def complete_multipart_upload(fs, cp, match=None, max_ambiguity_retries=3):
    """Complete the MPU, querying status before retrying an uncertain result.

    When a ``complete_multipart_upload`` response is lost we must NOT simply
    resubmit work; we first check whether the object already exists and
    whether the upload is still in progress.
    """
    match = match or {}
    parts_payload = {
        "Parts": [
            {"PartNumber": number, "ETag": cp.parts[number]}
            for number in sorted(cp.parts)
        ]
    }
    for attempt in range(max_ambiguity_retries):
        try:
            return await fs._call_s3(
                "complete_multipart_upload",
                Bucket=cp.bucket,
                Key=cp.key,
                UploadId=cp.upload_id,
                MultipartUpload=parts_payload,
                **match,
            )
        except ClientError as e:
            if _is_no_such_upload(e):
                raise UploadExpiredError(
                    f"Multipart upload {cp.upload_id} expired before completion"
                ) from e
            if not _is_ambiguous_completion_error(e):
                raise
        except Exception as e:
            if not _is_ambiguous_completion_error(e):
                raise
            logger.debug("Uncertain complete_multipart_upload result: %s", e)

        # The response is uncertain: query state instead of blindly retrying.
        if await _object_exists(fs, cp.bucket, cp.key):
            logger.warning(
                "complete_multipart_upload response was lost for s3://%s/%s "
                "but the object already exists",
                cp.bucket,
                cp.key,
            )
            return None
        try:
            await _list_all_parts(fs, cp.bucket, cp.key, cp.upload_id)
        except UploadExpiredError:
            raise
        logger.debug(
            "Object s3://%s/%s not visible after uncertain completion "
            "(attempt %d); retrying complete",
            cp.bucket,
            cp.key,
            attempt + 1,
        )
        await asyncio.sleep(min(0.1 * 2**attempt, 1.0))
    raise IncompleteUploadError(
        f"Could not confirm completion of s3://{cp.bucket}/{cp.key}; "
        "the multipart upload is still open and can be resumed again"
    )


async def _upload_missing_parts(fs, cp, f0, callback, max_concurrency, checkpoint_path):
    concurrency = max_concurrency or fs.max_concurrency
    if concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    missing = sorted(set(range(1, cp.total_parts + 1)) - set(cp.parts))

    # Upload sequentially and persist every confirmed part, so a crash
    # while uploading part N can never discard an already-confirmed part.
    for number in missing:
        f0.seek((number - 1) * cp.chunksize)
        data = f0.read(expected_part_size(cp.source_size, cp.chunksize, number))
        out = await fs._call_s3(
            "upload_part",
            Bucket=cp.bucket,
            Key=cp.key,
            PartNumber=number,
            UploadId=cp.upload_id,
            Body=data,
            **cp.kwargs,
        )
        cp.parts[number] = out["ETag"]
        callback.relative_update(len(data))
        if checkpoint_path:
            cp.save(checkpoint_path)



async def create_and_save_checkpoint(
    fs,
    lpath,
    rpath,
    checkpoint_path,
    chunksize=None,
    mode="overwrite",
    callback=_DEFAULT_CALLBACK,
    **kwargs,
):
    """Explicit checkpoint export entry point.

    Identify the source, initiate the multipart upload and write its
    checkpoint without uploading any part data.
    """
    from s3fs.core import calculate_chunksize

    bucket, key, _ = fs.split_path(rpath)
    lpath, size, digest = _describe_source(lpath)
    chunksize = calculate_chunksize(size, chunksize=chunksize)
    if chunksize < MIN_PART_SIZE:
        raise ValueError(
            f"chunksize for a multipart upload must be >= {MIN_PART_SIZE} bytes"
        )
    create_kwargs = dict(kwargs)
    mpu = await fs._call_s3(
        "create_multipart_upload", Bucket=bucket, Key=key, **create_kwargs
    )
    cp = build_checkpoint(
        bucket,
        key,
        mpu["UploadId"],
        lpath,
        chunksize,
        create_kwargs,
        mode,
        size=size,
        digest=digest,
    )
    cp.save(checkpoint_path)
    callback.set_size(size)
    return cp


async def _run_to_completion(fs, cp, f0, checkpoint_path, callback, max_concurrency,
                             match):
    await _upload_missing_parts(
        fs, cp, f0, callback, max_concurrency, checkpoint_path
    )
    await complete_multipart_upload(fs, cp, match=match)
    cp.state = "completed"
    if checkpoint_path:
        cp.save(checkpoint_path)


async def resume_put_file(
    fs,
    checkpoint_path,
    lpath=None,
    rpath=None,
    callback=_DEFAULT_CALLBACK,
    max_concurrency=None,
    chunksize=None,
    mode=None,
    **kwargs,
):
    """Explicit checkpoint restore entry point.

    Validate the checkpoint against the local source, reconcile confirmed
    parts with the server via ``ListParts`` and upload only missing parts.
    """
    cp = UploadCheckpoint.load(checkpoint_path)
    lpath = os.path.abspath(lpath or cp.source_path)
    bucket, key = cp.bucket, cp.key
    if rpath is not None:
        req_bucket, req_key, _ = fs.split_path(rpath)
        if (req_bucket, req_key) != (bucket, key):
            raise CheckpointMismatchError(
                f"Checkpoint destination s3://{bucket}/{key} does not match "
                f"s3://{req_bucket}/{req_key}"
            )
    validate_checkpoint(
        cp,
        lpath,
        bucket,
        key,
        chunksize=chunksize,
        mode=mode,
        kwargs=kwargs,
    )
    if cp.state == "completed":
        if not await _object_exists(fs, bucket, key):
            raise CheckpointMismatchError(
                f"Checkpoint marks s3://{bucket}/{key} completed but the "
                "object is missing"
            )
        return
    if await _already_completed(fs, cp, checkpoint_path):
        fs.invalidate_cache(f"{bucket}/{key}")
        return
    callback.set_size(cp.source_size)
    server_parts = await _list_all_parts(fs, bucket, key, cp.upload_id)
    confirmed = reconcile_parts(cp, server_parts)
    if confirmed:
        callback.relative_update(
            sum(
                expected_part_size(cp.source_size, cp.chunksize, number)
                for number in confirmed
            )
        )
    cp.parts = confirmed
    cp.save(checkpoint_path)
    match = {"IfNoneMatch": "*"} if cp.mode == "create" else {}
    with open(lpath, "rb") as f0:
        await _run_to_completion(
            fs, cp, f0, checkpoint_path, callback, max_concurrency, match
        )
    fs.invalidate_cache(f"{bucket}/{key}")


async def put_file_with_checkpoint(
    fs,
    lpath,
    rpath,
    checkpoint_path,
    callback=_DEFAULT_CALLBACK,
    chunksize=None,
    max_concurrency=None,
    mode="overwrite",
    **kwargs,
):
    """Put ``lpath`` to ``rpath`` using a durable checkpoint.

    If ``checkpoint_path`` already exists, the same multipart upload is
    resumed; otherwise a new upload is initiated and the checkpoint is
    updated as parts are confirmed.
    """
    bucket, key, _ = fs.split_path(rpath)
    match = {"IfNoneMatch": "*"} if mode == "create" else {}
    if os.path.exists(checkpoint_path):
        cp = UploadCheckpoint.load(checkpoint_path)
        lpath = os.path.abspath(lpath)
        validate_checkpoint(
            cp,
            lpath,
            bucket,
            key,
            chunksize=chunksize,
            mode=mode,
            kwargs=kwargs,
        )
        if cp.state == "completed":
            if not await _object_exists(fs, bucket, key):
                raise CheckpointMismatchError(
                    f"Checkpoint marks s3://{bucket}/{key} completed but the "
                    "object is missing"
                )
            return
        if await _already_completed(fs, cp, checkpoint_path):
            return
        callback.set_size(cp.source_size)
        server_parts = await _list_all_parts(fs, bucket, key, cp.upload_id)
        confirmed = reconcile_parts(cp, server_parts)
        if confirmed:
            callback.relative_update(
                sum(
                    expected_part_size(cp.source_size, cp.chunksize, number)
                    for number in confirmed
                )
            )
        cp.parts = confirmed
        cp.save(checkpoint_path)
    else:
        from s3fs.core import calculate_chunksize

        lpath = os.path.abspath(lpath)
        size = os.path.getsize(lpath)
        digest = file_sha256(lpath)
        chunksize = calculate_chunksize(size, chunksize=chunksize)
        if chunksize < MIN_PART_SIZE:
            raise ValueError(
                f"chunksize for a multipart upload must be >= {MIN_PART_SIZE} "
                "bytes when using checkpoint_path"
            )
        create_kwargs = dict(kwargs)
        mpu = await fs._call_s3(
            "create_multipart_upload", Bucket=bucket, Key=key, **create_kwargs
        )
        cp = build_checkpoint(
            bucket,
            key,
            mpu["UploadId"],
            lpath,
            chunksize,
            create_kwargs,
            mode,
            size=size,
            digest=digest,
        )
        cp.save(checkpoint_path)
        callback.set_size(size)

    with open(lpath, "rb") as f0:
        await _run_to_completion(
            fs, cp, f0, checkpoint_path, callback, max_concurrency, match
        )
    fs.invalidate_cache(rpath)
