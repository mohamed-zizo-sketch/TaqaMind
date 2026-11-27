"""Resource limits for ML CSV reading and verified partition snapshots."""

from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import pytest

from taqamind.core.automl.data_access import _read_partition_bytes, read_partition_dataframe
from taqamind.core.ingestion.resource_policy import CsvResourcePolicy
from taqamind.core.ingestion.safe_csv import SafeCsvError, SafeCsvReader
from taqamind.core.interfaces.data_access import DataAccessDeniedError
from taqamind.core.pipeline import CorePlatformServices


def test_partition_reader_preserves_stricter_custom_limits():
    custom = SafeCsvReader(delimiter=";", max_rows=1)
    policy = CsvResourcePolicy(max_bytes=100, max_rows=2)
    assert read_partition_dataframe(BytesIO(b"a;b\n01;NA\n"), custom, resource_policy=policy).iloc[
        0
    ].tolist() == ["01", "NA"]
    with pytest.raises(SafeCsvError) as error:
        read_partition_dataframe(BytesIO(b"a;b\n1;2\n3;4\n"), custom, resource_policy=policy)
    assert error.value.code == "CSV_ROW_LIMIT_EXCEEDED"


def test_partition_reader_bounded():
    stream = BytesIO(b"a,b\n1,2\n" + b"x" * 100)
    with pytest.raises(SafeCsvError) as error:
        read_partition_dataframe(stream, resource_policy=CsvResourcePolicy(max_bytes=20))
    assert error.value.code == "CSV_BYTE_LIMIT_EXCEEDED"
    assert stream.tell() == 21


@pytest.mark.parametrize("actual", [b"1234", b"123456", b"54321"])
def test_partition_snapshot_rejects_size_or_hash_mismatch(tmp_path, actual):
    path = tmp_path / "partition.csv"
    path.write_bytes(actual)
    reference = SimpleNamespace(size_bytes=5, content_sha256=sha256(b"12345").hexdigest())
    with pytest.raises(DataAccessDeniedError):
        _read_partition_bytes(path, reference, CsvResourcePolicy(max_bytes=5))


def test_partition_snapshot_exact_limit(tmp_path):
    path = tmp_path / "partition.csv"
    payload = b"a\n1\n"
    path.write_bytes(payload)
    reference = SimpleNamespace(size_bytes=len(payload), content_sha256=sha256(payload).hexdigest())
    with _read_partition_bytes(
        path, reference, CsvResourcePolicy(max_bytes=len(payload))
    ) as stream:
        assert stream.read() == payload


def test_partition_reference_limit_before_open(tmp_path):
    reference = SimpleNamespace(size_bytes=6, content_sha256="unused")
    with pytest.raises(DataAccessDeniedError, match="byte limit"):
        _read_partition_bytes(tmp_path / "missing.csv", reference, CsvResourcePolicy(max_bytes=5))


def test_core_policy_injection(tmp_path):
    raw = CsvResourcePolicy(max_bytes=100)
    internal = CsvResourcePolicy(max_bytes=200)
    core = CorePlatformServices.create_local(
        tmp_path, raw_csv_policy=raw, internal_csv_policy=internal
    )
    assert core.ingestion._resource_policy == raw
    assert core.internal_csv_policy == internal
    for component in (
        core.profiler,
        core.mapper,
        core.validator,
        core.cleaner,
        core.feature_engineer,
        core.membership_builder,
        core.readiness_checker,
        core.splitter,
        core.leakage_validator,
        core.development_data_provider,
        core.protected_test_data_provider,
        core.trainer,
        core.prediction_engine,
    ):
        assert component._resource_policy == internal
