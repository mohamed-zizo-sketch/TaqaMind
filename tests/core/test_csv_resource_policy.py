"""Tests for immutable CSV resource policies."""

from dataclasses import FrozenInstanceError

import pytest

from taqamind.core.ingestion.resource_policy import (
    DEFAULT_RAW_CSV_POLICY,
    CsvResourcePolicy,
)


def test_default_raw_csv_policy():
    policy = DEFAULT_RAW_CSV_POLICY

    assert policy.max_bytes == 52_428_800
    assert policy.max_rows == 250_000
    assert policy.max_columns == 500
    assert policy.max_cells == 5_000_000


def test_custom_resource_policy():
    policy = CsvResourcePolicy(
        max_bytes=1024,
        max_rows=100,
        max_columns=10,
        max_cells=500,
    )

    assert policy.max_bytes == 1024
    assert policy.max_rows == 100
    assert policy.max_columns == 10
    assert policy.max_cells == 500


@pytest.mark.parametrize(
    "field",
    [
        "max_bytes",
        "max_rows",
        "max_columns",
        "max_cells",
    ],
)
@pytest.mark.parametrize(
    "value",
    [0, -1, True, False, 1.5, "100", None],
)
def test_rejects_invalid_resource_limits(field, value):
    with pytest.raises(ValueError, match=field):
        CsvResourcePolicy(**{field: value})


def test_resource_policy_is_immutable():
    policy = CsvResourcePolicy()

    with pytest.raises(FrozenInstanceError):
        policy.max_bytes = 1024
