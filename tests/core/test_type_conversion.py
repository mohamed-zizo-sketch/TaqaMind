"""Regression tests for contract-directed scalar conversion."""

import numpy as np
import pandas as pd
import pytest

from taqamind.core.contracts.enums import DataType
from taqamind.core.preprocessing.type_conversion import ContractTypeConverter


@pytest.mark.parametrize("data_type", [DataType.STRING, DataType.CATEGORY])
def test_text_preserved(data_type):
    original = pd.Series(["000123", "NA", "NULL", " A ", pd.NA], dtype="string")
    result = ContractTypeConverter().convert(original, data_type=data_type)
    assert result.is_valid
    pd.testing.assert_series_equal(result.values, original)


@pytest.mark.parametrize("token", ["", "  ", 123])
def test_invalid_text(token):
    result = ContractTypeConverter().convert(pd.Series([token]), data_type=DataType.STRING)
    assert result.invalid_positions == (0,)


@pytest.mark.parametrize(
    ("token", "expected"),
    [("9007199254740993", 9007199254740993), ("9223372036854775807", 2**63 - 1),
     ("-9223372036854775808", -(2**63)), ("2.000", 2), ("2e3", 2000), ("0", 0)],
)
def test_exact_integer(token, expected):
    result = ContractTypeConverter().convert(
        pd.Series([token, pd.NA]), data_type=DataType.INTEGER
    )
    assert result.is_valid
    assert result.values.iloc[0] == expected
    assert pd.isna(result.values.iloc[1])
    assert str(result.values.dtype) == "Int64"


@pytest.mark.parametrize(
    "token", ["1.5", "1.00000000000000001", "9223372036854775808", "-9223372036854775809",
              "NA", "NULL", "inf", "1_000", True, float(2**53), np.float32(2**24)],
)
def test_invalid_integer(token):
    result = ContractTypeConverter().convert(
        pd.Series([token], dtype=object), data_type=DataType.INTEGER
    )
    assert result.invalid_positions == (0,)
    assert not result.is_valid


@pytest.mark.parametrize("token", ["inf", "NaN", "1e999", "1e-999", "bad", True])
def test_invalid_float(token):
    result = ContractTypeConverter().convert(pd.Series([token]), data_type=DataType.FLOAT)
    assert result.invalid_positions == (0,)


def test_float_and_missing():
    original = pd.Series(["1.2300", "0", pd.NA], index=[10, 20, 30], name="qty")
    result = ContractTypeConverter().convert(original, data_type=DataType.FLOAT)
    expected = pd.Series([1.23, 0.0, pd.NA], index=original.index, name="qty", dtype="Float64")
    pd.testing.assert_series_equal(result.values, expected)
    assert result.is_valid
    assert original.iloc[0] == "1.2300"


def test_boolean_literals():
    result = ContractTypeConverter().convert(
        pd.Series([" YES ", "false", "1.0", "0", pd.NA, "maybe"]),
        data_type=DataType.BOOLEAN,
    )
    assert result.invalid_positions == (5,)
    expected = pd.Series([True, False, True, False, pd.NA, pd.NA], dtype="boolean")
    pd.testing.assert_series_equal(result.values, expected)


def test_invalid_positions_are_not_index_labels():
    result = ContractTypeConverter().convert(
        pd.Series(["1", "bad", pd.NA], index=[9, 9, 20]), data_type=DataType.INTEGER
    )
    assert result.invalid_positions == (1,)
    assert result.values.index.tolist() == [9, 9, 20]


def test_date_and_datetime_utc_and_precision_preservation() -> None:
    date_res = ContractTypeConverter().convert(
        pd.Series(["2026-11-25", "not-a-date", pd.NA]), data_type=DataType.DATE
    )
    assert date_res.invalid_positions == (1,)
    assert date_res.values.iloc[0] == "2026-11-25"
    assert pd.isna(date_res.values.iloc[2])

    dt_res = ContractTypeConverter().convert(
        pd.Series(
            [
                "2026-11-25T14:30:05.123456+02:00",
                "2026-11-25T12:30:05Z",
                "invalid-ts",
                pd.NA,
            ]
        ),
        data_type=DataType.DATETIME,
    )
    assert dt_res.invalid_positions == (2,)
    assert dt_res.values.iloc[0] == "2026-11-25T12:30:05.123456Z"
    assert dt_res.values.iloc[1] == "2026-11-25T12:30:05Z"
    assert pd.isna(dt_res.values.iloc[3])
