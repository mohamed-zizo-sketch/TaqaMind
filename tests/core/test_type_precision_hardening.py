"""Regression tests for exact bounds and fail-closed scalar conversion."""

from types import SimpleNamespace

import pandas as pd
import pytest

from taqamind.core.automl.data_access import convert_dataframe_by_contract
from taqamind.core.contracts.enums import DataType
from taqamind.core.contracts.feature import FeatureConstraints
from taqamind.core.interfaces.data_access import DataAccessDeniedError
from taqamind.core.preprocessing.type_conversion import ContractTypeConverter


@pytest.mark.parametrize(
    "value",
    [
        "2026-01-01T12:00:00.123456789Z",
        pd.Timestamp("2026-01-01T14:00:00.123456789+02:00"),
        "2026-01-01T14:00:00.123456789+02:00",
    ],
)
def test_datetime_preserves_nanoseconds(value):
    result = ContractTypeConverter().convert(
        pd.Series([value], dtype=object), data_type=DataType.DATETIME
    )
    assert result.is_valid
    assert result.values.iloc[0] == "2026-01-01T12:00:00.123456789Z"


@pytest.mark.parametrize("bound", [2**53 + 1, 2**63 - 1, -(2**63) + 1])
def test_integer_constraints_roundtrip_exactly(bound):
    original = FeatureConstraints(minimum=bound, maximum=bound)
    restored = FeatureConstraints.model_validate_json(original.model_dump_json())
    assert type(restored.minimum) is int
    assert restored.minimum == restored.maximum == bound


@pytest.mark.parametrize("token", ["abc", "NaN", "Infinity", "1.5", "9223372036854775808"])
def test_ml_conversion_rejects_invalid_values(token):
    contract = SimpleNamespace(column_map={"quantity": SimpleNamespace(data_type=DataType.INTEGER)})
    with pytest.raises(DataAccessDeniedError, match="Invalid typed values"):
        convert_dataframe_by_contract(pd.DataFrame({"quantity": [token]}), contract=contract)


def test_ml_conversion_preserves_genuine_missing():
    contract = SimpleNamespace(column_map={"quantity": SimpleNamespace(data_type=DataType.INTEGER)})
    result = convert_dataframe_by_contract(
        pd.DataFrame({"quantity": pd.Series(["12", pd.NA], dtype="string")}), contract=contract
    )
    assert result["quantity"].iloc[0] == 12
    assert pd.isna(result["quantity"].iloc[1])
