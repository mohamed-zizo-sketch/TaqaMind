"""Regression coverage for consistent missing-value handling."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from test_cleaning_feature_engineering_and_preprocessing import make_contract_and_context, map_csv

from taqamind.core.contracts.enums import DataType, MissingValuePolicy
from taqamind.core.interfaces import (
    ArtifactFormat,
    ArtifactReference,
    DatasetArtifact,
    ExecutionStatus,
)
from taqamind.core.preprocessing import ContractDataCleaner
from taqamind.core.preprocessing.missing_values import normalize_text_missing
from taqamind.core.registry import LocalDatasetArtifactStorage
from taqamind.core.validation.semantic_validator import ContractSemanticValidator


@pytest.mark.parametrize("data_type", [DataType.STRING, DataType.CATEGORY])
@pytest.mark.parametrize("policy", list(MissingValuePolicy))
def test_blank_text_policy(tmp_path: Path, data_type, policy):
    contract, _ = make_contract_and_context()
    feature = contract.get_feature("shift_code").model_copy(
        update={
            "data_type": data_type,
            "nullable": policy != MissingValuePolicy.REJECT,
            "missing_policy": policy,
        }
    )
    storage = LocalDatasetArtifactStorage(tmp_path)
    series = pd.Series(["   ", "A", pd.NA], dtype="string")
    validation = ContractSemanticValidator(storage)._validate_column(series, feature)
    cleaned, cleaning = ContractDataCleaner(storage)._clean_column(series, feature)
    assert cleaned.isna().tolist() == [True, False, True]
    codes = {issue.code for issue in validation}
    assert "INVALID_DATA_TYPE" not in codes
    assert "CONSTRAINT_ALLOWED_VALUES_VIOLATION" not in codes
    assert ("MISSING_VALUES_REJECTED" in codes) == (policy == MissingValuePolicy.REJECT)
    assert any(issue.code == "CLEANING_MISSING_REJECTED" for issue in cleaning) == (
        policy == MissingValuePolicy.REJECT
    )


def test_numeric_whitespace_remains_invalid():
    series = pd.Series([" ", "NaN", "NULL", pd.NA], dtype="string")
    assert normalize_text_missing(series, data_type=DataType.FLOAT).equals(series)
    text = normalize_text_missing(series, data_type=DataType.STRING)
    assert text.isna().tolist() == [True, False, False, True]


def test_cleaner_blocks_missing_required_column(tmp_path: Path):
    contract, context = make_contract_and_context()
    storage = LocalDatasetArtifactStorage(tmp_path)

    mapped = map_csv(
        storage,
        context,
        contract,
        b"batch_id,planned_qty,waste_qty\nB1,100,0\n",
    )

    incomplete_payload = b"batch_id,waste_qty\nB1,0\n"

    incomplete_ref = ArtifactReference(
        factory_id=context.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=sha256(incomplete_payload).hexdigest(),
        size_bytes=len(incomplete_payload),
    )

    storage.put(
        factory_id=context.factory_id,
        reference=incomplete_ref,
        stream=BytesIO(incomplete_payload),
    )

    incomplete_metadata = mapped.metadata.model_copy(
        update={
            "columns": tuple(
                column for column in mapped.metadata.columns if column.name != "planned_qty"
            ),
        }
    )

    incomplete_artifact = DatasetArtifact(
        metadata=incomplete_metadata,
        reference=incomplete_ref,
    )

    result = ContractDataCleaner(storage).clean(
        context=context,
        artifact=incomplete_artifact,
        contract=contract,
    )

    assert result.status == ExecutionStatus.BLOCKED
    assert result.output_artifact is None
    assert result.report is not None

    assert any(issue.code == "CLEANING_MISSING_REQUIRED_COLUMNS" for issue in result.report.issues)
