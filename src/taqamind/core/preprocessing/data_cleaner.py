"""Deterministic contract-driven data cleaning engine."""

from datetime import UTC, datetime
from io import BytesIO

import numpy as np
import pandas as pd

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import (
    DatasetPurpose,
    DataType,
    IssueImpact,
    MissingValuePolicy,
    PipelineStage,
    ReadinessStatus,
    ValidationSeverity,
)
from taqamind.core.contracts.feature import FeatureContract
from taqamind.core.contracts.validation import ValidationIssue, ValidationReport
from taqamind.core.interfaces.artifacts import (
    ArtifactFormat,
    ArtifactReference,
    calculate_bytes_sha256,
)
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.dataset import (
    DatasetArtifact,
    DatasetColumnSchema,
    DatasetState,
)
from taqamind.core.interfaces.processing import (
    DataCleaner,
    validate_processing_input,
)
from taqamind.core.interfaces.results import (
    ExecutionStatus,
    StageResult,
    StageTiming,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage

_TRUE_LITERALS = {"true", "1", "yes", "1.0"}
_FALSE_LITERALS = {"false", "0", "no", "0.0"}


class ContractDataCleaner(DataCleaner):
    """Apply deterministic, row-level cleaning without learned dataset statistics."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        *,
        implementation_version: str = "0.1.0",
    ) -> None:
        self._storage = storage
        self._implementation_version = implementation_version

    def clean(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> StageResult:
        started_at = datetime.now(UTC)
        validate_processing_input(
            context=context,
            artifact=artifact,
            contract=contract,
            expected_state=DatasetState.MAPPED,
        )

        with self._storage.open_verified(
            factory_id=context.factory_id,
            reference=artifact.reference,
        ) as stream:
            df = pd.read_csv(stream, keep_default_na=True)

        cleaned_df = df.copy()
        issues: list[ValidationIssue] = []
        column_map = contract.column_map

        if cleaned_df.empty:
            issues.append(
                ValidationIssue(
                    code="CLEANING_EMPTY_DATASET",
                    stage=PipelineStage.CLEANING,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message="Cannot clean an empty dataset.",
                )
            )

        if (
            context.purpose == DatasetPurpose.PREDICTION
            and contract.target.feature.name in cleaned_df.columns
        ):
            issues.append(
                ValidationIssue(
                    code="TARGET_PRESENT_IN_PREDICTION",
                    stage=PipelineStage.CLEANING,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message="Prediction dataset cannot contain the target column.",
                    columns=(contract.target.feature.name,),
                )
            )

        for col_name in cleaned_df.columns:
            col_key = str(col_name)
            feature = column_map.get(col_key)
            if feature is None:
                issues.append(
                    ValidationIssue(
                        code="UNKNOWN_COLUMN_IN_CLEANING",
                        stage=PipelineStage.CLEANING,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=f"Column '{col_key}' is not declared in dataset contract.",
                        columns=(col_key,),
                    )
                )
                continue

            cleaned_series, col_issues = self._clean_column(cleaned_df[col_name], feature)
            cleaned_df[col_name] = cleaned_series
            issues.extend(col_issues)

        report = ValidationReport(
            dataset_id=context.dataset_id,
            contract_version=context.contract_version,
            dataset_revision_id=artifact.metadata.revision_id,
            domain=context.domain,
            process=context.process,
            purpose=context.purpose,
            completed_stages=(PipelineStage.CLEANING,),
            issues=tuple(issues),
        )

        if report.status == ReadinessStatus.BLOCKED:
            return StageResult(
                context=context,
                stage=PipelineStage.CLEANING,
                status=ExecutionStatus.BLOCKED,
                input_artifact=artifact,
                output_artifact=None,
                report=report,
                timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
            )

        cleaned_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
        cleaned_ref = ArtifactReference(
            factory_id=context.factory_id,
            format=ArtifactFormat.CSV,
            content_sha256=calculate_bytes_sha256(cleaned_bytes),
            size_bytes=len(cleaned_bytes),
        )
        self._storage.put(
            factory_id=context.factory_id,
            reference=cleaned_ref,
            stream=BytesIO(cleaned_bytes),
        )

        columns = tuple(
            DatasetColumnSchema(
                name=str(col),
                physical_dtype=str(cleaned_df[col].dtype),
            )
            for col in cleaned_df.columns
        )
        cleaned_metadata = artifact.metadata.derive(
            context=context,
            state=DatasetState.CLEANED,
            row_count=len(cleaned_df),
            columns=columns,
            operation="clean_deterministic",
            implementation_version=self._implementation_version,
            description=(
                "Applied deterministic type, whitespace, and format normalization "
                "without learned imputation or blind row removal."
            ),
        )
        output_artifact = DatasetArtifact(
            metadata=cleaned_metadata,
            reference=cleaned_ref,
        )
        return StageResult(
            context=context,
            stage=PipelineStage.CLEANING,
            status=ExecutionStatus.SUCCEEDED,
            input_artifact=artifact,
            output_artifact=output_artifact,
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )

    def _clean_column(
        self,
        series: pd.Series,
        feature: FeatureContract,
    ) -> tuple[pd.Series, list[ValidationIssue]]:
        issues: list[ValidationIssue] = []
        result = series.copy()

        if feature.data_type in {DataType.STRING, DataType.CATEGORY}:
            non_null = result.notna()
            stripped = result.astype(str).str.strip()
            result = result.where(~non_null, stripped)
            blank_mask = non_null & (stripped == "")
            if bool(blank_mask.any()):
                result = result.mask(blank_mask, pd.NA)

        elif feature.data_type == DataType.FLOAT:
            coerced = pd.to_numeric(result, errors="coerce")
            invalid_mask = result.notna() & (coerced.isna() | ~np.isfinite(coerced))
            if bool(invalid_mask.any()):
                issues.append(
                    self._blocking_issue(
                        code="CLEANING_INVALID_FLOAT",
                        feature=feature,
                        indices=np.flatnonzero(invalid_mask.to_numpy()),
                        message=(
                            f"Column '{feature.name}' contains non-numeric "
                            "or non-finite values."
                        ),
                    )
                )
            else:
                result = coerced.astype("float64")

        elif feature.data_type == DataType.INTEGER:
            coerced = pd.to_numeric(result, errors="coerce")
            is_finite = coerced.notna() & np.isfinite(coerced)
            fractional = pd.Series(False, index=result.index)
            fractional.loc[is_finite] = (coerced.loc[is_finite] % 1) != 0
            invalid_mask = result.notna() & (~is_finite | fractional)
            if bool(invalid_mask.any()):
                issues.append(
                    self._blocking_issue(
                        code="CLEANING_INVALID_INTEGER",
                        feature=feature,
                        indices=np.flatnonzero(invalid_mask.to_numpy()),
                        message=f"Column '{feature.name}' contains non-integer values.",
                    )
                )
            else:
                result = coerced.astype("Int64")

        elif feature.data_type == DataType.BOOLEAN:
            if not pd.api.types.is_bool_dtype(result):
                normalized_vals: list[bool | object] = []
                bad_indices: list[int] = []
                for idx, val in enumerate(result.tolist()):
                    if pd.isna(val):
                        normalized_vals.append(pd.NA)
                        continue
                    token = str(val).strip().casefold()
                    if token in _TRUE_LITERALS:
                        normalized_vals.append(True)
                    elif token in _FALSE_LITERALS:
                        normalized_vals.append(False)
                    else:
                        normalized_vals.append(val)
                        bad_indices.append(idx)
                if bad_indices:
                    issues.append(
                        self._blocking_issue(
                            code="CLEANING_INVALID_BOOLEAN",
                            feature=feature,
                            indices=np.asarray(bad_indices, dtype=int),
                            message=f"Column '{feature.name}' contains invalid boolean literals.",
                        )
                    )
                else:
                    result = pd.Series(normalized_vals, index=result.index, dtype="boolean")

        elif feature.data_type in {DataType.DATE, DataType.DATETIME}:
            parsed = pd.to_datetime(
                result,
                errors="coerce",
                utc=(feature.data_type == DataType.DATETIME),
            )
            invalid_mask = result.notna() & parsed.isna()
            if bool(invalid_mask.any()):
                issues.append(
                    self._blocking_issue(
                        code="CLEANING_INVALID_TIMESTAMP",
                        feature=feature,
                        indices=np.flatnonzero(invalid_mask.to_numpy()),
                        message=f"Column '{feature.name}' contains invalid date/datetime values.",
                    )
                )
            else:
                fmt = "%Y-%m-%d" if feature.data_type == DataType.DATE else "%Y-%m-%dT%H:%M:%SZ"
                formatted = parsed.dt.strftime(fmt)
                result = formatted.where(result.notna(), pd.NA)

        missing_indices = np.flatnonzero(result.isna().to_numpy())
        if len(missing_indices) > 0 and feature.missing_policy == MissingValuePolicy.REJECT:
            issues.append(
                self._blocking_issue(
                    code="CLEANING_MISSING_REJECTED",
                    feature=feature,
                    indices=missing_indices,
                    message=f"Non-nullable column '{feature.name}' still contains missing values.",
                )
            )
        elif len(missing_indices) > 0 and feature.missing_policy == MissingValuePolicy.IMPUTE:
            issues.append(
                ValidationIssue(
                    code="CLEANING_PRESERVED_FOR_TRAIN_IMPUTATION",
                    stage=PipelineStage.CLEANING,
                    severity=ValidationSeverity.INFO,
                    impact=IssueImpact.NONE,
                    message=(
                        f"Preserved {len(missing_indices)} missing value(s) in '{feature.name}' "
                        "for train-only imputation after splitting."
                    ),
                    columns=(feature.name,),
                    affected_rows=int(len(missing_indices)),
                    row_samples=tuple(int(i) for i in missing_indices[:20]),
                )
            )

        return result, issues

    @staticmethod
    def _blocking_issue(
        *,
        code: str,
        feature: FeatureContract,
        indices: np.ndarray,
        message: str,
    ) -> ValidationIssue:
        return ValidationIssue(
            code=code,
            stage=PipelineStage.CLEANING,
            severity=ValidationSeverity.ERROR,
            impact=IssueImpact.BLOCKING,
            message=message,
            columns=(feature.name,),
            affected_rows=int(len(indices)),
            row_samples=tuple(int(i) for i in indices[:20]),
        )
