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
from taqamind.core.ingestion.safe_csv import SafeCsvError, SafeCsvReader
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
    ExecutionError,
    ExecutionStatus,
    StageResult,
    StageTiming,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage
from taqamind.core.preprocessing.type_conversion import ContractTypeConverter


class ContractDataCleaner(DataCleaner):
    """Apply deterministic, row-level cleaning without learned dataset statistics."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        *,
        implementation_version: str = "0.2.0",
        delimiter: str = ",",
        max_rows: int | None = None,
    ) -> None:
        self._storage = storage
        self._implementation_version = implementation_version
        self._reader = SafeCsvReader(delimiter=delimiter, max_rows=max_rows)
        self._converter = ContractTypeConverter()

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
            raw_bytes = stream.read()

        try:
            df = self._reader.read(raw_bytes)
        except SafeCsvError as exc:
            return StageResult(
                context=context,
                stage=PipelineStage.CLEANING,
                status=ExecutionStatus.FAILED,
                input_artifact=artifact,
                report=None,
                timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
                error=ExecutionError(code=exc.code, message=str(exc)),
            )

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

        if (
            not cleaned_df.empty
            and contract.business_keys
            and all(key in cleaned_df.columns for key in contract.business_keys)
        ):
            before_keys = df[list(contract.business_keys)]
            after_keys = cleaned_df[list(contract.business_keys)]
            before_dups = (
                ~before_keys.isna().any(axis=1) & before_keys.duplicated(keep="first")
            )
            after_dups = (
                ~after_keys.isna().any(axis=1) & after_keys.duplicated(keep="first")
            )
            new_collisions = np.flatnonzero((after_dups & ~before_dups).to_numpy())
            if len(new_collisions) > 0:
                issues.append(
                    ValidationIssue(
                        code="WHITESPACE_NORMALIZATION_KEY_COLLISION",
                        stage=PipelineStage.CLEANING,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=(
                            "Whitespace normalization introduced duplicate business keys."
                        ),
                        columns=contract.business_keys,
                        affected_rows=int(len(new_collisions)),
                        row_samples=tuple(int(i) for i in new_collisions[:20]),
                    )
                )

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

        cleaned_bytes = cleaned_df.to_csv(index=False, lineterminator="\r\n").encode("utf-8")
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

        elif feature.data_type in {
            DataType.FLOAT,
            DataType.INTEGER,
            DataType.BOOLEAN,
            DataType.DATE,
            DataType.DATETIME,
        }:
            conversion = self._converter.convert(result, data_type=feature.data_type)
            if not conversion.is_valid:
                code_map = {
                    DataType.FLOAT: "CLEANING_INVALID_FLOAT",
                    DataType.INTEGER: "CLEANING_INVALID_INTEGER",
                    DataType.BOOLEAN: "CLEANING_INVALID_BOOLEAN",
                    DataType.DATE: "CLEANING_INVALID_TIMESTAMP",
                    DataType.DATETIME: "CLEANING_INVALID_TIMESTAMP",
                }
                msg_map = {
                    DataType.FLOAT: (
                        f"Column '{feature.name}' contains non-numeric or non-finite values."
                    ),
                    DataType.INTEGER: f"Column '{feature.name}' contains non-integer values.",
                    DataType.BOOLEAN: (
                        f"Column '{feature.name}' contains invalid boolean literals."
                    ),
                    DataType.DATE: (
                        f"Column '{feature.name}' contains invalid date values."
                    ),
                    DataType.DATETIME: (
                        f"Column '{feature.name}' contains invalid datetime values."
                    ),
                }
                issues.append(
                    self._blocking_issue(
                        code=code_map[feature.data_type],
                        feature=feature,
                        indices=np.asarray(conversion.invalid_positions, dtype=int),
                        message=msg_map[feature.data_type],
                    )
                )
            else:
                result = conversion.values

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
