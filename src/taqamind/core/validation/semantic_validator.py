"""Contract-driven semantic validation engine for mapped datasets."""

from datetime import UTC, datetime
from decimal import Decimal

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
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.dataset import DatasetArtifact, DatasetState
from taqamind.core.interfaces.processing import (
    SemanticValidator,
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

_MAX_ROW_SAMPLES = 20


class ContractSemanticValidator(SemanticValidator):
    """Validate mapped dataset values against DatasetContract rules."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        *,
        delimiter: str = ",",
        max_rows: int | None = None,
    ) -> None:
        self._storage = storage
        self._reader = SafeCsvReader(delimiter=delimiter, max_rows=max_rows)
        self._converter = ContractTypeConverter()

    def validate(
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
                stage=PipelineStage.SEMANTIC_VALIDATION,
                status=ExecutionStatus.FAILED,
                input_artifact=artifact,
                report=None,
                timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
                error=ExecutionError(code=exc.code, message=str(exc)),
            )

        issues: list[ValidationIssue] = []

        if df.empty:
            issues.append(
                ValidationIssue(
                    code="EMPTY_DATASET",
                    stage=PipelineStage.SEMANTIC_VALIDATION,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message="Mapped dataset contains zero rows.",
                    recommendation="Provide a non-empty dataset before validation.",
                )
            )

        column_map = contract.column_map
        unknown_columns = tuple(
            str(col) for col in df.columns if str(col) not in column_map
        )
        if unknown_columns:
            issues.append(
                ValidationIssue(
                    code="UNKNOWN_CANONICAL_COLUMNS",
                    stage=PipelineStage.SEMANTIC_VALIDATION,
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    message="Mapped dataset contains columns not defined in contract.",
                    columns=unknown_columns,
                    recommendation="Remove or map uncontracted columns before validation.",
                )
            )

        target_name = contract.target.feature.name
        if context.purpose == DatasetPurpose.PREDICTION and target_name in df.columns:
            issues.append(
                ValidationIssue(
                    code="TARGET_PRESENT_IN_PREDICTION",
                    stage=PipelineStage.SEMANTIC_VALIDATION,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message="Prediction dataset must not include the target column.",
                    columns=(target_name,),
                    recommendation="Exclude target column from prediction inputs.",
                )
            )

        if context.purpose != DatasetPurpose.RESEARCH:
            missing_required = tuple(
                col
                for col in contract.required_columns(context.purpose)
                if col not in df.columns
            )
            if missing_required:
                issues.append(
                    ValidationIssue(
                        code="MISSING_REQUIRED_COLUMNS",
                        stage=PipelineStage.SEMANTIC_VALIDATION,
                        severity=ValidationSeverity.CRITICAL,
                        impact=IssueImpact.BLOCKING,
                        message="Mapped dataset is missing required contract columns.",
                        columns=missing_required,
                        recommendation="Map all required columns for this purpose.",
                    )
                )

        if not df.empty and contract.business_keys:
            keys_present = all(key in df.columns for key in contract.business_keys)
            if keys_present:
                key_frame = df[list(contract.business_keys)]
                non_null_mask = ~key_frame.isna().any(axis=1)
                dup_mask = non_null_mask & key_frame.duplicated(keep="first")
                dup_indices = np.flatnonzero(dup_mask.to_numpy())
                if len(dup_indices) > 0:
                    issues.append(
                        ValidationIssue(
                            code="DUPLICATE_BUSINESS_KEY",
                            stage=PipelineStage.SEMANTIC_VALIDATION,
                            severity=ValidationSeverity.ERROR,
                            impact=IssueImpact.BLOCKING,
                            message=(
                                "Duplicate business key records violate row identity."
                            ),
                            columns=contract.business_keys,
                            affected_rows=int(len(dup_indices)),
                            row_samples=tuple(
                                int(idx) for idx in dup_indices[:_MAX_ROW_SAMPLES]
                            ),
                            recommendation="Deduplicate business keys before proceeding.",
                        )
                    )

        for col_name in df.columns:
            col_key = str(col_name)
            feature = column_map.get(col_key)
            if feature is None:
                continue
            issues.extend(self._validate_column(df[col_name], feature))

        if (
            not df.empty
            and contract.target.derived_rate is not None
            and context.purpose != DatasetPurpose.PREDICTION
        ):
            denom_name = contract.target.derived_rate.denominator
            if denom_name in df.columns:
                denom_feature = column_map.get(denom_name)
                denom_type = (
                    denom_feature.data_type
                    if denom_feature is not None
                    and denom_feature.data_type in {DataType.INTEGER, DataType.FLOAT}
                    else DataType.FLOAT
                )
                denom_res = self._converter.convert(df[denom_name], data_type=denom_type)
                bad_indices = [
                    idx
                    for idx, val in enumerate(denom_res.values)
                    if not pd.isna(val) and Decimal(str(val)) <= 0
                ]
                if bad_indices:
                    issues.append(
                        ValidationIssue(
                            code="INVALID_RATE_DENOMINATOR",
                            stage=PipelineStage.SEMANTIC_VALIDATION,
                            severity=ValidationSeverity.ERROR,
                            impact=IssueImpact.BLOCKING,
                            message=(
                                f"Rate denominator '{denom_name}' must be strictly positive."
                            ),
                            columns=(denom_name,),
                            affected_rows=int(len(bad_indices)),
                            row_samples=tuple(
                                int(idx) for idx in bad_indices[:_MAX_ROW_SAMPLES]
                            ),
                        )
                    )

        report = ValidationReport(
            dataset_id=context.dataset_id,
            contract_version=context.contract_version,
            dataset_revision_id=artifact.metadata.revision_id,
            domain=context.domain,
            process=context.process,
            purpose=context.purpose,
            completed_stages=(PipelineStage.SEMANTIC_VALIDATION,),
            issues=tuple(issues),
        )
        status = (
            ExecutionStatus.BLOCKED
            if report.status == ReadinessStatus.BLOCKED
            else ExecutionStatus.SUCCEEDED
        )
        return StageResult(
            context=context,
            stage=PipelineStage.SEMANTIC_VALIDATION,
            status=status,
            input_artifact=artifact,
            output_artifact=None,
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )

    def _validate_column(
        self,
        series: pd.Series,
        feature: FeatureContract,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        na_mask = series.isna()
        missing_indices = np.flatnonzero(na_mask.to_numpy())
        if len(missing_indices) > 0:
            if feature.missing_policy == MissingValuePolicy.REJECT:
                issues.append(
                    ValidationIssue(
                        code="MISSING_VALUES_REJECTED",
                        stage=PipelineStage.SEMANTIC_VALIDATION,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=(
                            f"Non-nullable column '{feature.name}' contains missing values."
                        ),
                        columns=(feature.name,),
                        affected_rows=int(len(missing_indices)),
                        row_samples=tuple(
                            int(idx) for idx in missing_indices[:_MAX_ROW_SAMPLES]
                        ),
                        recommendation="Populate missing values or reject incomplete rows.",
                    )
                )
            elif feature.missing_policy == MissingValuePolicy.IMPUTE:
                issues.append(
                    ValidationIssue(
                        code="MISSING_VALUES_PENDING_IMPUTATION",
                        stage=PipelineStage.SEMANTIC_VALIDATION,
                        severity=ValidationSeverity.WARNING,
                        impact=IssueImpact.NONE,
                        message=(
                            f"Column '{feature.name}' contains missing values scheduled "
                            "for cleaning/imputation."
                        ),
                        columns=(feature.name,),
                        affected_rows=int(len(missing_indices)),
                        row_samples=tuple(
                            int(idx) for idx in missing_indices[:_MAX_ROW_SAMPLES]
                        ),
                    )
                )

        non_null_mask = ~na_mask
        if not bool(non_null_mask.any()):
            return issues

        parsed_numeric, type_bad_indices = self._check_data_type(
            series=series,
            non_null_mask=non_null_mask,
            data_type=feature.data_type,
        )
        if len(type_bad_indices) > 0:
            issues.append(
                ValidationIssue(
                    code="INVALID_DATA_TYPE",
                    stage=PipelineStage.SEMANTIC_VALIDATION,
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    message=(
                        f"Column '{feature.name}' contains values incompatible with "
                        f"'{feature.data_type.value}'."
                    ),
                    columns=(feature.name,),
                    affected_rows=int(len(type_bad_indices)),
                    row_samples=tuple(
                        int(idx) for idx in type_bad_indices[:_MAX_ROW_SAMPLES]
                    ),
                )
            )

        constraints = feature.constraints
        if parsed_numeric is not None and (
            constraints.minimum is not None or constraints.maximum is not None
        ):
            min_dec = (
                Decimal(str(constraints.minimum))
                if constraints.minimum is not None
                else None
            )
            max_dec = (
                Decimal(str(constraints.maximum))
                if constraints.maximum is not None
                else None
            )
            range_bad_indices: list[int] = []
            for idx, val in enumerate(parsed_numeric):
                if pd.isna(val):
                    continue
                dec_val = (
                    Decimal(int(val))
                    if feature.data_type == DataType.INTEGER
                    else Decimal(str(val))
                )
                if min_dec is not None:
                    if (constraints.minimum_inclusive and dec_val < min_dec) or (
                        not constraints.minimum_inclusive and dec_val <= min_dec
                    ):
                        range_bad_indices.append(idx)
                        continue
                if max_dec is not None:
                    if (constraints.maximum_inclusive and dec_val > max_dec) or (
                        not constraints.maximum_inclusive and dec_val >= max_dec
                    ):
                        range_bad_indices.append(idx)
            if range_bad_indices:
                issues.append(
                    ValidationIssue(
                        code="CONSTRAINT_RANGE_VIOLATION",
                        stage=PipelineStage.SEMANTIC_VALIDATION,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=(
                            f"Column '{feature.name}' violates declared numeric bounds."
                        ),
                        columns=(feature.name,),
                        affected_rows=int(len(range_bad_indices)),
                        row_samples=tuple(
                            int(idx) for idx in range_bad_indices[:_MAX_ROW_SAMPLES]
                        ),
                    )
                )

        if constraints.allowed_values is not None:
            allowed = set(constraints.allowed_values)
            str_values = series[non_null_mask].astype(str).str.strip()
            invalid_mask = ~str_values.isin(allowed)
            bad_positions = np.flatnonzero(non_null_mask.to_numpy())[
                invalid_mask.to_numpy()
            ]
            if len(bad_positions) > 0:
                issues.append(
                    ValidationIssue(
                        code="CONSTRAINT_ALLOWED_VALUES_VIOLATION",
                        stage=PipelineStage.SEMANTIC_VALIDATION,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=(
                            f"Column '{feature.name}' contains values outside allowed set."
                        ),
                        columns=(feature.name,),
                        affected_rows=int(len(bad_positions)),
                        row_samples=tuple(
                            int(idx) for idx in bad_positions[:_MAX_ROW_SAMPLES]
                        ),
                    )
                )

        return issues

    def _check_data_type(
        self,
        *,
        series: pd.Series,
        non_null_mask: pd.Series,
        data_type: DataType,
    ) -> tuple[pd.Series | None, np.ndarray]:
        if data_type in {
            DataType.STRING,
            DataType.CATEGORY,
            DataType.INTEGER,
            DataType.FLOAT,
            DataType.BOOLEAN,
        }:
            conversion = self._converter.convert(series, data_type=data_type)
            numeric_values = (
                conversion.values
                if data_type in {DataType.INTEGER, DataType.FLOAT}
                else None
            )
            return numeric_values, np.asarray(conversion.invalid_positions, dtype=int)

        # DATE or DATETIME (deferred until AUD-007/AUD-010).
        parsed = pd.to_datetime(series, errors="coerce")
        bad_mask = non_null_mask & parsed.isna()
        return None, np.flatnonzero(bad_mask.to_numpy())
