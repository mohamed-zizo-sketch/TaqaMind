"""Contract-driven semantic validation engine for mapped datasets."""

from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import (
    DatasetPurpose,
    DataType,
    IssueImpact,
    PipelineStage,
    ReadinessStatus,
    ValidationSeverity,
)
from taqamind.core.contracts.feature import FeatureContract
from taqamind.core.contracts.validation import ValidationIssue, ValidationReport
from taqamind.core.ingestion.artifact_csv import read_verified_csv
from taqamind.core.ingestion.resource_policy import (
    CsvResourcePolicy,
    resolve_internal_csv_policy,
)
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
from taqamind.core.lineage.canonicalization import find_duplicate_canonical_business_keys
from taqamind.core.preprocessing.type_conversion import ContractTypeConverter
from taqamind.core.validation.column_validation import validate_column

_MAX_ROW_SAMPLES = 20


class ContractSemanticValidator(SemanticValidator):
    """Validate mapped dataset values against DatasetContract rules."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        *,
        delimiter: str = ",",
        max_rows: int | None = None,
        resource_policy: CsvResourcePolicy | None = None,
    ) -> None:
        self._storage = storage
        self._resource_policy = resolve_internal_csv_policy(resource_policy, max_rows=max_rows)
        self._reader = SafeCsvReader(delimiter=delimiter, resource_policy=self._resource_policy)
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

        try:
            df = read_verified_csv(
                storage=self._storage,
                factory_id=context.factory_id,
                reference=artifact.reference,
                reader=self._reader,
                policy=self._resource_policy,
            )
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
        unknown_columns = tuple(str(col) for col in df.columns if str(col) not in column_map)
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
                col for col in contract.required_columns(context.purpose) if col not in df.columns
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
                dup_indices = find_duplicate_canonical_business_keys(
                    df,
                    contract,
                    converter=self._converter,
                )
                if len(dup_indices) > 0:
                    issues.append(
                        ValidationIssue(
                            code="DUPLICATE_BUSINESS_KEY",
                            stage=PipelineStage.SEMANTIC_VALIDATION,
                            severity=ValidationSeverity.ERROR,
                            impact=IssueImpact.BLOCKING,
                            message=("Duplicate business key records violate row identity."),
                            columns=contract.business_keys,
                            affected_rows=int(len(dup_indices)),
                            row_samples=dup_indices[:_MAX_ROW_SAMPLES],
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
                            message=(f"Rate denominator '{denom_name}' must be strictly positive."),
                            columns=(denom_name,),
                            affected_rows=int(len(bad_indices)),
                            row_samples=tuple(int(idx) for idx in bad_indices[:_MAX_ROW_SAMPLES]),
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
        """Validate a mapped column using shared contract rules."""

        return validate_column(
            series,
            feature,
            stage=PipelineStage.SEMANTIC_VALIDATION,
            converter=self._converter,
        )
