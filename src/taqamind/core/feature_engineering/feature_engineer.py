"""Deterministic and leakage-safe industrial feature engineering engine."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO

import pandas as pd

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import (
    DatasetPurpose,
    DataType,
    FeatureOrigin,
    IssueImpact,
    PipelineStage,
    ReadinessStatus,
    ValidationSeverity,
)
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
    FeatureEngineer,
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


@dataclass(frozen=True, slots=True)
class DerivedFeatureSpec:
    """Declarative deterministic row-level transformation for a derived feature."""

    feature_name: str
    compute: Callable[[pd.DataFrame], pd.Series]


class DeterministicFeatureEngineer(FeatureEngineer):
    """Compute leakage-safe row-level derived features defined by contract specs."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        *,
        derived_specs: tuple[DerivedFeatureSpec, ...] = (),
        implementation_version: str = "0.2.0",
        delimiter: str = ",",
        max_rows: int | None = None,
    ) -> None:
        names = [spec.feature_name for spec in derived_specs]
        if len(names) != len(set(names)):
            raise ValueError("DerivedFeatureSpec feature_name entries must be unique.")
        self._storage = storage
        self._specs = {spec.feature_name: spec for spec in derived_specs}
        self._implementation_version = implementation_version
        self._reader = SafeCsvReader(delimiter=delimiter, max_rows=max_rows)
        self._converter = ContractTypeConverter()

    def engineer(
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
            expected_state=DatasetState.CLEANED,
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
                stage=PipelineStage.FEATURE_ENGINEERING,
                status=ExecutionStatus.FAILED,
                input_artifact=artifact,
                report=None,
                timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
                error=ExecutionError(code=exc.code, message=str(exc)),
            )

        engineered_df = df.copy()
        column_map = contract.column_map
        for col_name in engineered_df.columns:
            feature = column_map.get(str(col_name))
            if feature is None:
                continue
            if feature.data_type in {DataType.INTEGER, DataType.FLOAT, DataType.BOOLEAN}:
                conv = self._converter.convert(
                    engineered_df[col_name], data_type=feature.data_type
                )
                if conv.is_valid:
                    engineered_df[col_name] = conv.values
        issues: list[ValidationIssue] = []

        if (
            context.purpose == DatasetPurpose.PREDICTION
            and contract.target.feature.name in engineered_df.columns
        ):
            issues.append(
                ValidationIssue(
                    code="TARGET_PRESENT_IN_PREDICTION",
                    stage=PipelineStage.FEATURE_ENGINEERING,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message="Prediction dataset must not include the target column.",
                    columns=(contract.target.feature.name,),
                )
            )

        target_name = contract.target.feature.name
        column_map = contract.column_map

        # Validate any custom specs reference valid DERIVED contract features.
        for spec_name in self._specs:
            if spec_name not in column_map:
                issues.append(
                    ValidationIssue(
                        code="UNKNOWN_DERIVED_FEATURE_SPEC",
                        stage=PipelineStage.FEATURE_ENGINEERING,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=f"Derived spec '{spec_name}' is not in DatasetContract.",
                        columns=(spec_name,),
                    )
                )
            elif column_map[spec_name].origin != FeatureOrigin.DERIVED:
                issues.append(
                    ValidationIssue(
                        code="NON_DERIVED_FEATURE_SPEC",
                        stage=PipelineStage.FEATURE_ENGINEERING,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=f"Feature '{spec_name}' is not declared with origin DERIVED.",
                        columns=(spec_name,),
                    )
                )

        # Topologically order derived features so chained derivations execute safely.
        derived_features = [
            feature
            for feature in contract.features
            if feature.origin == FeatureOrigin.DERIVED
        ]
        ordered_derived = self._topological_sort(derived_features)

        for feature in ordered_derived:
            if (
                context.purpose == DatasetPurpose.PREDICTION
                and not contract.is_available_at_prediction(feature)
            ):
                continue

            deps = contract.get_dependencies(feature.name)
            if target_name in deps:
                issues.append(
                    ValidationIssue(
                        code="DERIVED_FEATURE_TARGET_LEAKAGE",
                        stage=PipelineStage.FEATURE_ENGINEERING,
                        severity=ValidationSeverity.CRITICAL,
                        impact=IssueImpact.BLOCKING,
                        message=f"Derived feature '{feature.name}' depends on target.",
                        columns=(feature.name,),
                    )
                )
                continue

            if feature.name in engineered_df.columns:
                continue

            spec = self._specs.get(feature.name)
            if spec is None:
                if feature.required or context.purpose == DatasetPurpose.PREDICTION:
                    issues.append(
                        ValidationIssue(
                            code="MISSING_DERIVED_FEATURE_SPEC",
                            stage=PipelineStage.FEATURE_ENGINEERING,
                            severity=ValidationSeverity.ERROR,
                            impact=IssueImpact.BLOCKING,
                            message=(
                                f"Required derived feature '{feature.name}' has no "
                                "registered transformation spec."
                            ),
                            columns=(feature.name,),
                        )
                    )
                continue

            missing_deps = tuple(
                dep for dep in feature.derived_from if dep not in engineered_df.columns
            )
            if missing_deps:
                issues.append(
                    ValidationIssue(
                        code="MISSING_DERIVED_DEPENDENCIES",
                        stage=PipelineStage.FEATURE_ENGINEERING,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=(
                            f"Derived feature '{feature.name}' is missing input columns."
                        ),
                        columns=missing_deps,
                    )
                )
                continue

            computed = spec.compute(engineered_df)
            if len(computed) != len(engineered_df):
                issues.append(
                    ValidationIssue(
                        code="DERIVED_FEATURE_LENGTH_MISMATCH",
                        stage=PipelineStage.FEATURE_ENGINEERING,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=(
                            f"Derived feature '{feature.name}' produced wrong row count."
                        ),
                        columns=(feature.name,),
                    )
                )
                continue

            engineered_df[feature.name] = computed.to_numpy()

        # Maintain canonical column ordering from DatasetContract.
        canonical_cols = [
            feature.name
            for feature in contract.all_columns
            if feature.name in engineered_df.columns
        ]
        engineered_df = engineered_df[canonical_cols]

        report = ValidationReport(
            dataset_id=context.dataset_id,
            contract_version=context.contract_version,
            dataset_revision_id=artifact.metadata.revision_id,
            domain=context.domain,
            process=context.process,
            purpose=context.purpose,
            completed_stages=(PipelineStage.FEATURE_ENGINEERING,),
            issues=tuple(issues),
        )

        if report.status == ReadinessStatus.BLOCKED:
            return StageResult(
                context=context,
                stage=PipelineStage.FEATURE_ENGINEERING,
                status=ExecutionStatus.BLOCKED,
                input_artifact=artifact,
                output_artifact=None,
                report=report,
                timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
            )

        engineered_bytes = engineered_df.to_csv(index=False, lineterminator="\r\n").encode("utf-8")
        engineered_ref = ArtifactReference(
            factory_id=context.factory_id,
            format=ArtifactFormat.CSV,
            content_sha256=calculate_bytes_sha256(engineered_bytes),
            size_bytes=len(engineered_bytes),
        )
        self._storage.put(
            factory_id=context.factory_id,
            reference=engineered_ref,
            stream=BytesIO(engineered_bytes),
        )

        columns = tuple(
            DatasetColumnSchema(
                name=str(col),
                physical_dtype=str(engineered_df[col].dtype),
            )
            for col in engineered_df.columns
        )
        engineered_metadata = artifact.metadata.derive(
            context=context,
            state=DatasetState.ENGINEERED,
            row_count=len(engineered_df),
            columns=columns,
            operation="engineer_features",
            implementation_version=self._implementation_version,
            description="Generated deterministic row-level industrial features.",
        )
        output_artifact = DatasetArtifact(
            metadata=engineered_metadata,
            reference=engineered_ref,
        )
        return StageResult(
            context=context,
            stage=PipelineStage.FEATURE_ENGINEERING,
            status=ExecutionStatus.SUCCEEDED,
            input_artifact=artifact,
            output_artifact=output_artifact,
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )

    @staticmethod
    def _topological_sort(features: list) -> list:
        by_name = {f.name: f for f in features}
        visited: set[str] = set()
        ordered: list = []

        def visit(name: str) -> None:
            if name in visited or name not in by_name:
                return
            visited.add(name)
            for dep in by_name[name].derived_from:
                visit(dep)
            ordered.append(by_name[name])

        for feature in features:
            visit(feature.name)
        return ordered
