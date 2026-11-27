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
    FeatureAvailability,
    FeatureOrigin,
    IssueImpact,
    PipelineStage,
    ReadinessStatus,
    ValidationSeverity,
)
from taqamind.core.contracts.validation import ValidationIssue, ValidationReport
from taqamind.core.feature_engineering.row_local import (
    RowLocalTransform,
    validate_row_local_transform,
)
from taqamind.core.ingestion.artifact_csv import csv_configuration_sha256, read_verified_csv
from taqamind.core.ingestion.resource_policy import (
    CsvResourcePolicy,
    resolve_internal_csv_policy,
)
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
from taqamind.core.preprocessing.missing_values import normalize_text_missing
from taqamind.core.preprocessing.type_conversion import ContractTypeConverter
from taqamind.core.validation.column_validation import validate_column_with_conversion


@dataclass(frozen=True, slots=True)
class DerivedFeatureSpec:
    """Declarative deterministic row-level transformation for a derived feature."""

    feature_name: str
    compute: Callable[[pd.DataFrame], pd.Series] | None = None
    transform: RowLocalTransform | None = None

    def __post_init__(self) -> None:
        if (self.compute is None) == (self.transform is None):
            raise ValueError("Provide exactly one of 'compute' or 'transform'.")
        if self.transform is not None:
            validate_row_local_transform(self.transform)

    @property
    def is_restricted_row_local(self) -> bool:
        """Return True when defined via restricted vectorized row-local transform."""
        return self.transform is not None


class DeterministicFeatureEngineer(FeatureEngineer):
    """Compute leakage-safe row-level derived features defined by contract specs."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        *,
        derived_specs: tuple[DerivedFeatureSpec, ...] = (),
        allow_unverified_callbacks: bool = False,
        implementation_version: str = "0.6.0",
        delimiter: str = ",",
        max_rows: int | None = None,
        resource_policy: CsvResourcePolicy | None = None,
    ) -> None:
        names = [spec.feature_name for spec in derived_specs]
        if len(names) != len(set(names)):
            raise ValueError("DerivedFeatureSpec feature_name entries must be unique.")
        self._storage = storage
        self._specs = {spec.feature_name: spec for spec in derived_specs}
        self._allow_unverified_callbacks = allow_unverified_callbacks
        self._implementation_version = implementation_version
        self._resource_policy = resolve_internal_csv_policy(resource_policy, max_rows=max_rows)
        self._reader = SafeCsvReader(delimiter=delimiter, resource_policy=self._resource_policy)
        spec_fingerprint = {
            spec.feature_name: (
                spec.transform.model_dump(mode="json")
                if spec.transform is not None
                else "unverified_callback"
            )
            for spec in derived_specs
        }
        self._configuration_sha256 = csv_configuration_sha256(
            self._resource_policy,
            delimiter=delimiter,
            extra={
                "allow_unverified_callbacks": allow_unverified_callbacks,
                "derived_specs": spec_fingerprint,
            },
        )
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
                stage=PipelineStage.FEATURE_ENGINEERING,
                status=ExecutionStatus.FAILED,
                input_artifact=artifact,
                report=None,
                timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
                error=ExecutionError(code=exc.code, message=str(exc)),
            )

        issues: list[ValidationIssue] = []
        engineered_df = df.copy()
        column_map = contract.column_map
        target_name = contract.target.feature.name

        if context.purpose == DatasetPurpose.PREDICTION and target_name in engineered_df.columns:
            issues.append(
                ValidationIssue(
                    code="TARGET_PRESENT_IN_PREDICTION",
                    stage=PipelineStage.FEATURE_ENGINEERING,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message="Prediction dataset must not include the target column.",
                    columns=(target_name,),
                )
            )

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

        derived_features = [
            feature for feature in contract.features if feature.origin == FeatureOrigin.DERIVED
        ]
        ordered_derived, cycle_names = self._topological_sort(derived_features)
        if cycle_names:
            issues.append(
                ValidationIssue(
                    code="DERIVED_FEATURE_DEPENDENCY_CYCLE",
                    stage=PipelineStage.FEATURE_ENGINEERING,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message="Circular dependency detected among derived features.",
                    columns=tuple(cycle_names),
                )
            )

        # Drop supplied precomputed derived columns that have registered specs BEFORE
        # initial input type conversion so obsolete precomputed values cannot block
        # safe recomputation or leak into downstream dependencies.
        for feature in ordered_derived:
            if (
                context.purpose == DatasetPurpose.PREDICTION
                and not contract.is_available_at_prediction(feature)
            ):
                continue
            if feature.name in engineered_df.columns and feature.name in self._specs:
                engineered_df = engineered_df.drop(columns=[feature.name])

        # Convert remaining input columns by contract type before transformation.
        for col_name in engineered_df.columns:
            feature = column_map.get(str(col_name))
            if feature is None:
                continue
            if feature.data_type in {DataType.INTEGER, DataType.FLOAT, DataType.BOOLEAN}:
                conv = self._converter.convert(
                    engineered_df[col_name],
                    data_type=feature.data_type,
                )
                if conv.is_valid:
                    engineered_df[col_name] = conv.values
                else:
                    issues.append(
                        ValidationIssue(
                            code="FEATURE_ENGINEERING_INVALID_INPUT_TYPE",
                            stage=PipelineStage.FEATURE_ENGINEERING,
                            severity=ValidationSeverity.ERROR,
                            impact=IssueImpact.BLOCKING,
                            message=f"Column '{col_name}' contains invalid typed values.",
                            columns=(str(col_name),),
                            affected_rows=len(conv.invalid_positions),
                            row_samples=conv.invalid_positions[:20],
                        )
                    )

        # Pre-execution structural, leakage, spec, and dependency validation.
        planned_available = set(engineered_df.columns)
        executable_derived: list = []

        for feature in ordered_derived:
            if (
                context.purpose == DatasetPurpose.PREDICTION
                and not contract.is_available_at_prediction(feature)
            ):
                if feature.name in engineered_df.columns:
                    issues.append(
                        ValidationIssue(
                            code="UNAVAILABLE_FEATURE_IN_PREDICTION",
                            stage=PipelineStage.FEATURE_ENGINEERING,
                            severity=ValidationSeverity.CRITICAL,
                            impact=IssueImpact.BLOCKING,
                            message=(
                                f"Derived feature '{feature.name}' is not available "
                                "at prediction time."
                            ),
                            columns=(feature.name,),
                        )
                    )
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

            post_prod_deps = tuple(
                dep
                for dep in deps
                if dep in column_map
                and column_map[dep].availability == FeatureAvailability.POST_PRODUCTION
                and feature.availability == FeatureAvailability.PRE_PRODUCTION
            )
            if post_prod_deps:
                issues.append(
                    ValidationIssue(
                        code="DERIVED_FEATURE_AVAILABILITY_LEAKAGE",
                        stage=PipelineStage.FEATURE_ENGINEERING,
                        severity=ValidationSeverity.CRITICAL,
                        impact=IssueImpact.BLOCKING,
                        message=(
                            f"Pre-production derived feature '{feature.name}' depends on "
                            "post-production features."
                        ),
                        columns=post_prod_deps,
                    )
                )
                continue

            spec = self._specs.get(feature.name)
            if feature.name in engineered_df.columns and spec is None:
                issues.append(
                    ValidationIssue(
                        code="UNVERIFIED_PRECOMPUTED_FEATURE",
                        stage=PipelineStage.FEATURE_ENGINEERING,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=(
                            f"Precomputed derived feature '{feature.name}' "
                            "has no registered transformation spec."
                        ),
                        columns=(feature.name,),
                        recommendation=(
                            "Register a transformation spec to recompute "
                            "the feature from its declared dependencies."
                        ),
                    )
                )
                continue

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

            if spec.transform is not None:
                try:
                    validate_row_local_transform(spec.transform)
                except TypeError:
                    issues.append(
                        ValidationIssue(
                            code="UNSUPPORTED_TRANSFORM_EXPRESSION",
                            stage=PipelineStage.FEATURE_ENGINEERING,
                            severity=ValidationSeverity.CRITICAL,
                            impact=IssueImpact.BLOCKING,
                            message=(
                                f"Derived feature '{feature.name}' uses an unsupported "
                                "transform expression node."
                            ),
                            columns=(feature.name,),
                        )
                    )
                    continue
                undeclared = tuple(
                    sorted(spec.transform.referenced_columns() - set(feature.derived_from))
                )
                if undeclared:
                    issues.append(
                        ValidationIssue(
                            code="UNDECLARED_TRANSFORM_DEPENDENCY",
                            stage=PipelineStage.FEATURE_ENGINEERING,
                            severity=ValidationSeverity.CRITICAL,
                            impact=IssueImpact.BLOCKING,
                            message=(
                                f"Restricted transform for '{feature.name}' references "
                                "undeclared dependency columns."
                            ),
                            columns=undeclared,
                        )
                    )
                    continue
            else:
                if not self._allow_unverified_callbacks:
                    issues.append(
                        ValidationIssue(
                            code="UNVERIFIED_CALLBACK_BLOCKED",
                            stage=PipelineStage.FEATURE_ENGINEERING,
                            severity=ValidationSeverity.CRITICAL,
                            impact=IssueImpact.BLOCKING,
                            message=(
                                f"Unverified Python callback for '{feature.name}' is blocked "
                                "under strict pre-split leakage safety."
                            ),
                            columns=(feature.name,),
                            recommendation=(
                                "Use a restricted RowLocalTransform or explicitly opt in "
                                "to legacy callback compatibility mode."
                            ),
                        )
                    )
                    continue
                issues.append(
                    ValidationIssue(
                        code="UNVERIFIED_CALLBACK_COMPATIBILITY_MODE",
                        stage=PipelineStage.FEATURE_ENGINEERING,
                        severity=ValidationSeverity.WARNING,
                        impact=IssueImpact.NONE,
                        message=(
                            f"Derived feature '{feature.name}' uses an unverified Python "
                            "callback in compatibility mode; row-locality is not proven."
                        ),
                        columns=(feature.name,),
                    )
                )

            missing_deps = tuple(
                dep for dep in feature.derived_from if dep not in planned_available
            )
            if missing_deps:
                issues.append(
                    ValidationIssue(
                        code="MISSING_DERIVED_DEPENDENCIES",
                        stage=PipelineStage.FEATURE_ENGINEERING,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=f"Derived feature '{feature.name}' is missing input columns.",
                        columns=missing_deps,
                    )
                )
                continue

            planned_available.add(feature.name)
            executable_derived.append((feature, spec))

        # Fail closed before executing any transformations if structural/config blockers exist.
        if any(issue.impact == IssueImpact.BLOCKING for issue in issues):
            return self._blocked_result(
                context=context,
                artifact=artifact,
                issues=issues,
                started_at=started_at,
            )

        # Execute validated derived transforms in topological order; fail closed on first error.
        for feature, spec in executable_derived:
            dep_frame = engineered_df.loc[:, list(feature.derived_from)].copy()
            try:
                if spec.transform is not None:
                    computed = spec.transform.evaluate(dep_frame)
                else:
                    assert spec.compute is not None
                    computed = spec.compute(dep_frame)
            except MemoryError:
                raise
            except Exception as exc:
                issues.append(
                    ValidationIssue(
                        code="DERIVED_FEATURE_COMPUTATION_FAILED",
                        stage=PipelineStage.FEATURE_ENGINEERING,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=(
                            f"Computation failed for derived feature "
                            f"'{feature.name}' "
                            f"({type(exc).__name__})."
                        ),
                        columns=(feature.name,),
                        recommendation=(
                            "Review the registered transformation and its declared dependencies."
                        ),
                    )
                )
                break

            if not isinstance(computed, pd.Series):
                issues.append(
                    ValidationIssue(
                        code="DERIVED_FEATURE_INVALID_TYPE",
                        stage=PipelineStage.FEATURE_ENGINEERING,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=f"Derived feature '{feature.name}' must return a pandas Series.",
                        columns=(feature.name,),
                    )
                )
                break

            if len(computed) != len(engineered_df) or not computed.index.equals(
                engineered_df.index
            ):
                issues.append(
                    ValidationIssue(
                        code="DERIVED_FEATURE_LENGTH_MISMATCH",
                        stage=PipelineStage.FEATURE_ENGINEERING,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=(
                            f"Derived feature '{feature.name}' produced wrong row count or index."
                        ),
                        columns=(feature.name,),
                    )
                )
                break

            normalized = normalize_text_missing(
                computed,
                data_type=feature.data_type,
            )

            output_issues, conversion = validate_column_with_conversion(
                normalized,
                feature,
                stage=PipelineStage.FEATURE_ENGINEERING,
                converter=self._converter,
            )
            issues.extend(output_issues)

            if any(issue.impact == IssueImpact.BLOCKING for issue in output_issues):
                break

            engineered_df[feature.name] = (
                conversion.values if conversion is not None else normalized
            )

        if any(issue.impact == IssueImpact.BLOCKING for issue in issues):
            return self._blocked_result(
                context=context,
                artifact=artifact,
                issues=issues,
                started_at=started_at,
            )

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
            configuration_sha256=self._configuration_sha256,
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
    def _blocked_result(
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        issues: list[ValidationIssue],
        started_at: datetime,
    ) -> StageResult:
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
        return StageResult(
            context=context,
            stage=PipelineStage.FEATURE_ENGINEERING,
            status=ExecutionStatus.BLOCKED,
            input_artifact=artifact,
            output_artifact=None,
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )

    @staticmethod
    def _topological_sort(features: list) -> tuple[list, list[str]]:
        by_name = {f.name: f for f in features}
        state: dict[str, int] = {}  # 0/absent=unvisited, 1=visiting, 2=visited
        ordered: list = []
        cycles: list[str] = []

        def visit(name: str) -> None:
            if name not in by_name:
                return
            st = state.get(name, 0)
            if st == 2:
                return
            if st == 1:
                if name not in cycles:
                    cycles.append(name)
                return
            state[name] = 1
            for dep in by_name[name].derived_from:
                visit(dep)
            state[name] = 2
            ordered.append(by_name[name])

        for feature in features:
            visit(feature.name)
        return ordered, cycles
