"""Thin iron_manufacturing wrappers around real Shared Core P1.2/P1.3/P1.5 engines.

Uses authoritative Shared Core engines directly:
    - `DeterministicColumnMapper` (taqamind.core.mapping)
    - `ContractSemanticValidator` (taqamind.core.validation)
    - `DeterministicFeatureEngineer` + `DerivedFeatureSpec` (taqamind.core.feature_engineering)

Thin wrappers exist strictly for:
    - Core AUD-001 mitigation: verified SI unit conversions during column mapping.
    - Core AUD-004 mitigation: string/category dtype preservation to avoid leading-zero loss.
    - Domain-specific iron rules: ambiguous raw header blocking, prediction-mode rejection of
      target/post-production columns, iron physical mass-balance checks, and per-process
      `DeterministicFeatureEngineer` routing.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from io import BytesIO

import pandas as pd

from taqamind.core.contracts import (
    DatasetContract,
    DatasetPurpose,
    DataType,
    IssueImpact,
    PipelineStage,
    ReadinessStatus,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from taqamind.core.feature_engineering import DerivedFeatureSpec, DeterministicFeatureEngineer
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
    ColumnMapper,
    FeatureEngineer,
    SemanticValidator,
    validate_approved_mapping,
    validate_processing_input,
)
from taqamind.core.interfaces.processing_models import (
    ApprovedMapping,
    ColumnMappingEntry,
    MappingMethod,
    MappingProposal,
)
from taqamind.core.interfaces.results import (
    ExecutionStatus,
    StageResult,
    StageTiming,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage
from taqamind.core.mapping import DeterministicColumnMapper
from taqamind.core.validation import ContractSemanticValidator
from taqamind.domains.iron_manufacturing.common.normalization import normalize_header
from taqamind.domains.iron_manufacturing.common.units import (
    UnitConversionError,
    convert_series_unit,
    get_conversion_factor,
)
from taqamind.domains.iron_manufacturing.config import AMBIGUOUS_RAW_HEADERS


class IronUnitAwareColumnMapper(ColumnMapper):
    """Wrap Shared Core DeterministicColumnMapper for AUD-001, AUD-004, and iron overrides."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        *,
        inner_mapper: DeterministicColumnMapper | None = None,
        explicit_column_overrides: dict[str, str] | None = None,
        explicit_unit_overrides: dict[str, str] | None = None,
        forbidden_predictors_by_process: dict[str, frozenset[str]] | None = None,
        implementation_version: str = "0.1.0",
    ) -> None:
        self._storage = storage
        self._inner = inner_mapper or DeterministicColumnMapper(
            storage, implementation_version=implementation_version
        )
        self._column_overrides = dict(explicit_column_overrides or {})
        self._unit_overrides = dict(explicit_unit_overrides or {})
        self._forbidden_by_process = dict(forbidden_predictors_by_process or {})
        self._implementation_version = implementation_version

    def propose_mapping(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> MappingProposal:
        base_proposal = self._inner.propose_mapping(
            context=context, artifact=artifact, contract=contract
        )
        source_columns = [column.name for column in artifact.metadata.columns]

        target_name = contract.target.feature.name
        candidate_features = {
            f.name: f
            for f in contract.all_columns
            if not (context.purpose == DatasetPurpose.PREDICTION and f.name == target_name)
        }
        norm_alias_lookup = {
            normalize_header(alias): feat
            for feat in candidate_features.values()
            for alias in feat.aliases
        }

        entries_by_source: dict[str, ColumnMappingEntry] = {}
        claimed: set[str] = set()

        # Keep Core exact/alias/rule matches unless source header is explicitly ambiguous
        for entry in base_proposal.entries:
            norm_src = normalize_header(entry.source_column)
            if (
                entry.source_column not in self._column_overrides
                and norm_src in AMBIGUOUS_RAW_HEADERS
                and entry.source_column != entry.canonical_column
            ):
                continue
            source_unit = self._unit_overrides.get(
                entry.source_column,
                self._unit_overrides.get(entry.canonical_column, entry.source_unit),
            )
            unit_changed = (
                entry.canonical_unit is not None
                and source_unit is not None
                and source_unit != entry.canonical_unit
            )
            if unit_changed:
                entry = ColumnMappingEntry(
                    source_column=entry.source_column,
                    canonical_column=entry.canonical_column,
                    method=entry.method,
                    source_unit=source_unit,
                    canonical_unit=entry.canonical_unit,
                    requires_confirmation=True,
                    reason=(
                        f"{entry.reason} Unit conversion declared: "
                        f"'{source_unit}' -> '{entry.canonical_unit}'."
                    ),
                )
            entries_by_source[entry.source_column] = entry
            claimed.add(entry.canonical_column)

        # Apply explicit factory overrides & normalized bilingual EN/AR aliases on remaining columns
        for src in source_columns:
            if src in entries_by_source:
                continue
            norm_src = normalize_header(src)

            if src in self._column_overrides:
                canon = self._column_overrides[src]
                feat = candidate_features.get(canon)
                if feat is not None and canon not in claimed:
                    src_unit = self._unit_overrides.get(
                        src, self._unit_overrides.get(canon, feat.unit)
                    )
                    claimed.add(canon)
                    entries_by_source[src] = ColumnMappingEntry(
                        source_column=src,
                        canonical_column=canon,
                        method=MappingMethod.MANUAL,
                        source_unit=src_unit,
                        canonical_unit=feat.unit,
                        requires_confirmation=True,
                        reason=f"Explicit factory mapping override for '{canon}'.",
                    )
                continue

            if norm_src in AMBIGUOUS_RAW_HEADERS:
                continue

            feat = norm_alias_lookup.get(norm_src)
            if feat is not None and feat.name not in claimed:
                src_unit = self._unit_overrides.get(
                    src, self._unit_overrides.get(feat.name, feat.unit)
                )
                claimed.add(feat.name)
                entries_by_source[src] = ColumnMappingEntry(
                    source_column=src,
                    canonical_column=feat.name,
                    method=MappingMethod.ALIAS,
                    source_unit=src_unit,
                    canonical_unit=feat.unit,
                    requires_confirmation=True,
                    reason=f"Matched normalized EN/AR alias for '{feat.name}'.",
                )

        entries = tuple(
            entries_by_source[src] for src in source_columns if src in entries_by_source
        )
        unmapped = tuple(src for src in source_columns if src not in entries_by_source)
        return MappingProposal(
            factory_id=context.factory_id,
            execution_id=context.execution_id,
            dataset_revision_id=artifact.metadata.revision_id,
            entries=entries,
            unmapped_columns=unmapped,
        )

    @staticmethod
    def _string_dtype_hints(
        mapping: ApprovedMapping,
        contract: DatasetContract,
    ) -> dict[str, type[str]]:
        col_map = contract.column_map
        return {
            entry.source_column: str
            for entry in mapping.proposal.entries
            if col_map[entry.canonical_column].data_type in {DataType.STRING, DataType.CATEGORY}
        }

    def _blocked_result(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        started_at: datetime,
        issues: list[ValidationIssue],
    ) -> StageResult:
        report = ValidationReport(
            dataset_id=context.dataset_id,
            contract_version=context.contract_version,
            dataset_revision_id=artifact.metadata.revision_id,
            domain=context.domain,
            process=context.process,
            purpose=context.purpose,
            completed_stages=(PipelineStage.COLUMN_MAPPING,),
            issues=tuple(issues),
        )
        return StageResult(
            context=context,
            stage=PipelineStage.COLUMN_MAPPING,
            status=ExecutionStatus.BLOCKED,
            input_artifact=artifact,
            output_artifact=None,
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )

    def apply_mapping(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        mapping: ApprovedMapping,
    ) -> StageResult:
        started_at = datetime.now(UTC)
        validate_approved_mapping(
            context=context,
            artifact=artifact,
            contract=contract,
            mapping=mapping,
        )

        # Reject prediction uploads that include target or forbidden post-production columns
        if context.purpose == DatasetPurpose.PREDICTION:
            forbidden = set(self._forbidden_by_process.get(contract.process, frozenset()))
            forbidden.add(contract.target.feature.name)
            forbidden_norm = {normalize_header(name) for name in forbidden}
            raw_headers = [col.name for col in artifact.metadata.columns]
            leaked_cols = tuple(
                col
                for col in raw_headers
                if col in forbidden or normalize_header(col) in forbidden_norm
            )
            if leaked_cols:
                return self._blocked_result(
                    context=context,
                    artifact=artifact,
                    started_at=started_at,
                    issues=[
                        ValidationIssue(
                            code="FORBIDDEN_POST_PRODUCTION_COLUMNS",
                            stage=PipelineStage.COLUMN_MAPPING,
                            severity=ValidationSeverity.CRITICAL,
                            impact=IssueImpact.BLOCKING,
                            message=(
                                "Prediction dataset contains target or post-production leakage "
                                f"columns: {list(leaked_cols)}."
                            ),
                            columns=leaked_cols,
                            recommendation=(
                                "Remove target and realized post-production columns before "
                                "running pre-production inference."
                            ),
                        )
                    ],
                )

        # Ensure required columns are mapped (catches unmapped ambiguous headers)
        mapped_canonicals = {entry.canonical_column for entry in mapping.proposal.entries}
        if context.purpose != DatasetPurpose.RESEARCH:
            missing_req = tuple(
                col
                for col in contract.required_columns(context.purpose)
                if col not in mapped_canonicals
            )
            if missing_req:
                return self._blocked_result(
                    context=context,
                    artifact=artifact,
                    started_at=started_at,
                    issues=[
                        ValidationIssue(
                            code="MISSING_REQUIRED_MAPPED_COLUMNS",
                            stage=PipelineStage.COLUMN_MAPPING,
                            severity=ValidationSeverity.CRITICAL,
                            impact=IssueImpact.BLOCKING,
                            message=(
                                "Required contract columns were not mapped: "
                                f"{list(missing_req)}."
                            ),
                            columns=missing_req,
                            recommendation=(
                                "Provide explicit column overrides for ambiguous or unmapped "
                                "headers."
                            ),
                        )
                    ],
                )

        entries = mapping.proposal.entries
        needs_conversion = any(
            entry.source_unit != entry.canonical_unit
            for entry in entries
            if entry.canonical_unit is not None
        )
        dtype_hints = self._string_dtype_hints(mapping, contract)

        # Delegate directly to Core DeterministicColumnMapper when no unit conversion is required
        # and no string/category columns need AUD-004 preservation override.
        if not needs_conversion and not dtype_hints:
            return self._inner.apply_mapping(
                context=context,
                artifact=artifact,
                contract=contract,
                mapping=mapping,
            )

        # Validate every declared unit conversion before transforming values (AUD-001)
        unit_issues: list[ValidationIssue] = []
        for entry in entries:
            if entry.canonical_unit is None:
                continue
            if entry.source_unit is None:
                unit_issues.append(
                    ValidationIssue(
                        code="INVALID_UNIT_CONVERSION",
                        stage=PipelineStage.COLUMN_MAPPING,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=(
                            f"Column '{entry.canonical_column}' requires canonical unit "
                            f"'{entry.canonical_unit}' but source_unit is missing."
                        ),
                        columns=(entry.canonical_column,),
                    )
                )
                continue
            if entry.source_unit != entry.canonical_unit:
                try:
                    get_conversion_factor(entry.source_unit, entry.canonical_unit)
                except UnitConversionError as exc:
                    unit_issues.append(
                        ValidationIssue(
                            code="INVALID_UNIT_CONVERSION",
                            stage=PipelineStage.COLUMN_MAPPING,
                            severity=ValidationSeverity.ERROR,
                            impact=IssueImpact.BLOCKING,
                            message=str(exc),
                            columns=(entry.canonical_column,),
                        )
                    )
        if unit_issues:
            return self._blocked_result(
                context=context,
                artifact=artifact,
                started_at=started_at,
                issues=unit_issues,
            )

        # AUD-004: preserve string/category columns verbatim when reading CSV
        with self._storage.open_verified(
            factory_id=context.factory_id,
            reference=artifact.reference,
        ) as stream:
            df = pd.read_csv(stream, keep_default_na=True, dtype=dtype_hints)

        rename_map = {entry.source_column: entry.canonical_column for entry in entries}
        canonical_order = [
            f.name for f in contract.all_columns if f.name in set(rename_map.values())
        ]
        mapped_df = df[list(rename_map.keys())].rename(columns=rename_map)[canonical_order].copy()

        for entry in entries:
            if (
                entry.canonical_unit is not None
                and entry.source_unit is not None
                and entry.source_unit != entry.canonical_unit
            ):
                mapped_df[entry.canonical_column] = convert_series_unit(
                    mapped_df[entry.canonical_column],
                    entry.source_unit,
                    entry.canonical_unit,
                )

        mapped_bytes = mapped_df.to_csv(index=False).encode("utf-8")
        mapped_ref = ArtifactReference(
            factory_id=context.factory_id,
            format=ArtifactFormat.CSV,
            content_sha256=calculate_bytes_sha256(mapped_bytes),
            size_bytes=len(mapped_bytes),
        )
        self._storage.put(
            factory_id=context.factory_id,
            reference=mapped_ref,
            stream=BytesIO(mapped_bytes),
        )

        config_payload = json.dumps(
            mapping.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        columns = tuple(
            DatasetColumnSchema(name=str(col), physical_dtype=str(mapped_df[col].dtype))
            for col in mapped_df.columns
        )
        mapped_metadata = artifact.metadata.derive(
            context=context,
            state=DatasetState.MAPPED,
            row_count=len(mapped_df),
            columns=columns,
            operation="apply_mapping",
            implementation_version=self._implementation_version,
            configuration_sha256=calculate_bytes_sha256(config_payload),
            description=(
                "Applied approved canonical column mapping with verified SI unit "
                "conversions (AUD-001) and string/category dtype preservation (AUD-004)."
            ),
        )
        output_artifact = DatasetArtifact(metadata=mapped_metadata, reference=mapped_ref)

        issues: list[ValidationIssue] = []
        if mapping.proposal.unmapped_columns:
            issues.append(
                ValidationIssue(
                    code="UNMAPPED_SOURCE_COLUMNS_DROPPED",
                    stage=PipelineStage.COLUMN_MAPPING,
                    severity=ValidationSeverity.INFO,
                    impact=IssueImpact.NONE,
                    message="Unmapped source columns were excluded from the mapped revision.",
                    columns=mapping.proposal.unmapped_columns,
                )
            )
        report = ValidationReport(
            dataset_id=context.dataset_id,
            contract_version=context.contract_version,
            dataset_revision_id=artifact.metadata.revision_id,
            domain=context.domain,
            process=context.process,
            purpose=context.purpose,
            completed_stages=(PipelineStage.COLUMN_MAPPING,),
            issues=tuple(issues),
        )
        return StageResult(
            context=context,
            stage=PipelineStage.COLUMN_MAPPING,
            status=ExecutionStatus.SUCCEEDED,
            input_artifact=artifact,
            output_artifact=output_artifact,
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )


class IronSemanticValidator(SemanticValidator):
    """Compose Shared Core ContractSemanticValidator with iron physical mass-balance rules."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        *,
        inner_validator: ContractSemanticValidator | None = None,
        process_validators: (
            dict[str, Callable[[pd.DataFrame], list[ValidationIssue]]] | None
        ) = None,
    ) -> None:
        self._storage = storage
        self._inner = inner_validator or ContractSemanticValidator(storage)
        self._process_validators = dict(process_validators or {})

    def validate(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> StageResult:
        core_result = self._inner.validate(
            context=context, artifact=artifact, contract=contract
        )
        validator_fn = self._process_validators.get(contract.process)
        if validator_fn is None or core_result.report is None:
            return core_result

        with self._storage.open_verified(
            factory_id=context.factory_id,
            reference=artifact.reference,
        ) as stream:
            df = pd.read_csv(stream, keep_default_na=True)

        extra_issues = validator_fn(df)
        if not extra_issues:
            return core_result

        merged_report = ValidationReport(
            dataset_id=core_result.report.dataset_id,
            contract_version=core_result.report.contract_version,
            dataset_revision_id=core_result.report.dataset_revision_id,
            domain=core_result.report.domain,
            process=core_result.report.process,
            purpose=core_result.report.purpose,
            completed_stages=core_result.report.completed_stages,
            issues=(*core_result.report.issues, *extra_issues),
        )
        status = (
            ExecutionStatus.BLOCKED
            if merged_report.status == ReadinessStatus.BLOCKED
            else ExecutionStatus.SUCCEEDED
        )
        return StageResult(
            context=context,
            stage=PipelineStage.SEMANTIC_VALIDATION,
            status=status,
            input_artifact=artifact,
            output_artifact=None,
            report=merged_report,
            timing=core_result.timing,
        )


class ProcessFeatureEngineerRouter(FeatureEngineer):
    """Dispatch each process to its dedicated Shared Core DeterministicFeatureEngineer."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        *,
        specs_by_process: dict[str, tuple[DerivedFeatureSpec, ...]],
        implementation_version: str = "0.1.0",
    ) -> None:
        self._engineers: dict[str, DeterministicFeatureEngineer] = {
            process_id: DeterministicFeatureEngineer(
                storage,
                derived_specs=specs,
                implementation_version=implementation_version,
            )
            for process_id, specs in specs_by_process.items()
        }

    def get_engineer(self, process_id: str) -> DeterministicFeatureEngineer:
        """Return the real Shared Core DeterministicFeatureEngineer for a process."""
        return self._engineers[process_id]

    def engineer(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> StageResult:
        validate_processing_input(
            context=context,
            artifact=artifact,
            contract=contract,
            expected_state=DatasetState.CLEANED,
        )
        return self._engineers[contract.process].engineer(
            context=context,
            artifact=artifact,
            contract=contract,
        )
