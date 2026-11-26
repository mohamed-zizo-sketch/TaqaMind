"""Readiness, dataset splitting, leakage validation, and split/usage registry adapters."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import numpy as np
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
from taqamind.core.interfaces import (
    DatasetArtifact,
    DatasetState,
    ExecutionStatus,
    LeakageCheck,
    LeakageCheckResult,
    LeakageCheckStatus,
    LeakageValidationResult,
    LeakageValidator,
    PipelineContext,
    RecordLineageRegistry,
    RecordScope,
    RecordUsage,
    RecordUsageRegistry,
    RegistryConflictError,
    RegistryNotFoundError,
    RegistryScopeError,
    SplitConfiguration,
    SplitManifest,
    SplitPartition,
    SplitRegistry,
    SplitRole,
    SplitStrategy,
    StageResult,
    StageTiming,
    TrainingAuthorization,
    TrainingReadinessChecker,
    TrainingReadinessResult,
    calculate_bytes_sha256,
    find_previously_exposed_test_records,
    validate_processing_input,
)
from taqamind.core.interfaces.ml import DatasetSplitter
from taqamind.core.registry import LocalDatasetArtifactStorage
from taqamind.domains.textile.adapters._store import (
    atomic_write_text,
    dump_json_model,
    iter_json_models,
    load_json_model,
    safe_resolve,
)


def _record_scope_from_context(
    context: PipelineContext,
    identity_namespace: str = "production_record",
) -> RecordScope:
    return RecordScope(
        factory_id=context.factory_id,
        domain=context.domain,
        process=context.process,
        dataset_id=context.dataset_id,
        identity_namespace=identity_namespace,
    )


def _read_engineered_frame(
    storage: LocalDatasetArtifactStorage,
    artifact: DatasetArtifact,
    contract: DatasetContract,
) -> pd.DataFrame:
    dtype_hints = {
        f.name: "string"
        for f in contract.all_columns
        if f.data_type in {DataType.STRING, DataType.CATEGORY, DataType.DATE, DataType.DATETIME}
    }
    with storage.open_verified(
        factory_id=artifact.metadata.factory_id,
        reference=artifact.reference,
    ) as stream:
        return pd.read_csv(stream, keep_default_na=True, dtype=dtype_hints)


class TextileTrainingReadinessChecker(TrainingReadinessChecker):
    """Assess whether an engineered dataset is ready for splitting and training.

    Migration path: Replace with Shared Core P1.6 `TrainingReadinessChecker`.
    """

    def __init__(
        self,
        storage: LocalDatasetArtifactStorage,
        lineage_registry: RecordLineageRegistry,
        *,
        min_rows: int = 12,
        min_positive_targets: int = 2,
        identity_namespace: str = "production_record",
    ) -> None:
        self._storage = storage
        self._lineage = lineage_registry
        self._min_rows = min_rows
        self._min_positive_targets = min_positive_targets
        self._identity_namespace = identity_namespace

    def assess(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> TrainingReadinessResult:
        started_at = datetime.now(UTC)
        validate_processing_input(
            context=context,
            artifact=artifact,
            contract=contract,
            expected_state=DatasetState.ENGINEERED,
        )
        if context.purpose != DatasetPurpose.TRAINING:
            raise ValueError("Training readiness requires a training context.")

        issues: list[ValidationIssue] = []

        if not contract.target.is_confirmed:
            issues.append(
                ValidationIssue(
                    code="TARGET_SEMANTICS_UNCONFIRMED",
                    stage=PipelineStage.READINESS,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message=(
                        "Target semantics are unconfirmed; production training is blocked."
                    ),
                    columns=(contract.target.feature.name,),
                )
            )

        df = _read_engineered_frame(self._storage, artifact, contract)
        target_name = contract.target.feature.name

        if len(df) < self._min_rows:
            issues.append(
                ValidationIssue(
                    code="INSUFFICIENT_ROW_COUNT",
                    stage=PipelineStage.READINESS,
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    message=(
                        f"Dataset has {len(df)} rows; minimum required is {self._min_rows}."
                    ),
                )
            )

        if target_name not in df.columns:
            issues.append(
                ValidationIssue(
                    code="MISSING_TRAINING_TARGET",
                    stage=PipelineStage.READINESS,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message=f"Target column '{target_name}' is missing.",
                    columns=(target_name,),
                )
            )
        else:
            y = pd.to_numeric(df[target_name], errors="coerce")
            if bool(y.isna().any()):
                issues.append(
                    ValidationIssue(
                        code="MISSING_OR_INVALID_TARGET_VALUES",
                        stage=PipelineStage.READINESS,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message="Target contains missing or non-numeric values.",
                        columns=(target_name,),
                        affected_rows=int(y.isna().sum()),
                    )
                )
            elif int((y > 0).sum()) < self._min_positive_targets:
                issues.append(
                    ValidationIssue(
                        code="INSUFFICIENT_POSITIVE_TARGETS",
                        stage=PipelineStage.READINESS,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message="Insufficient positive target events for regression.",
                        columns=(target_name,),
                    )
                )

        scope = _record_scope_from_context(context, self._identity_namespace)
        try:
            manifest = self._lineage.get_membership(
                scope=scope,
                dataset_revision_id=artifact.metadata.revision_id,
            )
            manifest.validate_artifact(artifact)
        except (RegistryNotFoundError, ValueError) as exc:
            issues.append(
                ValidationIssue(
                    code="MISSING_OR_INVALID_MEMBERSHIP",
                    stage=PipelineStage.READINESS,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message=f"Registered record membership unavailable or invalid: {exc}",
                )
            )

        report = ValidationReport(
            dataset_id=context.dataset_id,
            contract_version=context.contract_version,
            dataset_revision_id=artifact.metadata.revision_id,
            domain=context.domain,
            process=context.process,
            purpose=context.purpose,
            completed_stages=(PipelineStage.READINESS,),
            issues=tuple(issues),
        )
        status = (
            ExecutionStatus.BLOCKED
            if report.status == ReadinessStatus.BLOCKED
            else ExecutionStatus.SUCCEEDED
        )
        stage_result = StageResult(
            context=context,
            stage=PipelineStage.READINESS,
            status=status,
            input_artifact=artifact,
            output_artifact=None,
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )
        return TrainingReadinessResult(result=stage_result, contract=contract)


class TextileDatasetSplitter(DatasetSplitter):
    """Create disjoint group/temporal/random splits bound to registered record IDs.

    Migration path: Replace with Shared Core P1.6 `DatasetSplitter`.
    """

    def __init__(
        self,
        storage: LocalDatasetArtifactStorage,
        lineage_registry: RecordLineageRegistry,
        *,
        identity_namespace: str = "production_record",
    ) -> None:
        self._storage = storage
        self._lineage = lineage_registry
        self._identity_namespace = identity_namespace

    def split(
        self,
        *,
        readiness: TrainingReadinessResult,
        configuration: SplitConfiguration,
    ) -> SplitManifest:
        readiness.require_split_eligibility()
        context = readiness.result.context
        artifact = readiness.result.input_artifact
        contract = readiness.contract

        scope = _record_scope_from_context(context, self._identity_namespace)
        membership = self._lineage.get_membership(
            scope=scope,
            dataset_revision_id=artifact.metadata.revision_id,
        )
        membership.validate_artifact(artifact)
        if membership.has_repeated_record_ids:
            raise ValueError(
                "Dataset membership contains repeated record_ids; resolve duplicates before split."
            )

        df = _read_engineered_frame(self._storage, artifact, contract)
        row_record_ids = [item.record_id for item in membership.memberships]

        if configuration.strategy in {SplitStrategy.TEMPORAL, SplitStrategy.GROUP_TEMPORAL}:
            if contract.time_column is None:
                raise ValueError("Temporal split requires DatasetContract.time_column.")
        if configuration.strategy in {SplitStrategy.GROUP, SplitStrategy.GROUP_TEMPORAL}:
            if not contract.grouping_columns:
                raise ValueError("Group split requires DatasetContract.grouping_columns.")

        if configuration.strategy == SplitStrategy.GROUP:
            train_ids, val_ids, test_ids = self._split_group(
                df, row_record_ids, contract.grouping_columns, configuration
            )
        elif configuration.strategy == SplitStrategy.TEMPORAL:
            assert contract.time_column is not None
            train_ids, val_ids, test_ids = self._split_temporal(
                df, row_record_ids, contract.time_column, configuration
            )
        elif configuration.strategy == SplitStrategy.GROUP_TEMPORAL:
            assert contract.time_column is not None
            train_ids, val_ids, test_ids = self._split_group_temporal(
                df,
                row_record_ids,
                contract.grouping_columns,
                contract.time_column,
                configuration,
            )
        else:
            train_ids, val_ids, test_ids = self._split_random(row_record_ids, configuration)

        return SplitManifest(
            context=context,
            dataset_revision_id=artifact.metadata.revision_id,
            configuration=configuration,
            train=SplitPartition(role=SplitRole.TRAIN, record_ids=tuple(train_ids)),
            validation=SplitPartition(role=SplitRole.VALIDATION, record_ids=tuple(val_ids)),
            test=SplitPartition(role=SplitRole.TEST, record_ids=tuple(test_ids)),
        )

    @staticmethod
    def _allocate_blocks(
        blocks: list[list[UUID]],
        total_rows: int,
        configuration: SplitConfiguration,
    ) -> tuple[list[UUID], list[UUID], list[UUID]]:
        if len(blocks) < 3:
            raise ValueError("At least 3 distinct groups/timestamps are required for 3-way split.")
        target_train = max(1, round(total_rows * configuration.train_ratio))
        target_val = max(1, round(total_rows * configuration.validation_ratio))

        train: list[UUID] = []
        val: list[UUID] = []
        test: list[UUID] = []

        # Reserve last two blocks initially if needed so validation and test remain non-empty.
        for idx, block in enumerate(blocks):
            remaining_blocks = len(blocks) - idx
            if not test and remaining_blocks == 1:
                test.extend(block)
            elif not val and remaining_blocks == 2 and len(blocks) - idx >= 2:
                val.extend(block)
            elif len(train) < target_train and remaining_blocks > 2:
                train.extend(block)
            elif len(val) < target_val and remaining_blocks > 1:
                val.extend(block)
            else:
                test.extend(block)

        if not train or not val or not test:
            raise ValueError("Unable to populate non-empty train/validation/test partitions.")
        return train, val, test

    def _split_group(
        self,
        df: pd.DataFrame,
        record_ids: list[UUID],
        grouping_columns: tuple[str, ...],
        configuration: SplitConfiguration,
    ) -> tuple[list[UUID], list[UUID], list[UUID]]:
        groups: dict[tuple[str, ...], list[UUID]] = {}
        for idx, rid in enumerate(record_ids):
            key = tuple(str(df.iloc[idx][col]) for col in grouping_columns)
            groups.setdefault(key, []).append(rid)

        sorted_keys = sorted(groups.keys())
        rng = np.random.default_rng(configuration.random_seed)
        perm = rng.permutation(len(sorted_keys))
        ordered_blocks = [groups[sorted_keys[i]] for i in perm]
        return self._allocate_blocks(ordered_blocks, len(record_ids), configuration)

    def _split_temporal(
        self,
        df: pd.DataFrame,
        record_ids: list[UUID],
        time_column: str,
        configuration: SplitConfiguration,
    ) -> tuple[list[UUID], list[UUID], list[UUID]]:
        timestamps = pd.to_datetime(df[time_column], format="mixed", utc=True)
        # Group identical timestamps together so ties never straddle a split boundary.
        buckets: dict[pd.Timestamp, list[UUID]] = {}
        for idx, rid in enumerate(record_ids):
            ts = timestamps.iloc[idx]
            buckets.setdefault(ts, []).append(rid)
        ordered_blocks = [buckets[ts] for ts in sorted(buckets.keys())]
        return self._allocate_blocks(ordered_blocks, len(record_ids), configuration)

    def _split_group_temporal(
        self,
        df: pd.DataFrame,
        record_ids: list[UUID],
        grouping_columns: tuple[str, ...],
        time_column: str,
        configuration: SplitConfiguration,
    ) -> tuple[list[UUID], list[UUID], list[UUID]]:
        timestamps = pd.to_datetime(df[time_column], format="mixed", utc=True)
        groups: dict[tuple[str, ...], tuple[pd.Timestamp, list[UUID]]] = {}
        for idx, rid in enumerate(record_ids):
            key = tuple(str(df.iloc[idx][col]) for col in grouping_columns)
            ts = timestamps.iloc[idx]
            if key not in groups:
                groups[key] = (ts, [rid])
            else:
                max_ts, ids = groups[key]
                ids.append(rid)
                groups[key] = (max(max_ts, ts), ids)
        ordered = sorted(groups.items(), key=lambda item: (item[1][0], item[0]))
        ordered_blocks = [item[1][1] for item in ordered]
        return self._allocate_blocks(ordered_blocks, len(record_ids), configuration)

    def _split_random(
        self,
        record_ids: list[UUID],
        configuration: SplitConfiguration,
    ) -> tuple[list[UUID], list[UUID], list[UUID]]:
        rng = np.random.default_rng(configuration.random_seed)
        perm = rng.permutation(len(record_ids))
        blocks = [[record_ids[i]] for i in perm]
        return self._allocate_blocks(blocks, len(record_ids), configuration)


class TextileLeakageValidator(LeakageValidator):
    """Evaluate all required leakage checks against membership, contract, and usage history.

    Migration path: Replace with Shared Core P1.6 `LeakageValidator`.
    """

    def __init__(
        self,
        storage: LocalDatasetArtifactStorage,
        lineage_registry: RecordLineageRegistry,
        usage_registry: RecordUsageRegistry,
        *,
        identity_namespace: str = "production_record",
    ) -> None:
        self._storage = storage
        self._lineage = lineage_registry
        self._usage = usage_registry
        self._identity_namespace = identity_namespace

    def validate(
        self,
        *,
        readiness: TrainingReadinessResult,
        manifest: SplitManifest,
    ) -> LeakageValidationResult:
        manifest.validate_readiness(readiness)
        context = manifest.context
        contract = readiness.contract
        artifact = readiness.result.input_artifact
        scope = _record_scope_from_context(context, self._identity_namespace)

        membership = self._lineage.get_membership(
            scope=scope,
            dataset_revision_id=manifest.dataset_revision_id,
        )
        membership.validate_artifact(artifact)
        df = _read_engineered_frame(self._storage, artifact, contract)

        checks: list[LeakageCheckResult] = []
        issues: list[ValidationIssue] = []

        # 1. RECORD_OVERLAP
        train_set = set(manifest.train.record_ids)
        val_set = set(manifest.validation.record_ids)
        test_set = set(manifest.test.record_ids)
        overlap = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
        if overlap:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.RECORD_OVERLAP,
                    status=LeakageCheckStatus.FAILED,
                    reason="Overlapping record_ids across split partitions.",
                )
            )
            issues.append(self._blocking_issue("LEAKAGE_RECORD_OVERLAP", "Overlapping records."))
        else:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.RECORD_OVERLAP, status=LeakageCheckStatus.PASSED
                )
            )

        # 2. RECORD_COVERAGE
        expected_ids = tuple(item.record_id for item in membership.memberships)
        try:
            manifest.validate_record_coverage(expected_ids)
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.RECORD_COVERAGE, status=LeakageCheckStatus.PASSED
                )
            )
        except ValueError as exc:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.RECORD_COVERAGE,
                    status=LeakageCheckStatus.FAILED,
                    reason=str(exc),
                )
            )
            issues.append(self._blocking_issue("LEAKAGE_RECORD_COVERAGE", str(exc)))

        # 3. RECORD_IDENTITY & 4. DUPLICATE_CONTENT
        row_by_record: dict[UUID, int] = {}
        unconfirmed = False
        partition_hashes: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
        for item in membership.memberships:
            row_by_record[item.record_id] = item.row_position
            ver = self._lineage.get_version(scope=scope, record_version_id=item.record_version_id)
            if not ver.identity.eligible_for_automatic_matching:
                unconfirmed = True
            if item.record_id in train_set:
                partition_hashes["train"].add(ver.content_sha256)
            elif item.record_id in val_set:
                partition_hashes["val"].add(ver.content_sha256)
            elif item.record_id in test_set:
                partition_hashes["test"].add(ver.content_sha256)

        if unconfirmed:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.RECORD_IDENTITY,
                    status=LeakageCheckStatus.FAILED,
                    reason="One or more records lack confirmed business key identity.",
                )
            )
            issues.append(
                self._blocking_issue("LEAKAGE_UNCONFIRMED_IDENTITY", "Unconfirmed record identity.")
            )
        else:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.RECORD_IDENTITY, status=LeakageCheckStatus.PASSED
                )
            )

        dup_cross = (
            (partition_hashes["train"] & partition_hashes["val"])
            | (partition_hashes["train"] & partition_hashes["test"])
            | (partition_hashes["val"] & partition_hashes["test"])
        )
        if dup_cross:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.DUPLICATE_CONTENT,
                    status=LeakageCheckStatus.FAILED,
                    reason="Identical record content spans multiple split partitions.",
                )
            )
            issues.append(
                self._blocking_issue(
                    "LEAKAGE_DUPLICATE_CONTENT", "Duplicate content across splits."
                )
            )
        else:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.DUPLICATE_CONTENT, status=LeakageCheckStatus.PASSED
                )
            )

        # 5. GROUP_OVERLAP
        if contract.grouping_columns:
            part_groups: dict[str, set[tuple[str, ...]]] = {
                "train": set(),
                "val": set(),
                "test": set(),
            }
            for pname, rids in (("train", train_set), ("val", val_set), ("test", test_set)):
                for rid in rids:
                    r_idx = row_by_record[rid]
                    gkey = tuple(str(df.iloc[r_idx][col]) for col in contract.grouping_columns)
                    part_groups[pname].add(gkey)
            group_overlap = (
                (part_groups["train"] & part_groups["val"])
                | (part_groups["train"] & part_groups["test"])
                | (part_groups["val"] & part_groups["test"])
            )
            if group_overlap:
                checks.append(
                    LeakageCheckResult(
                        check=LeakageCheck.GROUP_OVERLAP,
                        status=LeakageCheckStatus.FAILED,
                        reason="Grouping column keys overlap across partitions.",
                    )
                )
                issues.append(
                    self._blocking_issue(
                        "LEAKAGE_GROUP_OVERLAP", "Group overlap across partitions."
                    )
                )
            else:
                checks.append(
                    LeakageCheckResult(
                        check=LeakageCheck.GROUP_OVERLAP, status=LeakageCheckStatus.PASSED
                    )
                )
        else:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.GROUP_OVERLAP,
                    status=LeakageCheckStatus.NOT_APPLICABLE,
                    reason="DatasetContract defines no grouping columns.",
                )
            )

        # 6. TEMPORAL_ORDER
        if manifest.configuration.strategy in {
            SplitStrategy.TEMPORAL,
            SplitStrategy.GROUP_TEMPORAL,
        }:
            assert contract.time_column is not None
            ts = pd.to_datetime(df[contract.time_column], format="mixed", utc=True)
            train_max = max(ts.iloc[row_by_record[r]] for r in train_set)
            val_min = min(ts.iloc[row_by_record[r]] for r in val_set)
            val_max = max(ts.iloc[row_by_record[r]] for r in val_set)
            test_min = min(ts.iloc[row_by_record[r]] for r in test_set)
            if train_max > val_min or val_max > test_min:
                checks.append(
                    LeakageCheckResult(
                        check=LeakageCheck.TEMPORAL_ORDER,
                        status=LeakageCheckStatus.FAILED,
                        reason="Temporal look-ahead violation between partitions.",
                    )
                )
                issues.append(
                    self._blocking_issue("LEAKAGE_TEMPORAL_ORDER", "Temporal ordering violated.")
                )
            else:
                checks.append(
                    LeakageCheckResult(
                        check=LeakageCheck.TEMPORAL_ORDER, status=LeakageCheckStatus.PASSED
                    )
                )
        else:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.TEMPORAL_ORDER,
                    status=LeakageCheckStatus.NOT_APPLICABLE,
                    reason="Non-temporal split strategy selected.",
                )
            )

        # 7. HISTORICAL_USAGE
        all_ids = tuple(
            manifest.train.record_ids + manifest.validation.record_ids + manifest.test.record_ids
        )
        history = self._usage.get_history(scope=scope, record_ids=all_ids)
        exposed_test = find_previously_exposed_test_records(
            manifest=manifest, scope=scope, history=history
        )
        if exposed_test:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.HISTORICAL_USAGE,
                    status=LeakageCheckStatus.FAILED,
                    reason=f"{len(exposed_test)} test record(s) already exposed in prior ML runs.",
                )
            )
            issues.append(
                self._blocking_issue(
                    "LEAKAGE_HISTORICAL_TEST_REUSE",
                    "Protected test contains previously exposed records.",
                )
            )
        else:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.HISTORICAL_USAGE, status=LeakageCheckStatus.PASSED
                )
            )

        # 8. FEATURE_AVAILABILITY
        try:
            contract.validate_prediction_features()
            for pname in contract.predictor_names:
                feat = contract.get_feature(pname)
                if not contract.is_available_at_prediction(feat):
                    raise ValueError(f"Predictor '{pname}' unavailable at prediction time.")
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.FEATURE_AVAILABILITY, status=LeakageCheckStatus.PASSED
                )
            )
        except ValueError as exc:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.FEATURE_AVAILABILITY,
                    status=LeakageCheckStatus.FAILED,
                    reason=str(exc),
                )
            )
            issues.append(self._blocking_issue("LEAKAGE_FEATURE_AVAILABILITY", str(exc)))

        report = ValidationReport(
            dataset_id=context.dataset_id,
            contract_version=context.contract_version,
            dataset_revision_id=manifest.dataset_revision_id,
            domain=context.domain,
            process=context.process,
            purpose=context.purpose,
            completed_stages=(PipelineStage.SPLITTING,),
            issues=tuple(issues),
        )
        return LeakageValidationResult(
            readiness=readiness,
            manifest=manifest,
            checks=tuple(checks),
            report=report,
        )

    @staticmethod
    def _blocking_issue(code: str, message: str) -> ValidationIssue:
        return ValidationIssue(
            code=code,
            stage=PipelineStage.SPLITTING,
            severity=ValidationSeverity.CRITICAL,
            impact=IssueImpact.BLOCKING,
            message=message,
        )


class LocalSplitRegistry(SplitRegistry):
    """Local JSON persistence for immutable SplitManifest and TrainingAuthorization.

    Migration path: Replace with Shared Core P1.6 `LocalSplitRegistry`.
    """

    def __init__(self, registry_dir: str | Path) -> None:
        self._root = Path(registry_dir).resolve()

    def _split_path(self, factory_id: UUID, split_id: UUID) -> Path:
        return safe_resolve(self._root, "splits", str(factory_id), f"{split_id}.json")

    def _validation_path(self, factory_id: UUID, validation_id: UUID) -> Path:
        return safe_resolve(
            self._root, "leakage_validations", str(factory_id), f"{validation_id}.json"
        )

    def register_split(self, *, factory_id: UUID, manifest: SplitManifest) -> SplitManifest:
        if manifest.context.factory_id != factory_id:
            raise RegistryScopeError("Split manifest factory mismatch.")
        path = self._split_path(factory_id, manifest.split_id)
        if path.is_file():
            existing = load_json_model(path, SplitManifest)
            if existing == manifest:
                return existing
            raise RegistryConflictError("Conflicting split manifest for existing split_id.")
        atomic_write_text(path, dump_json_model(manifest))
        return manifest

    def get_split(self, *, factory_id: UUID, split_id: UUID) -> SplitManifest:
        path = self._split_path(factory_id, split_id)
        if not path.is_file():
            raise RegistryNotFoundError(f"Split '{split_id}' not found.")
        return load_json_model(path, SplitManifest)

    def register_leakage_validation(
        self, *, factory_id: UUID, validation: LeakageValidationResult
    ) -> None:
        if validation.manifest.context.factory_id != factory_id:
            raise RegistryScopeError("Leakage validation factory mismatch.")
        self.get_split(factory_id=factory_id, split_id=validation.manifest.split_id)
        path = self._validation_path(factory_id, validation.validation_id)
        if path.is_file():
            existing = load_json_model(path, LeakageValidationResult)
            if existing == validation:
                return
            raise RegistryConflictError("Conflicting leakage validation record.")
        atomic_write_text(path, dump_json_model(validation))

    def issue_training_authorization(
        self, *, factory_id: UUID, validation_id: UUID
    ) -> TrainingAuthorization:
        path = self._validation_path(factory_id, validation_id)
        if not path.is_file():
            raise RegistryNotFoundError(f"Leakage validation '{validation_id}' not found.")
        validation = load_json_model(path, LeakageValidationResult)
        if validation.manifest.context.factory_id != factory_id:
            raise RegistryScopeError("Validation factory mismatch.")
        return TrainingAuthorization.from_validation(validation)

    def verify_training_authorization(
        self, *, factory_id: UUID, authorization: TrainingAuthorization
    ) -> None:
        if authorization.development_split.context.factory_id != factory_id:
            raise RegistryScopeError("Authorization factory mismatch.")
        path = self._validation_path(factory_id, authorization.leakage_validation_id)
        if not path.is_file():
            raise RegistryNotFoundError("Authorization references unregistered leakage validation.")
        validation = load_json_model(path, LeakageValidationResult)
        validation.require_training_eligibility()
        if validation.manifest.split_id != authorization.development_split.split_id:
            raise RegistryConflictError("Authorization split mismatch.")


class LocalRecordUsageRegistry(RecordUsageRegistry):
    """Append-only local JSON persistence for historical ML record exposures.

    Migration path: Replace with Shared Core P1.6 `LocalRecordUsageRegistry`.
    """

    def __init__(self, registry_dir: str | Path) -> None:
        self._root = Path(registry_dir).resolve()

    def _scope_dir(self, scope: RecordScope) -> Path:
        return safe_resolve(
            self._root,
            "record_usage",
            str(scope.factory_id),
            scope.domain,
            scope.process,
            scope.dataset_id,
            scope.identity_namespace,
            scope.identity_version,
        )

    def record_usage(self, *, scope: RecordScope, usages: tuple[RecordUsage, ...]) -> None:
        if not usages:
            return
        scope_dir = self._scope_dir(scope)
        for usage in usages:
            if usage.scope != scope:
                raise RegistryScopeError("RecordUsage scope mismatch.")
            path = safe_resolve(scope_dir, f"{usage.usage_id}.json")
            if path.is_file():
                continue
            atomic_write_text(path, dump_json_model(usage))

    def get_history(
        self, *, scope: RecordScope, record_ids: tuple[UUID, ...]
    ) -> tuple[RecordUsage, ...]:
        wanted = set(record_ids)
        all_usages = iter_json_models(self._scope_dir(scope), RecordUsage)
        matched = [u for u in all_usages if u.record_id in wanted]
        matched.sort(key=lambda u: (u.exposed_at, str(u.usage_id)))
        return tuple(matched)


def hash_contract(contract: DatasetContract) -> str:
    """Compute canonical SHA-256 of a DatasetContract for reuse & bundle binding."""
    canonical = json.dumps(
        contract.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return calculate_bytes_sha256(canonical)
