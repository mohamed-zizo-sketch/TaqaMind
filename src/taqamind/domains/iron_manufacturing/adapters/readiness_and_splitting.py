"""Protocol-compatible training readiness, dataset splitting, and leakage validation.

Implements Shared Core protocols:
    - `TrainingReadinessChecker` -> `LocalTrainingReadinessChecker`
    - `DatasetSplitter` -> `LocalDatasetSplitter`
    - `LeakageValidator` -> `LocalLeakageValidator`
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import numpy as np
import pandas as pd

from taqamind.core.contracts import (
    DatasetContract,
    DatasetPurpose,
    IssueImpact,
    PipelineStage,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.dataset import DatasetArtifact, DatasetState
from taqamind.core.interfaces.dataset_registry import (
    DatasetMembershipManifest,
    RecordLineageRegistry,
    RegistryNotFoundError,
)
from taqamind.core.interfaces.leakage import (
    LeakageCheck,
    LeakageCheckResult,
    LeakageCheckStatus,
    LeakageValidationResult,
)
from taqamind.core.interfaces.lineage import RecordIdentityStatus, RecordScope
from taqamind.core.interfaces.processing import validate_processing_input
from taqamind.core.interfaces.readiness import TrainingReadinessResult
from taqamind.core.interfaces.results import (
    ExecutionStatus,
    StageResult,
    StageTiming,
)
from taqamind.core.interfaces.splitting import (
    SplitConfiguration,
    SplitManifest,
    SplitPartition,
    SplitRole,
    SplitStrategy,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage
from taqamind.core.interfaces.training_registry import (
    RecordUsageRegistry,
    find_previously_exposed_test_records,
)
from taqamind.domains.iron_manufacturing.adapters.processing import _read_artifact_df
from taqamind.domains.iron_manufacturing.config import (
    DEFAULT_MIN_POSITIVE_TARGET_ROWS,
    DEFAULT_MIN_TEST_ROWS,
    DEFAULT_MIN_TRAINING_ROWS,
    DEFAULT_MIN_VALIDATION_ROWS,
)


def build_record_scope(
    context: PipelineContext,
    *,
    identity_namespace: str = "business_key",
    identity_version: str = "1.0.0",
) -> RecordScope:
    """Create the deterministic RecordScope for a PipelineContext."""
    return RecordScope(
        factory_id=context.factory_id,
        domain=context.domain,
        process=context.process,
        dataset_id=context.dataset_id,
        identity_namespace=identity_namespace,
        identity_version=identity_version,
    )


class LocalTrainingReadinessChecker:
    """Assess whether an ENGINEERED dataset is eligible for splitting and training."""

    def __init__(
        self,
        *,
        storage: DatasetArtifactStorage,
        lineage_registry: RecordLineageRegistry,
        min_rows: int = DEFAULT_MIN_TRAINING_ROWS,
        min_positive_target_rows: int = DEFAULT_MIN_POSITIVE_TARGET_ROWS,
    ) -> None:
        self._storage = storage
        self._lineage_registry = lineage_registry
        self._min_rows = min_rows
        self._min_positive_target_rows = min_positive_target_rows

    def assess(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
    ) -> TrainingReadinessResult:
        started_at = datetime.now(UTC)
        if context.purpose != DatasetPurpose.TRAINING:
            raise ValueError("Training readiness assessment requires a TRAINING context.")
        validate_processing_input(
            context=context,
            artifact=artifact,
            contract=contract,
            expected_state=DatasetState.ENGINEERED,
        )

        df = _read_artifact_df(self._storage, context.factory_id, artifact)
        issues: list[ValidationIssue] = []

        # 1. Target semantics confirmation
        if not contract.target.is_confirmed:
            issues.append(
                ValidationIssue(
                    code="UNCONFIRMED_TARGET_SEMANTICS",
                    stage=PipelineStage.READINESS,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message=(
                        f"Target '{contract.target.feature.name}' is not confirmed; "
                        "production training is blocked until industrial confirmation is recorded."
                    ),
                    columns=(contract.target.feature.name,),
                    recommendation="Confirm target measurement semantics and unit boundaries.",
                )
            )

        # 2. Target presence, completeness, finiteness, and variation
        target_col = contract.target.feature.name
        if target_col not in df.columns:
            issues.append(
                ValidationIssue(
                    code="MISSING_TARGET_COLUMN",
                    stage=PipelineStage.READINESS,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message=f"Target column '{target_col}' is missing from engineered dataset.",
                    columns=(target_col,),
                )
            )
        else:
            y = pd.to_numeric(df[target_col], errors="coerce")
            if bool(y.isna().any()) or not bool(np.isfinite(y.to_numpy(dtype=float)).all()):
                issues.append(
                    ValidationIssue(
                        code="INVALID_TARGET_VALUES",
                        stage=PipelineStage.READINESS,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message="Target contains missing or non-finite values.",
                        columns=(target_col,),
                        affected_rows=int(y.isna().sum()),
                    )
                )
            else:
                if float(y.var(ddof=0)) <= 1e-12:
                    issues.append(
                        ValidationIssue(
                            code="ZERO_TARGET_VARIANCE",
                            stage=PipelineStage.READINESS,
                            severity=ValidationSeverity.ERROR,
                            impact=IssueImpact.BLOCKING,
                            message="Target exhibits zero variance; cannot train meaningful model.",
                            columns=(target_col,),
                        )
                    )
                pos_count = int((y > 0).sum())
                if pos_count < self._min_positive_target_rows:
                    issues.append(
                        ValidationIssue(
                            code="INSUFFICIENT_POSITIVE_TARGETS",
                            stage=PipelineStage.READINESS,
                            severity=ValidationSeverity.ERROR,
                            impact=IssueImpact.BLOCKING,
                            message=(
                                f"Only {pos_count} positive target observation(s) found; "
                                f"at least {self._min_positive_target_rows} required."
                            ),
                            columns=(target_col,),
                        )
                    )

        # 3. Required predictors present
        missing_predictors = sorted(set(contract.predictor_names) - set(df.columns))
        if missing_predictors:
            issues.append(
                ValidationIssue(
                    code="MISSING_ENGINEERED_PREDICTORS",
                    stage=PipelineStage.READINESS,
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    message=f"Missing declared predictors: {missing_predictors}.",
                    columns=tuple(missing_predictors),
                )
            )

        # 4. Registered membership & confirmed identities
        scope = build_record_scope(context)
        try:
            membership = self._lineage_registry.get_membership(
                scope=scope,
                dataset_revision_id=artifact.metadata.revision_id,
            )
            membership.validate_artifact(artifact)
            if membership.has_repeated_record_ids:
                issues.append(
                    ValidationIssue(
                        code="REPEATED_RECORD_IDENTITIES",
                        stage=PipelineStage.READINESS,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message="Repeated record identities exist in training dataset membership.",
                    )
                )
            unconfirmed_count = 0
            for item in membership.memberships:
                ident = self._lineage_registry.get_identity(scope=scope, record_id=item.record_id)
                if ident.status != RecordIdentityStatus.CONFIRMED:
                    unconfirmed_count += 1
            if unconfirmed_count > 0:
                issues.append(
                    ValidationIssue(
                        code="UNCONFIRMED_RECORD_IDENTITIES",
                        stage=PipelineStage.READINESS,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=(
                            f"{unconfirmed_count} row(s) have unverified or ambiguous record "
                            "identities; independent splitting cannot be guaranteed."
                        ),
                        affected_rows=unconfirmed_count,
                    )
                )
        except RegistryNotFoundError:
            issues.append(
                ValidationIssue(
                    code="MISSING_RECORD_MEMBERSHIP",
                    stage=PipelineStage.READINESS,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message=(
                        "Record lineage membership manifest is not registered for this "
                        "engineered revision."
                    ),
                )
            )

        # 5. Sufficient observations & group diversity for independent evaluation
        if len(df) < self._min_rows:
            issues.append(
                ValidationIssue(
                    code="INSUFFICIENT_OBSERVATIONS",
                    stage=PipelineStage.READINESS,
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    message=(
                        f"Dataset has {len(df)} rows; minimum {self._min_rows} required "
                        "for disjoint train/validation/test partitions."
                    ),
                )
            )

        if contract.grouping_columns:
            gcol = contract.grouping_columns[0]
            if gcol in df.columns and int(df[gcol].nunique(dropna=True)) < 3:
                issues.append(
                    ValidationIssue(
                        code="INSUFFICIENT_INDEPENDENT_GROUPS",
                        stage=PipelineStage.READINESS,
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        message=(
                            f"Grouping column '{gcol}' has fewer than 3 distinct groups; "
                            "disjoint group train/validation/test split is impossible."
                        ),
                        columns=(gcol,),
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
            ExecutionStatus.BLOCKED if report.has_blocking_issues else ExecutionStatus.SUCCEEDED
        )
        stage_result = StageResult(
            context=context,
            stage=PipelineStage.READINESS,
            status=status,
            input_artifact=artifact,
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=datetime.now(UTC)),
        )
        return TrainingReadinessResult(result=stage_result, contract=contract)


class LocalDatasetSplitter:
    """Leakage-aware dataset splitter implementing `DatasetSplitter`."""

    def __init__(
        self,
        *,
        storage: DatasetArtifactStorage,
        lineage_registry: RecordLineageRegistry,
        usage_registry: RecordUsageRegistry | None = None,
    ) -> None:
        self._storage = storage
        self._lineage_registry = lineage_registry
        self._usage_registry = usage_registry

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

        df = _read_artifact_df(self._storage, context.factory_id, artifact)
        scope = build_record_scope(context)
        membership = self._lineage_registry.get_membership(
            scope=scope,
            dataset_revision_id=artifact.metadata.revision_id,
        )
        membership.validate_artifact(artifact)
        if membership.has_repeated_record_ids:
            raise ValueError("Cannot split dataset with repeated record identities.")

        pos_to_record_id = {m.row_position: m.record_id for m in membership.memberships}
        work = df.copy()
        work["_record_id"] = [pos_to_record_id[i] for i in range(len(work))]

        time_col = contract.time_column
        if time_col is not None and time_col in work.columns:
            work["_parsed_ts"] = pd.to_datetime(work[time_col], errors="coerce")

        strategy = configuration.strategy
        if strategy == SplitStrategy.GROUP_TEMPORAL:
            if time_col is None or not contract.grouping_columns:
                raise ValueError("GROUP_TEMPORAL split requires time and grouping columns.")
            gcol = contract.grouping_columns[0]
            group_order = (
                work.groupby(gcol, sort=False)["_parsed_ts"]
                .min()
                .sort_values(kind="mergesort")
                .index.tolist()
            )
            tr_ids, va_ids, te_ids = self._partition_by_groups(
                work, gcol, group_order, configuration
            )
        elif strategy == SplitStrategy.GROUP:
            if not contract.grouping_columns:
                raise ValueError("GROUP split requires grouping columns.")
            gcol = contract.grouping_columns[0]
            rng = np.random.default_rng(configuration.random_seed)
            groups = sorted(work[gcol].astype(str).unique().tolist())
            rng.shuffle(groups)
            tr_ids, va_ids, te_ids = self._partition_by_groups(work, gcol, groups, configuration)
        elif strategy == SplitStrategy.TEMPORAL:
            if time_col is None:
                raise ValueError("TEMPORAL split requires a time column.")
            ordered = work.sort_values("_parsed_ts", kind="mergesort")
            tr_ids, va_ids, te_ids = self._partition_ordered_records(
                ordered["_record_id"].tolist(), configuration
            )
        else:  # SplitStrategy.RANDOM
            rng = np.random.default_rng(configuration.random_seed)
            rec_ids = work["_record_id"].tolist()
            perm = rng.permutation(len(rec_ids))
            shuffled = [rec_ids[int(idx)] for idx in perm]
            tr_ids, va_ids, te_ids = self._partition_ordered_records(shuffled, configuration)

        return SplitManifest(
            context=context,
            dataset_revision_id=artifact.metadata.revision_id,
            configuration=configuration,
            train=SplitPartition(role=SplitRole.TRAIN, record_ids=tuple(tr_ids)),
            validation=SplitPartition(role=SplitRole.VALIDATION, record_ids=tuple(va_ids)),
            test=SplitPartition(role=SplitRole.TEST, record_ids=tuple(te_ids)),
        )

    @staticmethod
    def _partition_ordered_records(
        ordered_ids: list[UUID],
        cfg: SplitConfiguration,
    ) -> tuple[list[UUID], list[UUID], list[UUID]]:
        n = len(ordered_ids)
        if n < 3:
            raise ValueError("At least 3 records required for train/validation/test splitting.")
        n_train = max(1, int(round(n * cfg.train_ratio)))
        n_val = max(1, int(round(n * cfg.validation_ratio)))
        if n_train + n_val >= n:
            n_val = max(1, min(n_val, n - 2))
            n_train = max(1, n - n_val - 1)
        tr = ordered_ids[:n_train]
        va = ordered_ids[n_train : n_train + n_val]
        te = ordered_ids[n_train + n_val :]
        if not te:
            te = [va.pop()]
        return tr, va, te

    @staticmethod
    def _partition_by_groups(
        work: pd.DataFrame,
        group_col: str,
        ordered_groups: list[object],
        cfg: SplitConfiguration,
    ) -> tuple[list[UUID], list[UUID], list[UUID]]:
        n_g = len(ordered_groups)
        if n_g < 3:
            raise ValueError("At least 3 distinct groups are required for group splitting.")
        i1 = max(1, int(round(n_g * cfg.train_ratio)))
        i2 = max(i1 + 1, int(round(n_g * (cfg.train_ratio + cfg.validation_ratio))))
        if i2 >= n_g:
            i2 = n_g - 1
            i1 = max(1, i2 - 1)
        tr_groups = set(ordered_groups[:i1])
        va_groups = set(ordered_groups[i1:i2])
        te_groups = set(ordered_groups[i2:])
        tr_ids = work.loc[work[group_col].isin(tr_groups), "_record_id"].tolist()
        va_ids = work.loc[work[group_col].isin(va_groups), "_record_id"].tolist()
        te_ids = work.loc[work[group_col].isin(te_groups), "_record_id"].tolist()
        if not tr_ids or not va_ids or not te_ids:
            raise ValueError("Group split produced an empty partition.")
        return tr_ids, va_ids, te_ids


class LocalLeakageValidator:
    """Comprehensive leakage validator implementing Shared Core `LeakageValidator`."""

    def __init__(
        self,
        *,
        storage: DatasetArtifactStorage,
        lineage_registry: RecordLineageRegistry,
        usage_registry: RecordUsageRegistry,
        min_validation_rows: int = DEFAULT_MIN_VALIDATION_ROWS,
        min_test_rows: int = DEFAULT_MIN_TEST_ROWS,
    ) -> None:
        self._storage = storage
        self._lineage_registry = lineage_registry
        self._usage_registry = usage_registry
        self._min_validation_rows = min_validation_rows
        self._min_test_rows = min_test_rows

    def validate(
        self,
        *,
        readiness: TrainingReadinessResult,
        manifest: SplitManifest,
    ) -> LeakageValidationResult:
        manifest.validate_readiness(readiness)
        context = manifest.context
        artifact = readiness.result.input_artifact
        contract = readiness.contract
        scope = build_record_scope(context)

        df = _read_artifact_df(self._storage, context.factory_id, artifact)
        membership: DatasetMembershipManifest = self._lineage_registry.get_membership(
            scope=scope,
            dataset_revision_id=artifact.metadata.revision_id,
        )
        membership.validate_artifact(artifact)

        checks: list[LeakageCheckResult] = []
        issues: list[ValidationIssue] = []

        # 1. RECORD_OVERLAP
        tr_set = set(manifest.train.record_ids)
        va_set = set(manifest.validation.record_ids)
        te_set = set(manifest.test.record_ids)
        overlap = (tr_set & va_set) | (tr_set & te_set) | (va_set & te_set)
        if overlap:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.RECORD_OVERLAP,
                    status=LeakageCheckStatus.FAILED,
                    reason=f"Detected {len(overlap)} overlapping record IDs across partitions.",
                )
            )
            issues.append(
                ValidationIssue(
                    code="LEAKAGE_RECORD_OVERLAP",
                    stage=PipelineStage.SPLITTING,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message="Record overlap detected between train/validation/test partitions.",
                )
            )
        else:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.RECORD_OVERLAP,
                    status=LeakageCheckStatus.PASSED,
                )
            )

        # 2. RECORD_COVERAGE
        expected_ids = tuple(m.record_id for m in membership.memberships)
        try:
            manifest.validate_record_coverage(expected_ids)
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.RECORD_COVERAGE,
                    status=LeakageCheckStatus.PASSED,
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
            issues.append(
                ValidationIssue(
                    code="LEAKAGE_RECORD_COVERAGE",
                    stage=PipelineStage.SPLITTING,
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    message=str(exc),
                )
            )

        # 3. RECORD_IDENTITY
        unconfirmed = [
            m.record_id
            for m in membership.memberships
            if self._lineage_registry.get_identity(scope=scope, record_id=m.record_id).status
            != RecordIdentityStatus.CONFIRMED
        ]
        if unconfirmed:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.RECORD_IDENTITY,
                    status=LeakageCheckStatus.FAILED,
                    reason=f"{len(unconfirmed)} records lack confirmed business identity.",
                )
            )
            issues.append(
                ValidationIssue(
                    code="LEAKAGE_UNCONFIRMED_IDENTITY",
                    stage=PipelineStage.SPLITTING,
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    message="All records in a split must have confirmed identities.",
                )
            )
        else:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.RECORD_IDENTITY,
                    status=LeakageCheckStatus.PASSED,
                )
            )

        # 4. DUPLICATE_CONTENT (distinct events with equal features are allowed when
        # business keys are distinct and confirmed; flag only conflicting content versions
        # sharing the same record_id or duplicate version_id across partitions)
        rec_to_ver = {m.record_id: m.record_version_id for m in membership.memberships}
        tr_vers = {rec_to_ver[rid] for rid in manifest.train.record_ids}
        va_vers = {rec_to_ver[rid] for rid in manifest.validation.record_ids}
        te_vers = {rec_to_ver[rid] for rid in manifest.test.record_ids}
        if (tr_vers & va_vers) | (tr_vers & te_vers) | (va_vers & te_vers):
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.DUPLICATE_CONTENT,
                    status=LeakageCheckStatus.FAILED,
                    reason="Identical record version IDs appear in multiple partitions.",
                )
            )
            issues.append(
                ValidationIssue(
                    code="LEAKAGE_DUPLICATE_VERSION",
                    stage=PipelineStage.SPLITTING,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message="Duplicate record version IDs found across partitions.",
                )
            )
        else:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.DUPLICATE_CONTENT,
                    status=LeakageCheckStatus.PASSED,
                )
            )

        # 5. GROUP_OVERLAP
        rec_to_row = {m.record_id: m.row_position for m in membership.memberships}
        if contract.grouping_columns:
            gcol = contract.grouping_columns[0]
            tr_g = {df.iloc[rec_to_row[r]][gcol] for r in manifest.train.record_ids}
            va_g = {df.iloc[rec_to_row[r]][gcol] for r in manifest.validation.record_ids}
            te_g = {df.iloc[rec_to_row[r]][gcol] for r in manifest.test.record_ids}
            g_overlap = (tr_g & va_g) | (tr_g & te_g) | (va_g & te_g)
            if g_overlap:
                checks.append(
                    LeakageCheckResult(
                        check=LeakageCheck.GROUP_OVERLAP,
                        status=LeakageCheckStatus.FAILED,
                        reason=f"Group overlap on '{gcol}': {len(g_overlap)} shared group(s).",
                    )
                )
                issues.append(
                    ValidationIssue(
                        code="LEAKAGE_GROUP_OVERLAP",
                        stage=PipelineStage.SPLITTING,
                        severity=ValidationSeverity.CRITICAL,
                        impact=IssueImpact.BLOCKING,
                        message=f"Grouping column '{gcol}' overlaps across split partitions.",
                        columns=(gcol,),
                    )
                )
            else:
                checks.append(
                    LeakageCheckResult(
                        check=LeakageCheck.GROUP_OVERLAP,
                        status=LeakageCheckStatus.PASSED,
                    )
                )

        # 6. TEMPORAL_ORDER
        if manifest.configuration.strategy in {
            SplitStrategy.TEMPORAL,
            SplitStrategy.GROUP_TEMPORAL,
        }:
            tcol = contract.time_column
            assert tcol is not None
            ts_series = pd.to_datetime(df[tcol], errors="coerce")
            tr_max = ts_series.iloc[[rec_to_row[r] for r in manifest.train.record_ids]].max()
            va_min = ts_series.iloc[[rec_to_row[r] for r in manifest.validation.record_ids]].min()
            va_max = ts_series.iloc[[rec_to_row[r] for r in manifest.validation.record_ids]].max()
            te_min = ts_series.iloc[[rec_to_row[r] for r in manifest.test.record_ids]].min()

            # For pure TEMPORAL, enforce strict non-lookahead: tr_max <= va_min and va_max <= te_min
            # For GROUP_TEMPORAL, enforce group-start monotonic order (group min ts non-decreasing)
            temporal_ok = True
            reason_msg = ""
            if manifest.configuration.strategy == SplitStrategy.TEMPORAL:
                if tr_max > va_min or va_max > te_min:
                    temporal_ok = False
                    reason_msg = (
                        f"Temporal look-ahead detected: train_max={tr_max}, "
                        f"val_min={va_min}, val_max={va_max}, test_min={te_min}."
                    )
            else:
                gcol = contract.grouping_columns[0]
                g_mins = df.assign(_ts=ts_series).groupby(gcol)["_ts"].min()
                tr_g_max = max(
                    g_mins[df.iloc[rec_to_row[r]][gcol]] for r in manifest.train.record_ids
                )
                va_g_min = min(
                    g_mins[df.iloc[rec_to_row[r]][gcol]] for r in manifest.validation.record_ids
                )
                va_g_max = max(
                    g_mins[df.iloc[rec_to_row[r]][gcol]] for r in manifest.validation.record_ids
                )
                te_g_min = min(
                    g_mins[df.iloc[rec_to_row[r]][gcol]] for r in manifest.test.record_ids
                )
                if tr_g_max > va_g_min or va_g_max > te_g_min:
                    temporal_ok = False
                    reason_msg = "Group-temporal start ordering violated across partitions."

            if not temporal_ok:
                checks.append(
                    LeakageCheckResult(
                        check=LeakageCheck.TEMPORAL_ORDER,
                        status=LeakageCheckStatus.FAILED,
                        reason=reason_msg,
                    )
                )
                issues.append(
                    ValidationIssue(
                        code="LEAKAGE_TEMPORAL_LOOKAHEAD",
                        stage=PipelineStage.SPLITTING,
                        severity=ValidationSeverity.CRITICAL,
                        impact=IssueImpact.BLOCKING,
                        message=reason_msg,
                        columns=(tcol,),
                    )
                )
            else:
                checks.append(
                    LeakageCheckResult(
                        check=LeakageCheck.TEMPORAL_ORDER,
                        status=LeakageCheckStatus.PASSED,
                    )
                )

        # 7. HISTORICAL_USAGE (V1 conservative policy: no previously exposed record
        # — including corrected versions sharing record_id — may enter a new protected test split)
        history = self._usage_registry.get_history(
            scope=scope,
            record_ids=manifest.test.record_ids,
        )
        exposed_test = find_previously_exposed_test_records(
            manifest=manifest,
            scope=scope,
            history=history,
        )
        if exposed_test:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.HISTORICAL_USAGE,
                    status=LeakageCheckStatus.FAILED,
                    reason=(
                        f"{len(exposed_test)} protected test record(s) were previously "
                        "exposed in prior training/validation/evaluation workflows."
                    ),
                )
            )
            issues.append(
                ValidationIssue(
                    code="LEAKAGE_HISTORICAL_TEST_EXPOSURE",
                    stage=PipelineStage.SPLITTING,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message=(
                        "Protected test partition contains historically exposed record IDs "
                        "(including corrected versions of prior records)."
                    ),
                    affected_rows=len(exposed_test),
                )
            )
        else:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.HISTORICAL_USAGE,
                    status=LeakageCheckStatus.PASSED,
                )
            )

        # 8. FEATURE_AVAILABILITY
        unavailable_predictors = [
            name
            for name in contract.predictor_names
            if not contract.is_available_at_prediction(contract.get_feature(name))
        ]
        if unavailable_predictors:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.FEATURE_AVAILABILITY,
                    status=LeakageCheckStatus.FAILED,
                    reason=f"Unavailable predictors at prediction time: {unavailable_predictors}.",
                )
            )
            issues.append(
                ValidationIssue(
                    code="LEAKAGE_FEATURE_UNAVAILABLE",
                    stage=PipelineStage.SPLITTING,
                    severity=ValidationSeverity.CRITICAL,
                    impact=IssueImpact.BLOCKING,
                    message=(
                        f"Predictors unavailable at prediction stage: {unavailable_predictors}."
                    ),
                    columns=tuple(unavailable_predictors),
                )
            )
        else:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.FEATURE_AVAILABILITY,
                    status=LeakageCheckStatus.PASSED,
                )
            )

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
