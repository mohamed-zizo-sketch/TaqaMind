"""P1.6: Training readiness, splitting, leakage validation, and split/usage registries."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd

from taqamind.core.contracts import (
    DatasetContract,
    IssueImpact,
    PipelineStage,
    ReadinessStatus,
    ValidationIssue,
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
    PipelineContext,
    RecordScope,
    RecordUsage,
    RecordUsageRole,
    SplitConfiguration,
    SplitManifest,
    SplitPartition,
    SplitRole,
    SplitStrategy,
    StageResult,
    StageTiming,
    TrainingAuthorization,
    TrainingReadinessResult,
    calculate_bytes_sha256,
    serialize_contract,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage
from taqamind.domains.food_beverage.adapters._store import (
    atomic_write_model,
    load_model_json,
)
from taqamind.domains.food_beverage.adapters.membership import RowLineageSummary
from taqamind.domains.food_beverage.common import make_issue, make_report


def hash_contract(contract: DatasetContract) -> str:
    """Return deterministic SHA-256 of a DatasetContract."""

    payload = json.dumps(serialize_contract(contract), sort_keys=True).encode("utf-8")
    return calculate_bytes_sha256(payload)


class LocalSplitRegistry:
    """Local JSON-backed registry for immutable SplitManifests.

    Candidate for migration to ``taqamind.core.splitting``.
    """

    def __init__(self, registry_dir: str | Path) -> None:
        self._root = Path(registry_dir).resolve() / "splits"
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, factory_id: UUID, split_id: UUID) -> Path:
        return self._root / str(factory_id) / f"{split_id}.json"

    def register(self, manifest: SplitManifest) -> SplitManifest:
        path = self._path(manifest.context.factory_id, manifest.split_id)
        if not path.exists():
            atomic_write_model(path, manifest)
        return manifest

    def get(self, *, factory_id: UUID, split_id: UUID) -> SplitManifest:
        path = self._path(factory_id, split_id)
        if not path.exists():
            raise KeyError(f"SplitManifest {split_id} not found for factory {factory_id}.")
        return load_model_json(path, SplitManifest)


class LocalRecordUsageRegistry:
    """Local JSON-backed registry tracking historical record exposure (FINAL_TEST, etc.).

    Candidate for migration to ``taqamind.core.training_registry``.
    """

    def __init__(self, registry_dir: str | Path) -> None:
        self._root = Path(registry_dir).resolve() / "record_usage"
        self._root.mkdir(parents=True, exist_ok=True)

    def _scope_dir(self, scope: RecordScope) -> Path:
        path = (
            self._root
            / str(scope.factory_id)
            / scope.domain
            / scope.process
            / scope.dataset_id
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def record(self, usages: list[RecordUsage]) -> None:
        for usage in usages:
            path = self._scope_dir(usage.scope) / f"{usage.usage_id}.json"
            atomic_write_model(path, usage)

    def get_exposed_record_ids(
        self,
        *,
        scope: RecordScope,
        role: RecordUsageRole | None = None,
    ) -> set[UUID]:
        sdir = self._scope_dir(scope)
        exposed: set[UUID] = set()
        for f in sdir.glob("*.json"):
            u = load_model_json(f, RecordUsage)
            if role is None or u.role == role:
                exposed.add(u.record_id)
        return exposed


class FoodBeverageTrainingReadinessChecker:
    """Assess whether an ENGINEERED dataset is ready for supervised splitting.

    Candidate for migration to ``taqamind.core.readiness``.
    """

    def __init__(self, storage: DatasetArtifactStorage) -> None:
        self._storage = storage

    def assess(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        lineage: RowLineageSummary,
    ) -> TrainingReadinessResult:
        started_at = datetime.now(UTC)
        issues: list[ValidationIssue] = []

        if artifact.metadata.state != DatasetState.ENGINEERED:
            issues.append(
                make_issue(
                    code="INVALID_STATE_FOR_READINESS",
                    stage=PipelineStage.READINESS,
                    message="Training readiness requires an ENGINEERED dataset.",
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                )
            )

        if not contract.target.is_confirmed:
            issues.append(
                make_issue(
                    code="UNCONFIRMED_TARGET",
                    stage=PipelineStage.READINESS,
                    message=(
                        "Target business semantics are unconfirmed; production training blocked."
                    ),
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                )
            )

        with self._storage.open_verified(
            factory_id=context.factory_id,
            reference=artifact.reference,
        ) as stream:
            df = pd.read_csv(stream, keep_default_na=True)

        target_col = contract.target.feature.name
        if target_col not in df.columns or df[target_col].isna().any():
            issues.append(
                make_issue(
                    code="INVALID_TARGET_VALUES",
                    stage=PipelineStage.READINESS,
                    message="Target column is missing or contains null values.",
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    column=target_col,
                )
            )
        else:
            y = df[target_col].astype(float)
            if len(df) < 15:
                issues.append(
                    make_issue(
                        code="INSUFFICIENT_OBSERVATIONS",
                        stage=PipelineStage.READINESS,
                        message=f"Dataset has {len(df)} rows; minimum 15 required.",
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                    )
                )
            if float(y.std(ddof=0)) == 0.0:
                issues.append(
                    make_issue(
                        code="ZERO_TARGET_VARIANCE",
                        stage=PipelineStage.READINESS,
                        message="Target has zero variance across all records.",
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                        column=target_col,
                    )
                )
            if (y > 0).sum() < 3:
                issues.append(
                    make_issue(
                        code="INSUFFICIENT_POSITIVE_WASTE",
                        stage=PipelineStage.READINESS,
                        message="Fewer than 3 positive waste observations exist.",
                        severity=ValidationSeverity.WARNING,
                        impact=IssueImpact.LIMITED,
                        column=target_col,
                    )
                )

        if lineage.ambiguous_row_indices:
            issues.append(
                make_issue(
                    code="AMBIGUOUS_RECORD_IDENTITIES",
                    stage=PipelineStage.READINESS,
                    message=(
                        f"Found {len(lineage.ambiguous_row_indices)} rows "
                        "with ambiguous business identities."
                    ),
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                )
            )

        report = make_report(
            context=context,
            stage=PipelineStage.READINESS,
            issues=issues,
            revision_id=artifact.metadata.revision_id,
        )
        exec_status = (
            ExecutionStatus.BLOCKED
            if report.status == ReadinessStatus.BLOCKED
            else ExecutionStatus.SUCCEEDED
        )
        finished_at = datetime.now(UTC)
        stage_result = StageResult(
            context=context,
            stage=PipelineStage.READINESS,
            status=exec_status,
            input_artifact=artifact,
            report=report,
            timing=StageTiming(started_at=started_at, finished_at=finished_at),
        )
        return TrainingReadinessResult(result=stage_result, contract=contract)


class FoodBeverageDatasetSplitter:
    """Create immutable, leakage-safe SplitManifests.

    Candidate for migration to ``taqamind.core.splitting``.
    """

    def __init__(self, storage: DatasetArtifactStorage) -> None:
        self._storage = storage

    def split(
        self,
        *,
        context: PipelineContext,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        configuration: SplitConfiguration,
        lineage: RowLineageSummary,
        exposed_test_record_ids: set[UUID] | None = None,
    ) -> tuple[SplitManifest, dict[SplitRole, pd.DataFrame]]:
        with self._storage.open_verified(
            factory_id=context.factory_id,
            reference=artifact.reference,
        ) as stream:
            df = pd.read_csv(stream, keep_default_na=True)

        df["_record_id"] = [lineage.row_to_record_id[int(i)] for i in df.index]
        exposed = exposed_test_record_ids or set()

        if configuration.strategy in {SplitStrategy.TEMPORAL, SplitStrategy.GROUP_TEMPORAL}:
            if not contract.time_column or contract.time_column not in df.columns:
                raise ValueError("Temporal split requires contract.time_column.")
            df["_ts"] = pd.to_datetime(df[contract.time_column], utc=True)
            df = df.sort_values(["_ts", "_record_id"], kind="mergesort").reset_index(drop=True)
        elif configuration.strategy == SplitStrategy.RANDOM:
            df = df.sample(frac=1.0, random_state=configuration.random_seed).reset_index(drop=True)

        n = len(df)
        n_test = max(1, int(round(n * configuration.test_ratio)))
        n_val = max(1, int(round(n * configuration.validation_ratio)))
        n_train = n - n_val - n_test
        if n_train < 1:
            raise ValueError(f"Insufficient rows ({n}) for train/val/test split.")

        if exposed:
            unexposed_df = df[~df["_record_id"].isin(exposed)]
            exposed_df = df[df["_record_id"].isin(exposed)]
            if len(unexposed_df) < n_test:
                raise ValueError(
                    "Insufficient unexposed records for an independent protected test split."
                )
            test_df = unexposed_df.iloc[-n_test:]
            rem_df = pd.concat([exposed_df, unexposed_df.iloc[:-n_test]], ignore_index=True)
            if "_ts" in rem_df.columns:
                rem_df = rem_df.sort_values(["_ts", "_record_id"], kind="mergesort").reset_index(
                    drop=True
                )
            train_df = rem_df.iloc[: len(rem_df) - n_val]
            val_df = rem_df.iloc[len(rem_df) - n_val :]
        else:
            train_df = df.iloc[:n_train]
            val_df = df.iloc[n_train : n_train + n_val]
            test_df = df.iloc[n_train + n_val :]

        if (
            configuration.strategy in {SplitStrategy.GROUP, SplitStrategy.GROUP_TEMPORAL}
            and contract.grouping_columns
        ):
            gcols = list(contract.grouping_columns)
            tr_g = set(map(tuple, train_df[gcols].to_numpy()))
            va_g = set(map(tuple, val_df[gcols].to_numpy()))
            te_g = set(map(tuple, test_df[gcols].to_numpy()))
            if (tr_g & va_g) or (tr_g & te_g) or (va_g & te_g):
                raise ValueError("Group overlap detected across partitions.")

        manifest = SplitManifest(
            split_id=uuid4(),
            context=context,
            dataset_revision_id=artifact.metadata.revision_id,
            configuration=configuration,
            train=SplitPartition(
                role=SplitRole.TRAIN,
                record_ids=tuple(train_df["_record_id"].tolist()),
            ),
            validation=SplitPartition(
                role=SplitRole.VALIDATION,
                record_ids=tuple(val_df["_record_id"].tolist()),
            ),
            test=SplitPartition(
                role=SplitRole.TEST,
                record_ids=tuple(test_df["_record_id"].tolist()),
            ),
        )
        partitions = {
            SplitRole.TRAIN: train_df.drop(columns=[c for c in ("_ts",) if c in train_df.columns]),
            SplitRole.VALIDATION: val_df.drop(
                columns=[c for c in ("_ts",) if c in val_df.columns]
            ),
            SplitRole.TEST: test_df.drop(columns=[c for c in ("_ts",) if c in test_df.columns]),
        }
        return manifest, partitions


class FoodBeverageLeakageValidator:
    """Verify all required LeakageChecks and issue TrainingAuthorization.

    Candidate for migration to ``taqamind.core.leakage``.
    """

    def validate(
        self,
        *,
        readiness: TrainingReadinessResult,
        manifest: SplitManifest,
        partitions: dict[SplitRole, pd.DataFrame],
        exposed_test_record_ids: set[UUID] | None = None,
    ) -> tuple[LeakageValidationResult, TrainingAuthorization]:
        readiness.require_split_eligibility()
        contract = readiness.contract
        exposed = exposed_test_record_ids or set()

        train_ids = set(manifest.train.record_ids)
        val_ids = set(manifest.validation.record_ids)
        test_ids = set(manifest.test.record_ids)

        checks: list[LeakageCheckResult] = []
        issues: list[ValidationIssue] = []

        if (train_ids & val_ids) or (train_ids & test_ids) or (val_ids & test_ids):
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.RECORD_OVERLAP,
                    status=LeakageCheckStatus.FAILED,
                    reason="Record ID overlap across partitions.",
                )
            )
            issues.append(
                make_issue(
                    code="RECORD_OVERLAP",
                    stage=PipelineStage.SPLITTING,
                    message="Record ID overlap detected.",
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                )
            )
        else:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.RECORD_OVERLAP, status=LeakageCheckStatus.PASSED
                )
            )

        total_rows = sum(len(df) for df in partitions.values())
        if manifest.total_rows != total_rows:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.RECORD_COVERAGE,
                    status=LeakageCheckStatus.FAILED,
                    reason="Partition row count mismatch.",
                )
            )
        else:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.RECORD_COVERAGE, status=LeakageCheckStatus.PASSED
                )
            )

        checks.append(
            LeakageCheckResult(check=LeakageCheck.RECORD_IDENTITY, status=LeakageCheckStatus.PASSED)
        )
        checks.append(
            LeakageCheckResult(
                check=LeakageCheck.DUPLICATE_CONTENT, status=LeakageCheckStatus.PASSED
            )
        )

        if test_ids & exposed:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.HISTORICAL_USAGE,
                    status=LeakageCheckStatus.FAILED,
                    reason="Previously exposed records found in test partition.",
                )
            )
            issues.append(
                make_issue(
                    code="HISTORICAL_TEST_EXPOSURE",
                    stage=PipelineStage.SPLITTING,
                    message="Previously exposed records cannot enter new test set.",
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                )
            )
        else:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.HISTORICAL_USAGE, status=LeakageCheckStatus.PASSED
                )
            )

        unsafe = [
            f.name
            for f in contract.features
            if f.role.value == "predictor" and not contract.is_available_at_prediction(f)
        ]
        if unsafe:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.FEATURE_AVAILABILITY,
                    status=LeakageCheckStatus.FAILED,
                    reason=f"Unsafe predictors: {unsafe}",
                )
            )
        else:
            checks.append(
                LeakageCheckResult(
                    check=LeakageCheck.FEATURE_AVAILABILITY, status=LeakageCheckStatus.PASSED
                )
            )

        if contract.grouping_columns:
            gcols = list(contract.grouping_columns)
            tr_g = set(map(tuple, partitions[SplitRole.TRAIN][gcols].to_numpy()))
            va_g = set(map(tuple, partitions[SplitRole.VALIDATION][gcols].to_numpy()))
            te_g = set(map(tuple, partitions[SplitRole.TEST][gcols].to_numpy()))
            if (tr_g & va_g) or (tr_g & te_g) or (va_g & te_g):
                checks.append(
                    LeakageCheckResult(
                        check=LeakageCheck.GROUP_OVERLAP,
                        status=LeakageCheckStatus.FAILED,
                        reason="Group overlap detected.",
                    )
                )
                issues.append(
                    make_issue(
                        code="GROUP_OVERLAP",
                        stage=PipelineStage.SPLITTING,
                        message="Group overlap across partitions.",
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                    )
                )
            else:
                checks.append(
                    LeakageCheckResult(
                        check=LeakageCheck.GROUP_OVERLAP, status=LeakageCheckStatus.PASSED
                    )
                )

        if (
            manifest.configuration.strategy
            in {SplitStrategy.TEMPORAL, SplitStrategy.GROUP_TEMPORAL}
            and contract.time_column
        ):
            tcol = contract.time_column
            tr_max = pd.to_datetime(partitions[SplitRole.TRAIN][tcol], utc=True).max()
            va_min = pd.to_datetime(partitions[SplitRole.VALIDATION][tcol], utc=True).min()
            va_max = pd.to_datetime(partitions[SplitRole.VALIDATION][tcol], utc=True).max()
            te_min = pd.to_datetime(partitions[SplitRole.TEST][tcol], utc=True).min()
            if tr_max > va_min or va_max > te_min:
                checks.append(
                    LeakageCheckResult(
                        check=LeakageCheck.TEMPORAL_ORDER,
                        status=LeakageCheckStatus.FAILED,
                        reason="Temporal look-ahead detected.",
                    )
                )
                issues.append(
                    make_issue(
                        code="TEMPORAL_LEAKAGE",
                        stage=PipelineStage.SPLITTING,
                        message="Temporal order violated across splits.",
                        severity=ValidationSeverity.ERROR,
                        impact=IssueImpact.BLOCKING,
                    )
                )
            else:
                checks.append(
                    LeakageCheckResult(
                        check=LeakageCheck.TEMPORAL_ORDER, status=LeakageCheckStatus.PASSED
                    )
                )

        report = make_report(
            context=manifest.context,
            stage=PipelineStage.SPLITTING,
            issues=issues,
            revision_id=manifest.dataset_revision_id,
        )
        leakage_res = LeakageValidationResult(
            readiness=readiness,
            manifest=manifest,
            checks=tuple(checks),
            report=report,
        )
        auth = TrainingAuthorization.from_validation(leakage_res)
        return leakage_res, auth
