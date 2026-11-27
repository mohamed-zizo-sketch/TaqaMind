"""Core P1.6 LeakageValidator implementation."""

from __future__ import annotations

from uuid import UUID

import pandas as pd

from taqamind.core.contracts.enums import (
    IssueImpact,
    PipelineStage,
    ValidationSeverity,
)
from taqamind.core.contracts.validation import ValidationIssue, ValidationReport
from taqamind.core.ingestion.artifact_csv import read_verified_csv
from taqamind.core.ingestion.resource_policy import (
    CsvResourcePolicy,
    resolve_internal_csv_policy,
)
from taqamind.core.ingestion.safe_csv import SafeCsvReader
from taqamind.core.interfaces.dataset_registry import RecordLineageRegistry
from taqamind.core.interfaces.leakage import (
    LeakageCheck,
    LeakageCheckResult,
    LeakageCheckStatus,
    LeakageValidationResult,
)
from taqamind.core.interfaces.lineage import RecordScope
from taqamind.core.interfaces.ml import LeakageValidator
from taqamind.core.interfaces.readiness import TrainingReadinessResult
from taqamind.core.interfaces.splitting import SplitManifest, SplitStrategy
from taqamind.core.interfaces.storage import DatasetArtifactStorage
from taqamind.core.interfaces.training_registry import (
    RecordUsageRegistry,
    find_previously_exposed_test_records,
)


class CoreLeakageValidator(LeakageValidator):
    """Evaluate all required data-leakage checks across split partitions and usage history."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        lineage_registry: RecordLineageRegistry,
        usage_registry: RecordUsageRegistry,
        *,
        delimiter: str = ",",
        identity_namespace: str = "production_record",
        resource_policy: CsvResourcePolicy | None = None,
    ) -> None:
        self._storage = storage
        self._lineage = lineage_registry
        self._usage = usage_registry
        self._resource_policy = resolve_internal_csv_policy(resource_policy)
        self._reader = SafeCsvReader(delimiter=delimiter, resource_policy=self._resource_policy)
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
        scope = RecordScope(
            factory_id=context.factory_id,
            domain=context.domain,
            process=context.process,
            dataset_id=context.dataset_id,
            identity_namespace=self._identity_namespace,
        )

        membership = self._lineage.get_membership(
            scope=scope,
            dataset_revision_id=manifest.dataset_revision_id,
        )
        membership.validate_artifact(artifact)
        df = read_verified_csv(
            storage=self._storage,
            factory_id=context.factory_id,
            reference=artifact.reference,
            reader=self._reader,
            policy=self._resource_policy,
        )

        checks: list[LeakageCheckResult] = []
        issues: list[ValidationIssue] = []

        # 1. RECORD_OVERLAP
        train_set = set(manifest.train.record_ids)
        val_set = set(manifest.validation.record_ids)
        test_set = set(manifest.test.record_ids)
        if (train_set & val_set) | (train_set & test_set) | (val_set & test_set):
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
            if (
                (part_groups["train"] & part_groups["val"])
                | (part_groups["train"] & part_groups["test"])
                | (part_groups["val"] & part_groups["test"])
            ):
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
