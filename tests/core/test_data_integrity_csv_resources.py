"""Resource limits across shared data integrity consumers."""

import pytest
from test_readiness_splitting_and_leakage import _prepare_engineered

from taqamind.core.ingestion.resource_policy import CsvResourcePolicy
from taqamind.core.ingestion.safe_csv import SafeCsvError
from taqamind.core.interfaces.lineage import RecordScope
from taqamind.core.interfaces.results import ExecutionStatus
from taqamind.core.interfaces.splitting import SplitConfiguration, SplitStrategy
from taqamind.core.lineage.membership_builder import RecordMembershipBuilder
from taqamind.core.readiness.readiness_checker import CoreTrainingReadinessChecker
from taqamind.core.splitting.dataset_splitter import CoreDatasetSplitter
from taqamind.core.splitting.leakage_validator import CoreLeakageValidator
from taqamind.core.splitting.split_registry import LocalRecordUsageRegistry


def test_integrity_consumers_reject_oversized_artifact_before_open(tmp_path, monkeypatch):
    storage, _, lineage, contract, context, artifact = _prepare_engineered(
        tmp_path, confirmed=True, n_rows=18
    )
    readiness = CoreTrainingReadinessChecker(storage, lineage).assess(
        context=context, artifact=artifact, contract=contract
    )
    readiness.require_split_eligibility()
    configuration = SplitConfiguration(
        strategy=SplitStrategy.GROUP_TEMPORAL,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        justification="Resource limit regression test.",
    )
    manifest = CoreDatasetSplitter(storage, lineage).split(
        readiness=readiness, configuration=configuration
    )
    usage = LocalRecordUsageRegistry(tmp_path / "reg")
    scope = RecordScope(
        factory_id=context.factory_id,
        domain=context.domain,
        process=context.process,
        dataset_id=context.dataset_id,
        identity_namespace="production_record",
    )
    policy = CsvResourcePolicy(max_bytes=artifact.reference.size_bytes - 1)

    def forbidden_open(*args, **kwargs):
        pytest.fail("Oversized artifact must not be opened.")

    monkeypatch.setattr(storage, "open_verified", forbidden_open)

    with pytest.raises(SafeCsvError) as membership_error:
        RecordMembershipBuilder(storage, lineage, resource_policy=policy).build_and_register(
            scope=scope, artifact=artifact, contract=contract
        )
    assert membership_error.value.code == "CSV_BYTE_LIMIT_EXCEEDED"

    rejected = CoreTrainingReadinessChecker(storage, lineage, resource_policy=policy).assess(
        context=context, artifact=artifact, contract=contract
    )
    assert rejected.result.status == ExecutionStatus.FAILED
    assert rejected.result.error is not None
    assert rejected.result.error.code == "CSV_BYTE_LIMIT_EXCEEDED"

    with pytest.raises(SafeCsvError) as split_error:
        CoreDatasetSplitter(storage, lineage, resource_policy=policy).split(
            readiness=readiness, configuration=configuration
        )
    assert split_error.value.code == "CSV_BYTE_LIMIT_EXCEEDED"

    with pytest.raises(SafeCsvError) as leakage_error:
        CoreLeakageValidator(storage, lineage, usage, resource_policy=policy).validate(
            readiness=readiness, manifest=manifest
        )
    assert leakage_error.value.code == "CSV_BYTE_LIMIT_EXCEEDED"
