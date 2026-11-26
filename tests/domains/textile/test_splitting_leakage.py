"""Tests for training readiness, group/temporal splitting, leakage checks, and authorization."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from taqamind.core.contracts import DatasetPurpose
from taqamind.core.interfaces import (
    LeakageCheck,
    LeakageCheckStatus,
    SplitConfiguration,
    SplitStrategy,
)
from taqamind.domains.textile.knitting import (
    build_knitting_dataset_contract,
    build_synthetic_knitting_confirmation,
    generate_synthetic_knitting_training_frame,
)
from taqamind.domains.textile.pipeline import TextilePipelineService
from taqamind.domains.textile.weaving import (
    build_synthetic_weaving_confirmation,
    build_weaving_dataset_contract,
    generate_synthetic_weaving_training_frame,
)


def test_unconfirmed_target_blocks_readiness(tmp_path: Path) -> None:
    service = TextilePipelineService(tmp_path / "store")
    unconfirmed = build_weaving_dataset_contract()
    factory_id = uuid4()

    df = generate_synthetic_weaving_training_frame(num_orders=10, batches_per_order=2)
    csv_path = tmp_path / "weaving.csv"
    df.to_csv(csv_path, index=False)

    prep = service.ingest_and_prepare(
        factory_id=factory_id,
        process_id="weaving",
        csv_path=csv_path,
        purpose=DatasetPurpose.TRAINING,
        contract=unconfirmed,
    )
    readiness = service.stores.readiness_checker.assess(
        context=prep.context,
        artifact=prep.engineered_artifact,
        contract=unconfirmed,
    )
    with pytest.raises(ValueError):
        readiness.require_split_eligibility()


def test_weaving_group_split_disjoint_and_leakage_free(tmp_path: Path) -> None:
    service = TextilePipelineService(tmp_path / "store")
    contract = build_weaving_dataset_contract(confirmation=build_synthetic_weaving_confirmation())
    factory_id = uuid4()

    df = generate_synthetic_weaving_training_frame(num_orders=15, batches_per_order=3, seed=7)
    csv_path = tmp_path / "weaving.csv"
    df.to_csv(csv_path, index=False)

    prep = service.ingest_and_prepare(
        factory_id=factory_id,
        process_id="weaving",
        csv_path=csv_path,
        purpose=DatasetPurpose.TRAINING,
        contract=contract,
    )
    readiness = service.stores.readiness_checker.assess(
        context=prep.context,
        artifact=prep.engineered_artifact,
        contract=contract,
    )
    readiness.require_split_eligibility()

    manifest = service.stores.splitter.split(
        readiness=readiness,
        configuration=SplitConfiguration(
            strategy=SplitStrategy.GROUP,
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            random_seed=7,
            justification="Test group split.",
        ),
    )
    train_ids = set(manifest.train.record_ids)
    val_ids = set(manifest.validation.record_ids)
    test_ids = set(manifest.test.record_ids)
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    assert len(train_ids | val_ids | test_ids) == len(df)

    service.stores.split_registry.register_split(factory_id=factory_id, manifest=manifest)
    leakage = service.stores.leakage_validator.validate(readiness=readiness, manifest=manifest)
    leakage.require_training_eligibility()
    by_check = {c.check: c.status for c in leakage.checks}
    assert by_check[LeakageCheck.GROUP_OVERLAP] == LeakageCheckStatus.PASSED
    assert by_check[LeakageCheck.RECORD_OVERLAP] == LeakageCheckStatus.PASSED


def test_knitting_group_temporal_split_passes_temporal_order(tmp_path: Path) -> None:
    service = TextilePipelineService(tmp_path / "store")
    contract = build_knitting_dataset_contract(confirmation=build_synthetic_knitting_confirmation())
    factory_id = uuid4()

    df = generate_synthetic_knitting_training_frame(num_lots=18, runs_per_lot=2, seed=13)
    csv_path = tmp_path / "knitting.csv"
    df.to_csv(csv_path, index=False)

    prep = service.ingest_and_prepare(
        factory_id=factory_id,
        process_id="knitting",
        csv_path=csv_path,
        purpose=DatasetPurpose.TRAINING,
        contract=contract,
    )
    readiness = service.stores.readiness_checker.assess(
        context=prep.context,
        artifact=prep.engineered_artifact,
        contract=contract,
    )
    manifest = service.stores.splitter.split(
        readiness=readiness,
        configuration=SplitConfiguration(
            strategy=SplitStrategy.GROUP_TEMPORAL,
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            random_seed=13,
            justification="Test group-temporal split.",
        ),
    )
    leakage = service.stores.leakage_validator.validate(readiness=readiness, manifest=manifest)
    leakage.require_training_eligibility()
    by_check = {c.check: c.status for c in leakage.checks}
    assert by_check[LeakageCheck.TEMPORAL_ORDER] == LeakageCheckStatus.PASSED
    assert by_check[LeakageCheck.GROUP_OVERLAP] == LeakageCheckStatus.PASSED
