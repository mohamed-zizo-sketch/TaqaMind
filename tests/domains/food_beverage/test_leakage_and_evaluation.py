"""Tests for training readiness, leakage prevention, and protected final evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from taqamind.core.interfaces import (
    AcceptancePolicy,
    EvaluationDecision,
    ModelStatus,
    ModelTransitionAction,
    ModelTransitionRequest,
    RecordUsageRole,
)
from taqamind.domains.food_beverage.pipeline import FoodBeveragePipeline
from tests.domains.food_beverage.conftest import (
    FACTORY_A,
    make_synthetic_bakery_dataframe,
)


def test_unconfirmed_target_blocks_training(
    tmp_pipeline: FoodBeveragePipeline,
    bakery_train_csv: Path,
) -> None:
    with pytest.raises(ValueError, match="not eligible for production splitting"):
        tmp_pipeline.train_model(
            csv_path=bakery_train_csv,
            factory_id=FACTORY_A,
            process_id="bakery",
            target_confirmation=None,  # UNCONFIRMED
        )


def test_disjoint_temporal_split_and_baseline_comparison(
    tmp_pipeline: FoodBeveragePipeline,
    bakery_train_csv: Path,
    synthetic_confirmation,
) -> None:
    res = tmp_pipeline.train_model(
        csv_path=bakery_train_csv,
        factory_id=FACTORY_A,
        process_id="bakery",
        target_confirmation=synthetic_confirmation,
    )
    m = res.split_manifest
    tr = set(m.train.record_ids)
    va = set(m.validation.record_ids)
    te = set(m.test.record_ids)
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)
    assert res.evaluation.decision == EvaluationDecision.ACCEPTED
    assert res.evaluation.test_improvement_pct > 1.0
    assert res.registry_entry is not None
    assert res.registry_entry.status == ModelStatus.ACTIVE


def test_rejected_model_cannot_be_activated(
    tmp_pipeline: FoodBeveragePipeline,
    bakery_train_csv: Path,
    synthetic_confirmation,
) -> None:
    # Impossible acceptance threshold forces REJECTED evaluation
    strict_policy = AcceptancePolicy(
        minimum_improvement_pct=99.99,
        justification="Strict policy test for rejection.",
    )
    res = tmp_pipeline.train_model(
        csv_path=bakery_train_csv,
        factory_id=FACTORY_A,
        process_id="bakery",
        target_confirmation=synthetic_confirmation,
        acceptance_policy=strict_policy,
    )
    assert res.evaluation.decision == EvaluationDecision.REJECTED
    assert res.registry_entry is not None
    assert res.registry_entry.status == ModelStatus.REJECTED

    with pytest.raises(ValueError):
        tmp_pipeline.model_registry.transition(
            request=ModelTransitionRequest(
                scope=res.registry_entry.version.scope,
                target_model_version_id=res.registry_entry.version.model_version_id,
                action=ModelTransitionAction.FIRST_ACTIVATION,
                requested_by="tester",
                reason="Should fail because rejected",
                comparison=None,
            )
        )


def test_historical_exposure_excluded_from_new_test_partition(
    tmp_pipeline: FoodBeveragePipeline,
    tmp_path: Path,
    synthetic_confirmation,
) -> None:
    df1 = make_synthetic_bakery_dataframe(n_rows=40, seed=10, include_target=True)
    p1 = tmp_path / "run1.csv"
    df1.to_csv(p1, index=False)

    res1 = tmp_pipeline.train_model(
        csv_path=p1,
        factory_id=FACTORY_A,
        process_id="bakery",
        target_confirmation=synthetic_confirmation,
    )
    test_rids = tuple(res1.split_manifest.test.record_ids)
    history = tmp_pipeline.usage_registry.get_history(
        scope=res1.prepared.lineage.manifest.scope,
        record_ids=test_rids,
    )
    exposed_ids = {u.record_id for u in history if u.role == RecordUsageRole.PROTECTED_TEST}
    assert exposed_ids == set(test_rids)

    # Add 20 new records and retrain -> exposed records must NOT be in new test partition
    df_new = make_synthetic_bakery_dataframe(
        n_rows=20, seed=20, include_target=True, batch_offset=100
    )
    df_combined = df1._append(df_new, ignore_index=True)
    p2 = tmp_path / "run2.csv"
    df_combined.to_csv(p2, index=False)

    from taqamind.core.interfaces import TrainingMode

    res2 = tmp_pipeline.train_model(
        csv_path=p2,
        factory_id=FACTORY_A,
        process_id="bakery",
        target_confirmation=synthetic_confirmation,
        intent=TrainingMode.RETRAINING,
    )
    assert set(res2.split_manifest.test.record_ids).isdisjoint(exposed_ids)
