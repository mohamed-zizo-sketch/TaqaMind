"""Tests for controlled AutoML training, protected test evaluation, and acceptance policy."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from taqamind.core.contracts import DatasetPurpose
from taqamind.core.interfaces import AcceptancePolicy, EvaluationDecision
from taqamind.domains.textile.config import build_default_training_configuration
from taqamind.domains.textile.pipeline import TextilePipelineService
from taqamind.domains.textile.weaving import (
    build_synthetic_weaving_confirmation,
    build_weaving_dataset_contract,
    generate_synthetic_weaving_training_frame,
)


def test_rejected_challenger_cannot_be_activated(tmp_path: Path) -> None:
    service = TextilePipelineService(tmp_path / "store")
    contract = build_weaving_dataset_contract(confirmation=build_synthetic_weaving_confirmation())
    factory_id = uuid4()

    df = generate_synthetic_weaving_training_frame(
        num_orders=28, batches_per_order=3, seed=42, zero_inflation_prob=0.25
    )
    csv_path = tmp_path / "weaving.csv"
    df.to_csv(csv_path, index=False)

    # Set an impossibly strict acceptance policy so the challenger is REJECTED.
    impossible_policy = AcceptancePolicy(
        minimum_improvement_pct=99.9,
        justification="Force rejection to verify rejected models cannot activate.",
    )
    prep = service.ingest_and_prepare(
        factory_id=factory_id,
        process_id="weaving",
        csv_path=csv_path,
        purpose=DatasetPurpose.TRAINING,
        contract=contract,
    )
    result = service.train(
        factory_id=factory_id,
        process_id="weaving",
        prepared=prep,
        contract=contract,
        training_config=build_default_training_configuration(include_heavy=False),
        policy=impossible_policy,
        activate_if_accepted=True,
    )
    assert result.evaluation.decision == EvaluationDecision.REJECTED
    assert not result.activated
    assert result.model_version is not None

    from taqamind.core.interfaces import ModelScope, RegistryConflictError

    scope = ModelScope(
        factory_id=factory_id,
        domain="textile",
        process="weaving",
        task_id="weaving_waste_regression",
    )
    assert service.stores.model_registry.get_active(scope=scope) is None

    # Protected test records were exposed once; re-evaluating the same split must conflict.
    from taqamind.domains.textile.adapters.data_access_training import (
        TextileProtectedFinalEvaluator,
    )

    evaluator = TextileProtectedFinalEvaluator(
        test_provider=service.stores.test_provider,
        evaluation_registry=service.stores.evaluation_registry,
        model_storage=service.stores.model_storage,
        contract=contract,
    )
    with pytest.raises(RegistryConflictError):
        evaluator.evaluate(selection=result.selection, manifest=result.split_manifest)
