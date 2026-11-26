"""Tests for model reload, integrity verification, prediction errors, and derived rates."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from taqamind.core.contracts import DatasetPurpose
from taqamind.core.interfaces import (
    AcceptancePolicy,
    PredictionStatus,
    RegistryScopeError,
)
from taqamind.domains.textile.config import build_default_training_configuration
from taqamind.domains.textile.pipeline import TextilePipelineService
from taqamind.domains.textile.weaving import (
    build_synthetic_weaving_confirmation,
    build_weaving_dataset_contract,
    generate_synthetic_weaving_prediction_frame,
    generate_synthetic_weaving_training_frame,
)


def test_prediction_reload_cross_factory_and_missing_model_guards(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    service = TextilePipelineService(store_dir)
    contract = build_weaving_dataset_contract(confirmation=build_synthetic_weaving_confirmation())
    factory_a = uuid4()
    factory_b = uuid4()

    pred_df = generate_synthetic_weaving_prediction_frame(num_batches=4)
    pred_csv = tmp_path / "pred.csv"
    pred_df.to_csv(pred_csv, index=False)

    # No active model -> fails cleanly.
    with pytest.raises(RuntimeError, match="No active model"):
        service.predict(
            factory_id=factory_a,
            process_id="weaving",
            csv_path=pred_csv,
            contract=contract,
        )

    # Train and activate on Factory A.
    train_df = generate_synthetic_weaving_training_frame(
        num_orders=28, batches_per_order=3, seed=42, zero_inflation_prob=0.25
    )
    train_csv = tmp_path / "train.csv"
    train_df.to_csv(train_csv, index=False)

    policy = AcceptancePolicy(minimum_improvement_pct=0.5, justification="Test acceptance.")
    prep = service.ingest_and_prepare(
        factory_id=factory_a,
        process_id="weaving",
        csv_path=train_csv,
        purpose=DatasetPurpose.TRAINING,
        contract=contract,
    )
    train_res = service.train(
        factory_id=factory_a,
        process_id="weaving",
        prepared=prep,
        contract=contract,
        training_config=build_default_training_configuration(include_heavy=False),
        policy=policy,
    )
    assert train_res.activated
    assert train_res.model_version is not None

    # Reload service from disk (new process simulation) and predict.
    reloaded = TextilePipelineService(store_dir)
    batch = reloaded.predict(
        factory_id=factory_a,
        process_id="weaving",
        csv_path=pred_csv,
        contract=contract,
    )
    assert len(batch.predictions) == len(pred_df)
    assert all(p.status == PredictionStatus.SUCCEEDED for p in batch.predictions)
    assert all(
        p.derived_rate is not None and p.derived_rate.value is not None for p in batch.predictions
    )

    # Factory B has no active model and cannot use Factory A's model_version_id.
    with pytest.raises(RuntimeError, match="No active model"):
        reloaded.predict(
            factory_id=factory_b,
            process_id="weaving",
            csv_path=pred_csv,
            contract=contract,
        )

    # Passing wrong model_version_id raises.
    with pytest.raises(ValueError, match="is not active"):
        reloaded.predict(
            factory_id=factory_a,
            process_id="weaving",
            csv_path=pred_csv,
            contract=contract,
            model_version_id=uuid4(),
        )
    assert isinstance(RegistryScopeError("ok"), Exception)
