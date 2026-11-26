"""CRITICAL ACCEPTANCE TEST: Prove zero retraining/fitting during repeated predictions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pandas as pd
from sklearn.base import BaseEstimator

from taqamind.core.contracts import DatasetPurpose
from taqamind.core.interfaces import AcceptancePolicy, ModelScope, PredictionStatus
from taqamind.core.preprocessing import TabularPreprocessor
from taqamind.domains.textile.config import build_default_training_configuration
from taqamind.domains.textile.pipeline import TextilePipelineService
from taqamind.domains.textile.weaving import (
    build_synthetic_weaving_confirmation,
    build_weaving_dataset_contract,
    generate_synthetic_weaving_prediction_frame,
    generate_synthetic_weaving_training_frame,
)


def test_repeated_predictions_never_retrain_or_fit(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    service = TextilePipelineService(store_dir)
    contract = build_weaving_dataset_contract(confirmation=build_synthetic_weaving_confirmation())
    factory_id = uuid4()

    train_df = generate_synthetic_weaving_training_frame(
        num_orders=28, batches_per_order=3, seed=42, zero_inflation_prob=0.25
    )
    train_csv = tmp_path / "train.csv"
    train_df.to_csv(train_csv, index=False)

    policy = AcceptancePolicy(minimum_improvement_pct=0.5, justification="No-retrain test.")
    prepared = service.ingest_and_prepare(
        factory_id=factory_id,
        process_id="weaving",
        csv_path=train_csv,
        purpose=DatasetPurpose.TRAINING,
        contract=contract,
    )
    train_res = service.train(
        factory_id=factory_id,
        process_id="weaving",
        prepared=prepared,
        contract=contract,
        training_config=build_default_training_configuration(include_heavy=True),
        policy=policy,
    )
    assert train_res.activated
    assert train_res.model_version is not None
    active_version_id = train_res.model_version.model_version_id
    active_sha256 = train_res.model_version.artifact.content_sha256

    # Prepare 3 separate prediction CSVs: new batch 1, mixed old+new, and new batch 2 after reload.
    pred_1 = generate_synthetic_weaving_prediction_frame(num_batches=4, seed=101, start_index=1)
    pred_2 = pd.concat(
        [
            pred_1.iloc[:2],
            generate_synthetic_weaving_prediction_frame(num_batches=3, seed=102, start_index=20),
        ],
        ignore_index=True,
    )
    pred_3 = generate_synthetic_weaving_prediction_frame(num_batches=5, seed=103, start_index=50)

    p1_csv = tmp_path / "pred1.csv"
    p2_csv = tmp_path / "pred2.csv"
    p3_csv = tmp_path / "pred3.csv"
    pred_1.to_csv(p1_csv, index=False)
    pred_2.to_csv(p2_csv, index=False)
    pred_3.to_csv(p3_csv, index=False)

    fit_calls: list[str] = []

    def _forbidden_fit(*args, **kwargs):  # type: ignore[no-untyped-def]
        fit_calls.append("called")
        raise AssertionError("Fitting/retraining is strictly forbidden during prediction!")

    with (
        patch.object(TabularPreprocessor, "fit", side_effect=_forbidden_fit),
        patch.object(TabularPreprocessor, "fit_transform", side_effect=_forbidden_fit),
        patch.object(BaseEstimator, "fit", side_effect=_forbidden_fit, create=True),
    ):
        b1 = service.predict(
            factory_id=factory_id, process_id="weaving", csv_path=p1_csv, contract=contract
        )
        b2 = service.predict(
            factory_id=factory_id, process_id="weaving", csv_path=p2_csv, contract=contract
        )
        # Simulate process restart / fresh service instance.
        reloaded_service = TextilePipelineService(store_dir)
        b3 = reloaded_service.predict(
            factory_id=factory_id, process_id="weaving", csv_path=p3_csv, contract=contract
        )

    assert fit_calls == []
    for batch in (b1, b2, b3):
        assert batch.model.model_version_id == active_version_id
        assert all(p.status == PredictionStatus.SUCCEEDED for p in batch.predictions)

    assert len({b1.context.execution_id, b2.context.execution_id, b3.context.execution_id}) == 3

    scope = ModelScope(
        factory_id=factory_id,
        domain="textile",
        process="weaving",
        task_id="weaving_waste_regression",
    )
    current_active = reloaded_service.stores.model_registry.get_active(scope=scope)
    assert current_active is not None
    assert current_active.version.model_version_id == active_version_id
    assert current_active.version.artifact.content_sha256 == active_sha256
