"""End-to-end synthetic CSV -> training -> evaluation -> activation -> prediction tests."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from taqamind.core.contracts import DatasetPurpose
from taqamind.core.interfaces import AcceptancePolicy, EvaluationDecision, PredictionStatus
from taqamind.domains.textile.config import build_default_training_configuration
from taqamind.domains.textile.knitting import (
    build_knitting_dataset_contract,
    build_synthetic_knitting_confirmation,
    generate_synthetic_knitting_prediction_frame,
    generate_synthetic_knitting_training_frame,
)
from taqamind.domains.textile.pipeline import TextilePipelineService
from taqamind.domains.textile.weaving import (
    build_synthetic_weaving_confirmation,
    build_weaving_dataset_contract,
    generate_synthetic_weaving_prediction_frame,
    generate_synthetic_weaving_training_frame,
)


@pytest.mark.parametrize("process_id", ["weaving", "knitting"])
def test_end_to_end_synthetic_training_and_prediction(tmp_path: Path, process_id: str) -> None:
    service = TextilePipelineService(tmp_path / f"store_{process_id}")
    factory_id = uuid4()

    if process_id == "weaving":
        contract = build_weaving_dataset_contract(
            confirmation=build_synthetic_weaving_confirmation()
        )
        train_df = generate_synthetic_weaving_training_frame(
            num_orders=28, batches_per_order=3, seed=42, zero_inflation_prob=0.25
        )
        pred_df = generate_synthetic_weaving_prediction_frame(num_batches=6, seed=99)
    else:
        contract = build_knitting_dataset_contract(
            confirmation=build_synthetic_knitting_confirmation()
        )
        train_df = generate_synthetic_knitting_training_frame(
            num_lots=36, runs_per_lot=3, seed=42, zero_inflation_prob=0.15
        )
        pred_df = generate_synthetic_knitting_prediction_frame(num_runs=6, seed=77)

    train_csv = tmp_path / f"{process_id}_train.csv"
    pred_csv = tmp_path / f"{process_id}_pred.csv"
    train_df.to_csv(train_csv, index=False)
    pred_df.to_csv(pred_csv, index=False)

    policy = AcceptancePolicy(minimum_improvement_pct=0.5, justification="E2E test.")
    prepared = service.ingest_and_prepare(
        factory_id=factory_id,
        process_id=process_id,
        csv_path=train_csv,
        purpose=DatasetPurpose.TRAINING,
        contract=contract,
    )
    train_res = service.train(
        factory_id=factory_id,
        process_id=process_id,
        prepared=prepared,
        contract=contract,
        training_config=build_default_training_configuration(include_heavy=True),
        policy=policy,
    )
    assert train_res.evaluation.decision == EvaluationDecision.ACCEPTED
    assert train_res.activated
    assert train_res.model_version is not None

    batch = service.predict(
        factory_id=factory_id,
        process_id=process_id,
        csv_path=pred_csv,
        contract=contract,
    )
    assert len(batch.predictions) == len(pred_df)
    assert all(p.status == PredictionStatus.SUCCEEDED for p in batch.predictions)
    assert all(
        p.predicted_value is not None and p.predicted_value >= 0.0 for p in batch.predictions
    )
