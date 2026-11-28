"""Tests for model persistence, reload, no-retraining prediction, retraining, and CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from taqamind.core.interfaces import (
    ModelStatus,
    PredictionBatchStatus,
    PredictionStatus,
    TrainingMode,
)
from taqamind.core.interfaces.storage import ArtifactIntegrityError
from taqamind.core.preprocessing import TabularPreprocessor
from taqamind.domains.food_beverage import cli
from taqamind.domains.food_beverage.config import FoodBeverageStorageConfig
from taqamind.domains.food_beverage.pipeline import FoodBeveragePipeline
from tests.domains.food_beverage.conftest import (
    FACTORY_A,
    FACTORY_B,
    make_synthetic_bakery_dataframe,
)


def test_train_reload_fresh_process_and_no_retraining_prediction(
    tmp_path: Path,
    short_tmp_dir: Path,
    bakery_train_csv: Path,
    bakery_predict_csv: Path,
    synthetic_confirmation,
) -> None:
    storage_cfg = FoodBeverageStorageConfig(root_dir=short_tmp_dir / "s")
    train_pipe = FoodBeveragePipeline(storage_config=storage_cfg)
    train_res = train_pipe.train_model(
        csv_path=bakery_train_csv,
        factory_id=FACTORY_A,
        process_id="bakery",
        target_confirmation=synthetic_confirmation,
    )
    assert train_res.registry_entry is not None
    active_id = train_res.registry_entry.version.model_version_id
    active_sha = train_res.registry_entry.version.artifact.content_sha256

    # Simulate fresh process restart with a brand new FoodBeveragePipeline instance
    fresh_pipe = FoodBeveragePipeline(storage_config=storage_cfg)

    df_hist = pd.read_csv(bakery_train_csv).drop(columns=["waste_quantity_kg"]).head(5)
    df_new = pd.read_csv(bakery_predict_csv)
    mixed_csv = tmp_path / "mixed_predict.csv"
    pd.concat([df_hist, df_new], ignore_index=True).to_csv(mixed_csv, index=False)

    with (
        patch.object(
            fresh_pipe.trainer,
            "train",
            side_effect=AssertionError("Prediction must NEVER call ModelTrainer.train!"),
        ),
        patch.object(
            TabularPreprocessor,
            "fit",
            side_effect=AssertionError("Prediction must NEVER call TabularPreprocessor.fit!"),
        ),
        patch.object(
            TabularPreprocessor,
            "fit_transform",
            side_effect=AssertionError(
                "Prediction must NEVER call TabularPreprocessor.fit_transform!"
            ),
        ),
        patch.object(
            Pipeline,
            "fit",
            side_effect=AssertionError("Prediction must NEVER call Pipeline.fit!"),
        ),
        patch.object(
            Pipeline,
            "fit_transform",
            side_effect=AssertionError("Prediction must NEVER call Pipeline.fit_transform!"),
        ),
    ):
        pred1 = fresh_pipe.predict(
            csv_path=bakery_predict_csv,
            factory_id=FACTORY_A,
            process_id="bakery",
        )
        pred2 = fresh_pipe.predict(
            csv_path=bakery_predict_csv,
            factory_id=FACTORY_A,
            process_id="bakery",
        )
        pred3 = fresh_pipe.predict(
            csv_path=mixed_csv,
            factory_id=FACTORY_A,
            process_id="bakery",
        )

    assert pred1.status == PredictionBatchStatus.SUCCEEDED
    assert pred3.status == PredictionBatchStatus.SUCCEEDED
    assert len(pred3.predictions) == len(df_hist) + len(df_new)
    assert pred1.model.model_version_id == active_id
    assert pred1.context.execution_id != pred2.context.execution_id
    assert pred2.context.execution_id != pred3.context.execution_id
    assert all(
        p.status == PredictionStatus.SUCCEEDED and p.predicted_value >= 0.0
        for p in pred1.predictions
    )

    # Verify active model checksum and version remained untouched
    info = fresh_pipe.get_model_info(factory_id=FACTORY_A, process_id="bakery")
    assert info["active_model_version_id"] == str(active_id)
    loaded_entry = fresh_pipe.model_registry.get(
        scope=train_res.registry_entry.version.scope,
        model_version_id=active_id,
    )
    assert loaded_entry.version.artifact.content_sha256 == active_sha


def test_missing_active_model_and_wrong_factory_rejected(
    tmp_pipeline: FoodBeveragePipeline,
    bakery_train_csv: Path,
    bakery_predict_csv: Path,
    synthetic_confirmation,
) -> None:
    # No model trained yet
    with pytest.raises(ValueError, match="No ACTIVE model"):
        tmp_pipeline.predict(
            csv_path=bakery_predict_csv,
            factory_id=FACTORY_A,
            process_id="bakery",
        )

    # Train for FACTORY_A, try predicting for FACTORY_B
    tmp_pipeline.train_model(
        csv_path=bakery_train_csv,
        factory_id=FACTORY_A,
        process_id="bakery",
        target_confirmation=synthetic_confirmation,
    )
    with pytest.raises(ValueError, match="No ACTIVE model"):
        tmp_pipeline.predict(
            csv_path=bakery_predict_csv,
            factory_id=FACTORY_B,
            process_id="bakery",
        )


def test_corrupted_model_artifact_rejected(
    tmp_pipeline: FoodBeveragePipeline,
    bakery_train_csv: Path,
    bakery_predict_csv: Path,
    synthetic_confirmation,
) -> None:
    res = tmp_pipeline.train_model(
        csv_path=bakery_train_csv,
        factory_id=FACTORY_A,
        process_id="bakery",
        target_confirmation=synthetic_confirmation,
    )
    assert res.registry_entry is not None
    bundle_path = (
        tmp_pipeline.storage_config.root_dir
        / "model_storage"
        / res.registry_entry.version.artifact.storage_key
    )
    bundle_path.write_bytes(b"corrupted_bytes")

    with pytest.raises(ArtifactIntegrityError, match="Artifact (size|checksum)"):
        tmp_pipeline.predict(
            csv_path=bakery_predict_csv,
            factory_id=FACTORY_A,
            process_id="bakery",
        )


def test_feature_order_independence_and_retraining_reuse(
    tmp_pipeline: FoodBeveragePipeline,
    tmp_path: Path,
    bakery_train_csv: Path,
    bakery_predict_csv: Path,
    synthetic_confirmation,
) -> None:
    r1 = tmp_pipeline.train_model(
        csv_path=bakery_train_csv,
        factory_id=FACTORY_A,
        process_id="bakery",
        target_confirmation=synthetic_confirmation,
    )
    assert not r1.reused_existing
    assert r1.registry_entry is not None
    v1_id = r1.registry_entry.version.model_version_id

    # Shuffle columns in prediction CSV
    df_pred = pd.read_csv(bakery_predict_csv)
    shuffled_path = tmp_path / "shuffled_pred.csv"
    df_pred[list(reversed(df_pred.columns))].to_csv(shuffled_path, index=False)

    pred_orig = tmp_pipeline.predict(
        csv_path=bakery_predict_csv, factory_id=FACTORY_A, process_id="bakery"
    )
    pred_shuf = tmp_pipeline.predict(
        csv_path=shuffled_path, factory_id=FACTORY_A, process_id="bakery"
    )
    assert [p.predicted_value for p in pred_orig.predictions] == [
        p.predicted_value for p in pred_shuf.predictions
    ]

    # Explicit retrain on exact same dataset -> reuses verified result
    r_reuse = tmp_pipeline.train_model(
        csv_path=bakery_train_csv,
        factory_id=FACTORY_A,
        process_id="bakery",
        target_confirmation=synthetic_confirmation,
        intent=TrainingMode.RETRAINING,
    )
    assert r_reuse.reused_existing

    # Explicit retrain on expanded dataset -> creates candidate v2 while preserving v1 as ACTIVE
    df_extra = make_synthetic_bakery_dataframe(
        n_rows=30, seed=77, include_target=True, batch_offset=500
    )
    df_all = pd.read_csv(bakery_train_csv)._append(df_extra, ignore_index=True)
    p_all = tmp_path / "train_expanded.csv"
    df_all.to_csv(p_all, index=False)

    r2 = tmp_pipeline.train_model(
        csv_path=p_all,
        factory_id=FACTORY_A,
        process_id="bakery",
        target_confirmation=synthetic_confirmation,
        intent=TrainingMode.RETRAINING,
    )
    assert not r2.reused_existing
    assert r2.registry_entry is not None
    assert r2.registry_entry.version.version_number == 2
    assert r2.registry_entry.status == ModelStatus.VALIDATED
    # Original active model v1 remains ACTIVE until explicit promotion
    assert (
        tmp_pipeline.model_registry.get(
            scope=r1.registry_entry.version.scope, model_version_id=v1_id
        ).status
        == ModelStatus.ACTIVE
    )


def test_cli_end_to_end_commands(
    tmp_path: Path,
    short_tmp_dir: Path,
    bakery_train_csv: Path,
    bakery_predict_csv: Path,
) -> None:
    store = short_tmp_dir / "c"
    tpl_out = tmp_path / "tpl.csv"
    pred_out = tmp_path / "pred.json"

    assert (
        cli.main(
            [
                "--storage-dir",
                str(store),
                "template",
                "--process",
                "bakery",
                "--purpose",
                "prediction",
                "--output",
                str(tpl_out),
            ]
        )
        == 0
    )
    assert tpl_out.exists()

    assert (
        cli.main(
            [
                "--storage-dir",
                str(store),
                "inspect",
                "--csv",
                str(bakery_train_csv),
                "--factory-id",
                str(FACTORY_A),
                "--process",
                "bakery",
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "--storage-dir",
                str(store),
                "validate",
                "--csv",
                str(bakery_train_csv),
                "--factory-id",
                str(FACTORY_A),
                "--process",
                "bakery",
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "--storage-dir",
                str(store),
                "train",
                "--csv",
                str(bakery_train_csv),
                "--factory-id",
                str(FACTORY_A),
                "--process",
                "bakery",
                "--synthetic-confirm",
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "--storage-dir",
                str(store),
                "predict",
                "--csv",
                str(bakery_predict_csv),
                "--factory-id",
                str(FACTORY_A),
                "--process",
                "bakery",
                "--output",
                str(pred_out),
            ]
        )
        == 0
    )
    assert pred_out.exists()
    assert (
        cli.main(
            [
                "--storage-dir",
                str(store),
                "model-info",
                "--factory-id",
                str(FACTORY_A),
                "--process",
                "bakery",
            ]
        )
        == 0
    )
