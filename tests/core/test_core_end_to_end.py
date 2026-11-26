"""Core P1.9 end-to-end verification: factory isolation, retraining, and failure recovery."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from taqamind.core.contracts import DatasetPurpose
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.data_access import DataAccessDeniedError
from taqamind.core.interfaces.dataset_registry import RegistryNotFoundError
from taqamind.core.interfaces.model_registry import (
    ModelScope,
    ModelStatus,
    ModelTransitionAction,
    ModelTransitionRequest,
)
from taqamind.core.interfaces.prediction import PredictionStatus
from taqamind.core.interfaces.training_registry import TrainingIntent, TrainingMode
from taqamind.core.pipeline import CorePlatformServices
from tests.core.test_automl_and_evaluation import _build_contract
from tests.core.test_model_registry_and_prediction import (
    _ingest_through_engineered,
    _train_and_evaluate_model,
)


def test_p19_factory_isolation_and_unauthorized_cross_factory_access_blocked(
    tmp_path: Path,
) -> None:
    services = CorePlatformServices.create_local(tmp_path)
    contract = _build_contract()
    factory_a = uuid4()
    factory_b = uuid4()

    # Factory A trains and activates v1
    v1_a = _train_and_evaluate_model(services, factory_id=factory_a, contract=contract, seed=42)
    scope_a = v1_a.version.scope
    services.model_registry.transition(
        request=ModelTransitionRequest(
            scope=scope_a,
            action=ModelTransitionAction.FIRST_ACTIVATION,
            target_model_version_id=v1_a.version.model_version_id,
            expected_active_model_version_id=None,
            requested_by="admin-a",
            reason="Activate Factory A model.",
        )
    )

    # Factory B has NO active model in its own scope
    scope_b = ModelScope(
        factory_id=factory_b,
        domain=contract.domain,
        process=contract.process,
        task_id=contract.target.target_id,
    )
    assert services.model_registry.get_active(scope=scope_b) is None
    with pytest.raises(RegistryNotFoundError):
        services.model_registry.get(
            scope=scope_b,
            model_version_id=v1_a.version.model_version_id,
        )

    # Factory B cannot predict using Factory A's model identity
    pred_ctx_b = PipelineContext(
        factory_id=factory_b,
        dataset_revision_id=uuid4(),
        domain=contract.domain,
        process=contract.process,
        task_id=contract.target.target_id,
        dataset_id=contract.dataset_id,
        contract_version=contract.version,
        task_version="1.0.0",
        target_version="1.0.0",
        purpose=DatasetPurpose.PREDICTION,
        model_version_id=v1_a.version.model_version_id,
    )
    pred_csv = (
        b"run_id,lot_id,run_ts,planned_mass_kg,line_code\n"
        b"PB01,LOT_B1,2026-07-01T00:00:00+00:00,200.0,L1\n"
    )
    pred_eng_b = _ingest_through_engineered(
        services,
        ctx=pred_ctx_b,
        contract=contract,
        csv_payload=pred_csv,
    )
    with pytest.raises((DataAccessDeniedError, ValueError)):
        services.prediction_engine.predict(
            context=pred_ctx_b,
            artifact=pred_eng_b,
            contract=contract,
            model=v1_a.version.identity,
        )


def test_p19_retraining_intent_and_batch_prediction_e2e(tmp_path: Path) -> None:
    services = CorePlatformServices.create_local(tmp_path)
    contract = _build_contract()
    factory_id = uuid4()

    v1 = _train_and_evaluate_model(services, factory_id=factory_id, contract=contract, seed=42)
    scope = v1.version.scope
    services.model_registry.transition(
        request=ModelTransitionRequest(
            scope=scope,
            action=ModelTransitionAction.FIRST_ACTIVATION,
            target_model_version_id=v1.version.model_version_id,
            expected_active_model_version_id=None,
            requested_by="admin",
            reason="Initial activation.",
        )
    )

    # Explicit retraining intent must reference previous_model_version_id
    retrain_ctx = PipelineContext(
        factory_id=factory_id,
        dataset_revision_id=uuid4(),
        domain=contract.domain,
        process=contract.process,
        task_id=contract.target.target_id,
        dataset_id=contract.dataset_id,
        contract_version=contract.version,
        task_version="1.0.0",
        target_version="1.0.0",
        purpose=DatasetPurpose.TRAINING,
        created_at=datetime.now(UTC) - timedelta(minutes=5),
    )
    retrain_intent = TrainingIntent(
        context=retrain_ctx,
        mode=TrainingMode.RETRAINING,
        requested_by="ml-engineer",
        reason="Quarterly scheduled retraining.",
        previous_model_version_id=v1.version.model_version_id,
    )
    services.training_registry.register_intent(factory_id=factory_id, intent=retrain_intent)

    # Active model remains intact and serves predictions cleanly
    active = services.model_registry.get_active(scope=scope)
    assert active is not None and active.status == ModelStatus.ACTIVE

    pred_ctx = PipelineContext(
        factory_id=factory_id,
        dataset_revision_id=uuid4(),
        domain=contract.domain,
        process=contract.process,
        task_id=contract.target.target_id,
        dataset_id=contract.dataset_id,
        contract_version=contract.version,
        task_version="1.0.0",
        target_version="1.0.0",
        purpose=DatasetPurpose.PREDICTION,
        model_version_id=active.version.model_version_id,
    )
    pred_csv = (
        b"run_id,lot_id,run_ts,planned_mass_kg,line_code\n"
        b"PE01,LOT_E1,2026-08-01T00:00:00+00:00,220.0,L1\n"
        b"PE02,LOT_E1,2026-08-01T01:00:00+00:00,320.0,L2\n"
    )
    pred_eng = _ingest_through_engineered(
        services,
        ctx=pred_ctx,
        contract=contract,
        csv_payload=pred_csv,
    )
    batch = services.prediction_engine.predict(
        context=pred_ctx,
        artifact=pred_eng,
        contract=contract,
        model=active.version.identity,
    )
    assert batch.successful_count == 2
    assert all(p.status == PredictionStatus.SUCCEEDED for p in batch.predictions)
