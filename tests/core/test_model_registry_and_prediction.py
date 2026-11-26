"""Tests for Core P1.8 ModelRegistry, PredictionEngine, and zero-retraining invariant."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from taqamind.core.automl.trainer import ControlledAutoMLTrainer, TabularPreprocessor
from taqamind.core.contracts import DatasetPurpose
from taqamind.core.interfaces.artifacts import (
    ArtifactFormat,
    ArtifactReference,
    calculate_bytes_sha256,
)
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.dataset import DatasetArtifact, DatasetMetadata, DatasetState
from taqamind.core.interfaces.evaluation import AcceptancePolicy, FrozenModelSelection
from taqamind.core.interfaces.lineage import RecordScope
from taqamind.core.interfaces.model_registry import (
    ModelFeatureSpecification,
    ModelScope,
    ModelStatus,
    ModelTransitionAction,
    ModelTransitionRequest,
    ModelVersion,
    PromotionComparison,
)
from taqamind.core.interfaces.prediction import PredictionModelIdentity, PredictionStatus
from taqamind.core.interfaces.processing_models import ApprovedMapping
from taqamind.core.interfaces.splitting import SplitConfiguration, SplitStrategy
from taqamind.core.interfaces.training import (
    CandidateConfiguration,
    CandidateRole,
    ModelFamily,
    RegressionMetric,
    TrainingConfiguration,
)
from taqamind.core.pipeline import CorePlatformServices
from tests.core.test_automl_and_evaluation import _build_contract


def _ingest_through_engineered(
    services: CorePlatformServices,
    *,
    ctx: PipelineContext,
    contract,
    csv_payload: bytes,
):
    ref = ArtifactReference(
        factory_id=ctx.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=calculate_bytes_sha256(csv_payload),
        size_bytes=len(csv_payload),
    )
    services.dataset_storage.put(
        factory_id=ctx.factory_id,
        reference=ref,
        stream=BytesIO(csv_payload),
    )
    raw = DatasetArtifact(
        metadata=DatasetMetadata(
            revision_id=ctx.dataset_revision_id,
            root_revision_id=ctx.dataset_revision_id,
            factory_id=ctx.factory_id,
            domain=ctx.domain,
            process=ctx.process,
            dataset_id=ctx.dataset_id,
            contract_version=ctx.contract_version,
            task_id=ctx.task_id,
            task_version=ctx.task_version,
            target_version=ctx.target_version,
            source_file_sha256=ref.content_sha256,
            state=DatasetState.RAW,
        ),
        reference=ref,
    )
    services.dataset_registry.register(factory_id=ctx.factory_id, artifact=raw)
    ing = services.ingestion.ingest(context=ctx, artifact=raw, contract=contract).output_artifact
    assert ing is not None
    services.dataset_registry.register(factory_id=ctx.factory_id, artifact=ing)

    prop = services.mapper.propose_mapping(context=ctx, artifact=ing, contract=contract)
    entries = tuple(
        e.model_copy(update={"source_unit": e.canonical_unit})
        if e.canonical_unit is not None and e.source_unit is None
        else e
        for e in prop.entries
    )
    mapped = services.mapper.apply_mapping(
        context=ctx,
        artifact=ing,
        contract=contract,
        mapping=ApprovedMapping(
            proposal=prop.model_copy(update={"entries": entries}),
            approved_by="qa",
            confirmed_source_columns=tuple(e.source_column for e in entries),
        ),
    ).output_artifact
    assert mapped is not None
    services.dataset_registry.register(factory_id=ctx.factory_id, artifact=mapped)

    cleaned = services.cleaner.clean(
        context=ctx,
        artifact=mapped,
        contract=contract,
    ).output_artifact
    assert cleaned is not None
    services.dataset_registry.register(factory_id=ctx.factory_id, artifact=cleaned)

    engineered = services.feature_engineer.engineer(
        context=ctx,
        artifact=cleaned,
        contract=contract,
    ).output_artifact
    assert engineered is not None
    services.dataset_registry.register(factory_id=ctx.factory_id, artifact=engineered)

    scope = RecordScope(
        factory_id=ctx.factory_id,
        domain=ctx.domain,
        process=ctx.process,
        dataset_id=ctx.dataset_id,
        identity_namespace="production_record",
    )
    services.membership_builder.build_and_register(
        scope=scope,
        artifact=engineered,
        contract=contract,
    )
    return engineered


def _train_and_evaluate_model(
    services: CorePlatformServices,
    *,
    factory_id,
    contract,
    seed: int = 42,
    previous_model_version_id=None,
):
    t0 = datetime.now(UTC) - timedelta(hours=1)
    rev_id = uuid4()
    ctx = PipelineContext(
        factory_id=factory_id,
        dataset_revision_id=rev_id,
        domain=contract.domain,
        process=contract.process,
        task_id=contract.target.target_id,
        dataset_id=contract.dataset_id,
        contract_version=contract.version,
        task_version="1.0.0",
        target_version="1.0.0",
        purpose=DatasetPurpose.TRAINING,
        created_at=t0 + timedelta(minutes=10),
    )

    lines = ["run_id,lot_id,run_ts,planned_mass_kg,line_code,scrap_kg"]
    base_ts = datetime(2026, 1, 1, tzinfo=UTC)
    offset = 0 if previous_model_version_id is None else 100
    for idx in range(36):
        lot = f"LOT_{offset + idx // 2:03d}"
        ts = (base_ts + timedelta(hours=offset + idx)).isoformat()
        mass = 100.0 + (idx % 7) * 50.0
        line = "L1" if idx % 2 == 0 else "L2"
        scrap = round(0.05 * mass + (12.0 if line == "L2" else 2.0), 3)
        lines.append(f"R{offset + idx:03d},{lot},{ts},{mass},{line},{scrap}")
    payload = ("\n".join(lines) + "\n").encode("utf-8")

    engineered = _ingest_through_engineered(
        services,
        ctx=ctx,
        contract=contract,
        csv_payload=payload,
    )
    readiness = services.readiness_checker.assess(
        context=ctx,
        artifact=engineered,
        contract=contract,
    )
    readiness.require_split_eligibility()

    manifest = services.splitter.split(
        readiness=readiness,
        configuration=SplitConfiguration(
            strategy=SplitStrategy.GROUP_TEMPORAL,
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            random_seed=seed,
            justification="Chronological split.",
        ),
    )
    services.split_registry.register_split(factory_id=factory_id, manifest=manifest)
    leakage = services.leakage_validator.validate(readiness=readiness, manifest=manifest)
    leakage.require_training_eligibility()
    services.split_registry.register_leakage_validation(factory_id=factory_id, validation=leakage)
    auth = services.split_registry.issue_training_authorization(
        factory_id=factory_id,
        validation_id=leakage.validation_id,
    )

    policy = AcceptancePolicy(
        minimum_improvement_pct=1.0,
        maximum_test_mae=25.0,
        justification="Standard acceptance policy.",
        created_at=ctx.created_at - timedelta(minutes=5),
    )
    config = TrainingConfiguration(
        random_seed=seed,
        candidates=(
            CandidateConfiguration(family=ModelFamily.DUMMY_REGRESSOR, role=CandidateRole.BASELINE),
            CandidateConfiguration(family=ModelFamily.RIDGE, role=CandidateRole.CHALLENGER),
            CandidateConfiguration(
                family=ModelFamily.HIST_GRADIENT_BOOSTING,
                role=CandidateRole.CHALLENGER,
            ),
        ),
        justification="Core P1.8 candidate training.",
    )
    run_result = services.trainer.train(
        authorization=auth,
        contract=contract,
        configuration=config,
        policy=policy,
    )
    selection = FrozenModelSelection.from_training_run(training_run=run_result, policy=policy)
    evaluator = services.create_evaluator(contract)
    eval_result = evaluator.evaluate(selection=selection, manifest=manifest)

    challenger_res = run_result.best_challenger
    assert challenger_res is not None and challenger_res.model_reference is not None
    challenger_cfg = config.get_candidate(challenger_res.candidate_id)

    identity = PredictionModelIdentity(
        model_version_id=uuid4(),
        factory_id=factory_id,
        domain=ctx.domain,
        process=ctx.process,
        task_id=ctx.task_id,
        dataset_id=ctx.dataset_id,
        contract_version=ctx.contract_version,
        task_version=ctx.task_version,
        target_version=ctx.target_version,
    )
    contract_sha = calculate_bytes_sha256(contract.model_dump_json().encode("utf-8"))
    version = ModelVersion(
        identity=identity,
        version_number=1 if previous_model_version_id is None else 2,
        training_run_id=run_result.training_run_id,
        selection_id=selection.selection_id,
        split_id=manifest.split_id,
        training_dataset_revision_id=engineered.metadata.revision_id,
        contract_sha256=contract_sha,
        implementation_sha256=calculate_bytes_sha256(b"core-automl-v1"),
        candidate=challenger_cfg,
        artifact=challenger_res.model_reference,
        feature_specification=ModelFeatureSpecification(
            feature_names=tuple(contract.predictor_names),
            target_name=contract.target.feature.name,
            target_unit=contract.target.feature.unit,
        ),
        previous_model_version_id=previous_model_version_id,
        created_at=max(selection.selected_at, datetime.now(UTC)),
    )
    scope = ModelScope.from_identity(identity)
    services.model_registry.register_candidate(scope=scope, version=version)
    validated = services.model_registry.apply_evaluation(
        scope=scope,
        model_version_id=version.model_version_id,
        evaluation=eval_result,
    )
    return validated


def test_p18_model_lifecycle_promotion_rollback_and_zero_retraining_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = CorePlatformServices.create_local(tmp_path)
    contract = _build_contract()
    factory_id = uuid4()

    v1 = _train_and_evaluate_model(services, factory_id=factory_id, contract=contract, seed=42)
    scope = v1.version.scope

    # First activation
    active_v1 = services.model_registry.transition(
        request=ModelTransitionRequest(
            scope=scope,
            action=ModelTransitionAction.FIRST_ACTIVATION,
            target_model_version_id=v1.version.model_version_id,
            expected_active_model_version_id=None,
            requested_by="plant-admin",
            reason="Initial production model activation.",
        )
    )
    assert active_v1.status == ModelStatus.ACTIVE

    # Train v2, promote, then rollback to v1
    v2 = _train_and_evaluate_model(
        services,
        factory_id=factory_id,
        contract=contract,
        seed=43,
        previous_model_version_id=v1.version.model_version_id,
    )
    promoted_v2 = services.model_registry.transition(
        request=ModelTransitionRequest(
            scope=scope,
            action=ModelTransitionAction.PROMOTION,
            target_model_version_id=v2.version.model_version_id,
            expected_active_model_version_id=v1.version.model_version_id,
            comparison=PromotionComparison(
                scope=scope,
                active_model_version_id=v1.version.model_version_id,
                challenger_model_version_id=v2.version.model_version_id,
                evaluation_dataset_revision_id=uuid4(),
                evaluation_record_ids=(uuid4(), uuid4()),
                metric=RegressionMetric.MAE,
                active_mae=2.0,
                challenger_mae=1.5,
                minimum_improvement_pct=1.0,
            ),
            requested_by="plant-admin",
            reason="Promote v2.",
        )
    )
    assert promoted_v2.status == ModelStatus.ACTIVE
    assert services.model_registry.get(
        scope=scope,
        model_version_id=v1.version.model_version_id,
    ).status == ModelStatus.ARCHIVED

    rolled_back_v1 = services.model_registry.transition(
        request=ModelTransitionRequest(
            scope=scope,
            action=ModelTransitionAction.ROLLBACK,
            target_model_version_id=v1.version.model_version_id,
            expected_active_model_version_id=v2.version.model_version_id,
            requested_by="plant-admin",
            reason="Rollback to v1.",
        )
    )
    assert rolled_back_v1.status == ModelStatus.ACTIVE
    assert services.model_registry.get(
        scope=scope,
        model_version_id=v2.version.model_version_id,
    ).status == ModelStatus.ARCHIVED

    # Prepare 3 unseen prediction batches + zero-retraining spy!
    def _forbid_fit(*args, **kwargs):
        raise AssertionError("Training/fitting must NEVER run during prediction!")

    monkeypatch.setattr(TabularPreprocessor, "fit", _forbid_fit)
    monkeypatch.setattr(ControlledAutoMLTrainer, "train", _forbid_fit)
    monkeypatch.setattr(Ridge, "fit", _forbid_fit)
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", _forbid_fit)

    # Fresh service instance from disk to prove persisted state reloads cleanly
    reloaded = CorePlatformServices.create_local(tmp_path)
    for batch_idx in range(3):
        pred_rev_id = uuid4()
        pred_ctx = PipelineContext(
            factory_id=factory_id,
            dataset_revision_id=pred_rev_id,
            domain=contract.domain,
            process=contract.process,
            task_id=contract.target.target_id,
            dataset_id=contract.dataset_id,
            contract_version=contract.version,
            task_version="1.0.0",
            target_version="1.0.0",
            purpose=DatasetPurpose.PREDICTION,
            model_version_id=v1.version.model_version_id,
        )
        # Note: prediction CSV has NO target column (scrap_kg)!
        lines = [
            "run_id,lot_id,run_ts,planned_mass_kg,line_code",
            f"P{batch_idx}01,PLOT_{batch_idx},2026-06-01T00:00:00+00:00,250.0,L1",
            f"P{batch_idx}02,PLOT_{batch_idx},2026-06-01T01:00:00+00:00,300.0,L2",
        ]
        pred_eng = _ingest_through_engineered(
            reloaded,
            ctx=pred_ctx,
            contract=contract,
            csv_payload=("\n".join(lines) + "\n").encode("utf-8"),
        )
        batch_res = reloaded.prediction_engine.predict(
            context=pred_ctx,
            artifact=pred_eng,
            contract=contract,
            model=v1.version.identity,
        )
        assert batch_res.successful_count == 2
        assert batch_res.rejected_count == 0
        assert all(p.status == PredictionStatus.SUCCEEDED for p in batch_res.predictions)
        assert all(
            p.derived_rate is not None and p.derived_rate.value is not None
            for p in batch_res.predictions
        )
