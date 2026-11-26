"""Controlled initial training, explicit retraining, promotion, and rollback service."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

from taqamind.core.contracts import (
    DatasetContract,
    DatasetPurpose,
    IndustrialTaskContract,
    ValidationReport,
)
from taqamind.core.interfaces.artifacts import (
    ArtifactFormat,
    ArtifactReference,
    calculate_bytes_sha256,
)
from taqamind.core.interfaces.candidates import TrainingRunResult
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.dataset import DatasetArtifact, DatasetMetadata
from taqamind.core.interfaces.dataset_registry import (
    DatasetRegistry,
    RecordLineageRegistry,
)
from taqamind.core.interfaces.evaluation import (
    AcceptancePolicy,
    EvaluationDecision,
    FinalEvaluationResult,
    FrozenModelSelection,
)
from taqamind.core.interfaces.leakage import LeakageValidationResult
from taqamind.core.interfaces.ml import (
    DatasetSplitter,
    LeakageValidator,
    ModelTrainer,
    TrainingReadinessChecker,
)
from taqamind.core.interfaces.model_registry import (
    ModelFeatureSpecification,
    ModelRegistryEntry,
    ModelScope,
    ModelStatus,
    ModelTransitionAction,
    ModelTransitionRequest,
    ModelVersion,
    PromotionComparison,
)
from taqamind.core.interfaces.prediction import PredictionModelIdentity
from taqamind.core.interfaces.processing import (
    ColumnMapper,
    DataCleaner,
    DataIngestion,
    DataProfiler,
    FeatureEngineer,
    SemanticValidator,
)
from taqamind.core.interfaces.processing_models import ApprovedMapping, ProfilingResult
from taqamind.core.interfaces.readiness import TrainingReadinessResult
from taqamind.core.interfaces.results import ExecutionStatus
from taqamind.core.interfaces.splitting import (
    SplitConfiguration,
    SplitManifest,
    SplitStrategy,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage, ModelArtifactStorage
from taqamind.core.interfaces.training import (
    CandidateConfiguration,
    CandidateRole,
    ModelFamily,
    ModelParameter,
    RegressionMetric,
    TrainingBudget,
    TrainingConfiguration,
)
from taqamind.core.interfaces.training_registry import (
    TrainingIntent,
    TrainingMode,
    TrainingRegistry,
    TrainingReuseKey,
    calculate_split_fingerprint,
    calculate_training_configuration_fingerprint,
)
from taqamind.domains.iron_manufacturing.adapters._store import (
    read_artifact_df as _read_artifact_df,
)
from taqamind.domains.iron_manufacturing.adapters.data_access import (
    LocalProtectedTestDataProvider,
)
from taqamind.domains.iron_manufacturing.adapters.model_registry import LocalModelRegistry
from taqamind.domains.iron_manufacturing.adapters.modeling import (
    ProtectedFinalModelEvaluator,
    deserialize_trusted_bundle,
)
from taqamind.domains.iron_manufacturing.adapters.training_registry import (
    LocalEvaluationRegistry,
    LocalSplitRegistry,
)
from taqamind.domains.iron_manufacturing.common.hashing import (
    calculate_contract_sha256,
    calculate_implementation_sha256,
)
from taqamind.domains.iron_manufacturing.config import (
    DEFAULT_RANDOM_SEED,
    ZERO_INFLATION_HURDLE_THRESHOLD,
)
from taqamind.domains.iron_manufacturing.services.lineage_service import (
    LineageResolutionSummary,
    register_dataset_lineage,
)


@dataclass(frozen=True)
class TrainingWorkflowOutcome:
    """Structured audit outcome of an initial training or explicit retraining request."""

    status: str  # "succeeded" | "rejected" | "blocked" | "reused"
    mode: TrainingMode
    execution_id: UUID
    source_file_sha256: str
    is_duplicate_source_file: bool
    profiling_result: ProfilingResult | None
    lineage_summary: LineageResolutionSummary | None
    readiness_result: TrainingReadinessResult | None
    split_manifest: SplitManifest | None
    leakage_validation: LeakageValidationResult | None
    training_run: TrainingRunResult | None
    final_evaluation: FinalEvaluationResult | None
    model_entry: ModelRegistryEntry | None
    active_model_entry: ModelRegistryEntry | None
    reused_training_run: bool
    reuse_key_fingerprint: str | None
    validation_report: ValidationReport | None
    blocked_reason: str | None
    promotion_comparison: PromotionComparison | None = None

    def to_summary_dict(self) -> dict[str, Any]:
        eval_dict: dict[str, Any] | None = None
        if self.final_evaluation is not None:
            eval_dict = {
                "evaluation_id": str(self.final_evaluation.evaluation_id),
                "decision": self.final_evaluation.decision.value,
                "baseline_test_mae": round(self.final_evaluation.baseline_mae, 6),
                "selected_test_mae": round(self.final_evaluation.selected_mae, 6),
                "test_improvement_pct": (
                    round(self.final_evaluation.test_improvement_pct, 3)
                    if self.final_evaluation.test_improvement_pct is not None
                    else None
                ),
                "rejection_reasons": list(self.final_evaluation.rejection_reasons),
            }
        return {
            "status": self.status,
            "mode": self.mode.value,
            "execution_id": str(self.execution_id),
            "source_file_sha256": self.source_file_sha256,
            "is_duplicate_source_file": self.is_duplicate_source_file,
            "reused_training_run": self.reused_training_run,
            "reuse_key_fingerprint": self.reuse_key_fingerprint,
            "model_version_id": (
                str(self.model_entry.version.model_version_id) if self.model_entry else None
            ),
            "model_version_number": (
                self.model_entry.version.version_number if self.model_entry else None
            ),
            "model_status": self.model_entry.status.value if self.model_entry else None,
            "selected_family": (
                self.model_entry.version.candidate.family.value if self.model_entry else None
            ),
            "active_model_version_id": (
                str(self.active_model_entry.version.model_version_id)
                if self.active_model_entry
                else None
            ),
            "final_evaluation": eval_dict,
            "blocked_reason": self.blocked_reason,
        }


def build_default_training_configuration(
    *,
    contract: DatasetContract,
    zero_target_fraction: float = 0.0,
    random_seed: int = DEFAULT_RANDOM_SEED,
    include_xgboost: bool = True,
) -> TrainingConfiguration:
    """Select bounded, allowlisted candidates tailored to dataset characteristics."""
    candidates: list[CandidateConfiguration] = [
        CandidateConfiguration(
            family=ModelFamily.DUMMY_REGRESSOR,
            role=CandidateRole.BASELINE,
            parameters=(ModelParameter(name="strategy", value="median"),),
        ),
    ]

    # Add industrial historical rate baseline when denominator is approved & pre-production
    if contract.target.derived_rate is not None:
        denom = contract.get_feature(contract.target.derived_rate.denominator)
        if denom.name in contract.predictor_names and contract.is_available_at_prediction(denom):
            candidates.append(
                CandidateConfiguration(
                    family=ModelFamily.HISTORICAL_RATE,
                    role=CandidateRole.BASELINE,
                    parameters=(
                        ModelParameter(
                            name="denominator_column",
                            value=contract.target.derived_rate.denominator,
                        ),
                    ),
                )
            )

    candidates.extend(
        [
            CandidateConfiguration(
                family=ModelFamily.RIDGE,
                role=CandidateRole.CHALLENGER,
                parameters=(ModelParameter(name="alpha", value=2.0),),
            ),
            CandidateConfiguration(
                family=ModelFamily.ELASTIC_NET,
                role=CandidateRole.CHALLENGER,
                parameters=(
                    ModelParameter(name="alpha", value=0.05),
                    ModelParameter(name="l1_ratio", value=0.4),
                    ModelParameter(name="max_iter", value=2000),
                ),
            ),
            CandidateConfiguration(
                family=ModelFamily.RANDOM_FOREST,
                role=CandidateRole.CHALLENGER,
                parameters=(
                    ModelParameter(name="n_estimators", value=80),
                    ModelParameter(name="max_depth", value=10),
                    ModelParameter(name="min_samples_leaf", value=2),
                ),
            ),
            CandidateConfiguration(
                family=ModelFamily.HIST_GRADIENT_BOOSTING,
                role=CandidateRole.CHALLENGER,
                parameters=(
                    ModelParameter(name="max_iter", value=100),
                    ModelParameter(name="max_depth", value=5),
                    ModelParameter(name="learning_rate", value=0.08),
                    ModelParameter(name="min_samples_leaf", value=5),
                ),
            ),
        ]
    )

    if include_xgboost:
        candidates.append(
            CandidateConfiguration(
                family=ModelFamily.XGBOOST,
                role=CandidateRole.CHALLENGER,
                parameters=(
                    ModelParameter(name="n_estimators", value=90),
                    ModelParameter(name="max_depth", value=4),
                    ModelParameter(name="learning_rate", value=0.08),
                    ModelParameter(name="subsample", value=0.85),
                ),
            )
        )

    if zero_target_fraction >= ZERO_INFLATION_HURDLE_THRESHOLD:
        candidates.append(
            CandidateConfiguration(
                family=ModelFamily.HURDLE_REGRESSION,
                role=CandidateRole.CHALLENGER,
                parameters=(
                    ModelParameter(name="max_depth", value=4),
                    ModelParameter(name="learning_rate", value=0.08),
                ),
            )
        )

    return TrainingConfiguration(
        primary_metric=RegressionMetric.MAE,
        secondary_metrics=(
            RegressionMetric.RMSE,
            RegressionMetric.WAPE,
            RegressionMetric.R2,
        ),
        candidates=tuple(candidates),
        budget=TrainingBudget(
            max_trials=max(12, len(candidates)),
            max_training_seconds=300,
            max_parallel_jobs=1,
            max_memory_mb=2048,
        ),
        random_seed=random_seed,
        justification=(
            "Controlled bounded candidate search comparing dummy/rate baselines against "
            "linear and tree-based regressors (plus hurdle model when zero-inflated)."
        ),
    )


def build_default_split_configuration(
    contract: DatasetContract,
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> SplitConfiguration:
    """Choose default leakage-safe splitting strategy for the contract."""
    if contract.time_column and contract.grouping_columns:
        return SplitConfiguration(
            strategy=SplitStrategy.GROUP_TEMPORAL,
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            random_seed=random_seed,
            justification=(
                "Combined chronological group-temporal split prevents order/supplier leakage "
                "and temporal look-ahead."
            ),
        )
    if contract.time_column:
        return SplitConfiguration(
            strategy=SplitStrategy.TEMPORAL,
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            random_seed=random_seed,
            justification="Chronological temporal split prevents future look-ahead.",
        )
    if contract.grouping_columns:
        return SplitConfiguration(
            strategy=SplitStrategy.GROUP,
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            random_seed=random_seed,
            justification="Group-disjoint split prevents dependent industrial batch leakage.",
        )
    return SplitConfiguration(
        strategy=SplitStrategy.RANDOM,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        random_seed=random_seed,
        justification="Fallback random split because no time or grouping columns are declared.",
    )


class IronTrainingService:
    """Orchestrates initial training, explicit retraining, promotion, and rollback."""

    def __init__(
        self,
        *,
        dataset_storage: DatasetArtifactStorage,
        model_storage: ModelArtifactStorage,
        dataset_registry: DatasetRegistry,
        lineage_registry: RecordLineageRegistry,
        split_registry: LocalSplitRegistry,
        training_registry: TrainingRegistry,
        evaluation_registry: LocalEvaluationRegistry,
        model_registry: LocalModelRegistry,
        test_data_provider: LocalProtectedTestDataProvider,
        ingestion: DataIngestion,
        profiler: DataProfiler,
        column_mapper: ColumnMapper,
        semantic_validator: SemanticValidator,
        data_cleaner: DataCleaner,
        feature_engineer: FeatureEngineer,
        readiness_checker: TrainingReadinessChecker,
        splitter: DatasetSplitter,
        leakage_validator: LeakageValidator,
        model_trainer: ModelTrainer,
    ) -> None:
        self._dataset_storage = dataset_storage
        self._model_storage = model_storage
        self._dataset_registry = dataset_registry
        self._lineage_registry = lineage_registry
        self._split_registry = split_registry
        self._training_registry = training_registry
        self._evaluation_registry = evaluation_registry
        self._model_registry = model_registry
        self._test_data_provider = test_data_provider
        self._ingestion = ingestion
        self._profiler = profiler
        self._column_mapper = column_mapper
        self._semantic_validator = semantic_validator
        self._data_cleaner = data_cleaner
        self._feature_engineer = feature_engineer
        self._readiness_checker = readiness_checker
        self._splitter = splitter
        self._leakage_validator = leakage_validator
        self._model_trainer = model_trainer

    def run_training(
        self,
        *,
        factory_id: UUID,
        task: IndustrialTaskContract,
        csv_path: Path | str,
        mode: TrainingMode,
        requested_by: str,
        reason: str,
        approved_by: str = "factory_engineer",
        confirm_all_proposed_mappings: bool = False,
        confirmed_source_columns: tuple[str, ...] = (),
        split_configuration: SplitConfiguration | None = None,
        training_configuration: TrainingConfiguration | None = None,
        acceptance_policy: AcceptancePolicy | None = None,
        allow_confirmed_corrections: bool = False,
        correction_reason: str | None = None,
        auto_activate_first: bool = True,
    ) -> TrainingWorkflowOutcome:
        """Execute end-to-end controlled training or explicit retraining."""
        contract: DatasetContract = task.dataset_contract
        scope = ModelScope(
            factory_id=factory_id,
            domain=contract.domain,
            process=contract.process,
            task_id=task.task_id,
        )
        current_active = self._model_registry.get_active(scope=scope)

        # Enforce explicit training mode semantics against current lifecycle state
        if mode == TrainingMode.INITIAL and current_active is not None:
            raise ValueError(
                "An active model already exists for this factory and task; "
                "use explicit TrainingMode.RETRAINING instead of INITIAL."
            )
        if mode == TrainingMode.RETRAINING and current_active is None:
            raise ValueError(
                "Explicit retraining requires an existing active model version; "
                "run TrainingMode.INITIAL first."
            )

        # Predefine acceptance policy BEFORE creating execution context
        policy = acceptance_policy or AcceptancePolicy(
            minimum_improvement_pct=5.0,
            justification=(
                "Challenger must reduce protected-test MAE by at least 5% relative "
                "to the best baseline model."
            ),
            created_at=datetime.now(UTC) - timedelta(seconds=1),
        )

        raw_bytes = Path(csv_path).read_bytes()
        source_sha = calculate_bytes_sha256(raw_bytes)
        raw_revision_id = uuid4()
        context = PipelineContext.from_task(
            factory_id=factory_id,
            dataset_revision_id=raw_revision_id,
            task=task,
            purpose=DatasetPurpose.TRAINING,
        )
        # Guarantee policy precedes context creation even on fast clocks
        if policy.created_at > context.created_at:
            policy = policy.model_copy(
                update={"created_at": context.created_at - timedelta(microseconds=1)}
            )

        intent = TrainingIntent(
            context=context,
            mode=mode,
            requested_by=requested_by,
            reason=reason,
            previous_model_version_id=(
                current_active.version.model_version_id
                if mode == TrainingMode.RETRAINING and current_active is not None
                else None
            ),
        )
        self._training_registry.register_intent(factory_id=factory_id, intent=intent)

        # Register immutable RAW artifact & detect exact duplicate file uploads
        raw_ref = ArtifactReference(
            factory_id=factory_id,
            format=ArtifactFormat.CSV,
            content_sha256=source_sha,
            size_bytes=len(raw_bytes),
        )
        self._dataset_storage.put(
            factory_id=factory_id,
            reference=raw_ref,
            stream=io.BytesIO(raw_bytes),
        )
        raw_artifact = DatasetArtifact(
            metadata=DatasetMetadata.create_raw(context=context, source_file_sha256=source_sha),
            reference=raw_ref,
        )
        reg_res = self._dataset_registry.register(factory_id=factory_id, artifact=raw_artifact)
        is_dup_file = reg_res.has_matching_source

        def _blocked(
            reason_msg: str,
            *,
            report: ValidationReport | None = None,
            prof: ProfilingResult | None = None,
            lin: LineageResolutionSummary | None = None,
            readiness: TrainingReadinessResult | None = None,
            manifest: SplitManifest | None = None,
            leak: LeakageValidationResult | None = None,
        ) -> TrainingWorkflowOutcome:
            return TrainingWorkflowOutcome(
                status="blocked",
                mode=mode,
                execution_id=context.execution_id,
                source_file_sha256=source_sha,
                is_duplicate_source_file=is_dup_file,
                profiling_result=prof,
                lineage_summary=lin,
                readiness_result=readiness,
                split_manifest=manifest,
                leakage_validation=leak,
                training_run=None,
                final_evaluation=None,
                model_entry=None,
                active_model_entry=current_active,
                reused_training_run=False,
                reuse_key_fingerprint=None,
                validation_report=report,
                blocked_reason=reason_msg,
            )

        # Stage 1: Ingestion
        ingest_res = self._ingestion.ingest(
            context=context, artifact=raw_artifact, contract=contract
        )
        if ingest_res.status != ExecutionStatus.SUCCEEDED or ingest_res.output_artifact is None:
            return _blocked("CSV ingestion failed.", report=ingest_res.report)
        self._dataset_registry.register(factory_id=factory_id, artifact=ingest_res.output_artifact)

        # Stage 2: Profiling
        prof_res = self._profiler.profile(
            context=context, artifact=ingest_res.output_artifact, contract=contract
        )

        # Stage 3: Column Mapping
        proposal = self._column_mapper.propose_mapping(
            context=context, artifact=ingest_res.output_artifact, contract=contract
        )
        confirmed_proposal = proposal
        if confirm_all_proposed_mappings:
            confirmed_entries = tuple(
                e.model_copy(update={"source_unit": e.canonical_unit})
                if e.canonical_unit is not None and e.source_unit is None
                else e
                for e in proposal.entries
            )
            confirmed_proposal = proposal.model_copy(update={"entries": confirmed_entries})
        confirmed = (
            tuple(e.source_column for e in confirmed_proposal.entries if e.requires_confirmation)
            if confirm_all_proposed_mappings
            else confirmed_source_columns
        )
        try:
            approved = ApprovedMapping(
                proposal=confirmed_proposal,
                approved_by=approved_by,
                confirmed_source_columns=confirmed,
            )
            map_res = self._column_mapper.apply_mapping(
                context=context,
                artifact=ingest_res.output_artifact,
                contract=contract,
                mapping=approved,
            )
        except ValueError as exc:
            return _blocked(f"Column mapping rejected: {exc}", prof=prof_res)

        if map_res.status != ExecutionStatus.SUCCEEDED or map_res.output_artifact is None:
            return _blocked(
                "Column mapping or unit conversion blocked.",
                report=map_res.report,
                prof=prof_res,
            )
        self._dataset_registry.register(factory_id=factory_id, artifact=map_res.output_artifact)

        # Stage 4: Semantic Validation
        sem_res = self._semantic_validator.validate(
            context=context, artifact=map_res.output_artifact, contract=contract
        )
        if sem_res.status != ExecutionStatus.SUCCEEDED:
            return _blocked(
                "Semantic validation blocked dataset.",
                report=sem_res.report,
                prof=prof_res,
            )

        # Stage 5: Deterministic Cleaning
        clean_res = self._data_cleaner.clean(
            context=context, artifact=map_res.output_artifact, contract=contract
        )
        if clean_res.status != ExecutionStatus.SUCCEEDED or clean_res.output_artifact is None:
            return _blocked(
                "Deterministic cleaning blocked dataset.",
                report=clean_res.report,
                prof=prof_res,
            )
        self._dataset_registry.register(factory_id=factory_id, artifact=clean_res.output_artifact)

        # Stage 6: Deterministic Feature Engineering
        eng_res = self._feature_engineer.engineer(
            context=context, artifact=clean_res.output_artifact, contract=contract
        )
        if eng_res.status != ExecutionStatus.SUCCEEDED or eng_res.output_artifact is None:
            return _blocked(
                "Deterministic feature engineering blocked dataset.",
                report=eng_res.report,
                prof=prof_res,
            )
        engineered_artifact = eng_res.output_artifact
        self._dataset_registry.register(factory_id=factory_id, artifact=engineered_artifact)

        # Stage 7: Record Lineage & Membership Registration
        eng_df = _read_artifact_df(self._dataset_storage, factory_id, engineered_artifact)
        try:
            lineage_summary = register_dataset_lineage(
                context=context,
                artifact=engineered_artifact,
                df=eng_df,
                contract=contract,
                lineage_registry=self._lineage_registry,
                allow_confirmed_corrections=allow_confirmed_corrections,
                correction_confirmed_by=approved_by if allow_confirmed_corrections else None,
                correction_reason=correction_reason,
                register_new_records=True,
            )
        except ValueError as exc:
            return _blocked(str(exc), prof=prof_res)

        # Stage 8: Training Readiness Assessment
        readiness = self._readiness_checker.assess(
            context=context,
            artifact=engineered_artifact,
            contract=contract,
        )
        if not readiness.eligible_for_split:
            return _blocked(
                "Training readiness gate blocked execution.",
                report=readiness.result.report,
                prof=prof_res,
                lin=lineage_summary,
                readiness=readiness,
            )

        # Stage 9: Leakage-Safe Splitting & Leakage Validation
        split_cfg = split_configuration or build_default_split_configuration(contract)
        manifest = self._splitter.split(readiness=readiness, configuration=split_cfg)
        self._split_registry.register_split(factory_id=factory_id, manifest=manifest)

        leakage = self._leakage_validator.validate(readiness=readiness, manifest=manifest)
        self._split_registry.register_leakage_validation(
            factory_id=factory_id, validation=leakage
        )
        if not leakage.eligible_for_training:
            return _blocked(
                "Leakage validation failed; production training blocked.",
                report=leakage.report,
                prof=prof_res,
                lin=lineage_summary,
                readiness=readiness,
                manifest=manifest,
                leak=leakage,
            )

        authorization = self._split_registry.issue_training_authorization(
            factory_id=factory_id,
            validation_id=leakage.validation_id,
        )

        # Stage 10: Build TrainingConfiguration & Compute Reuse Key
        target_col = contract.target.feature.name
        zero_frac = float((pd.to_numeric(eng_df[target_col], errors="coerce") == 0.0).mean())
        train_cfg = training_configuration or build_default_training_configuration(
            contract=contract,
            zero_target_fraction=zero_frac,
        )
        contract_sha = calculate_contract_sha256(contract)
        impl_sha = calculate_implementation_sha256()
        split_sha = calculate_split_fingerprint(manifest, lineage_summary.manifest)
        cfg_sha = calculate_training_configuration_fingerprint(train_cfg, policy)

        reuse_key = TrainingReuseKey.from_intent(
            intent=intent,
            dataset_content_sha256=engineered_artifact.reference.content_sha256,
            contract_sha256=contract_sha,
            split_sha256=split_sha,
            training_configuration_sha256=cfg_sha,
            implementation_sha256=impl_sha,
        )

        reusable_run = self._training_registry.find_reusable_result(key=reuse_key)
        if reusable_run is not None:
            return TrainingWorkflowOutcome(
                status="reused",
                mode=mode,
                execution_id=context.execution_id,
                source_file_sha256=source_sha,
                is_duplicate_source_file=is_dup_file,
                profiling_result=prof_res,
                lineage_summary=lineage_summary,
                readiness_result=readiness,
                split_manifest=manifest,
                leakage_validation=leakage,
                training_run=reusable_run,
                final_evaluation=None,
                model_entry=None,
                active_model_entry=current_active,
                reused_training_run=True,
                reuse_key_fingerprint=reuse_key.fingerprint,
                validation_report=leakage.report,
                blocked_reason=None,
            )

        # Stage 11: Controlled Model Training (development data ONLY)
        training_run = self._model_trainer.train(
            authorization=authorization,
            contract=contract,
            configuration=train_cfg,
            policy=policy,
        )
        self._training_registry.register_result(
            factory_id=factory_id,
            key=reuse_key,
            result=training_run,
        )

        if not training_run.eligible_for_final_evaluation:
            return TrainingWorkflowOutcome(
                status="rejected",
                mode=mode,
                execution_id=context.execution_id,
                source_file_sha256=source_sha,
                is_duplicate_source_file=is_dup_file,
                profiling_result=prof_res,
                lineage_summary=lineage_summary,
                readiness_result=readiness,
                split_manifest=manifest,
                leakage_validation=leakage,
                training_run=training_run,
                final_evaluation=None,
                model_entry=None,
                active_model_entry=current_active,
                reused_training_run=False,
                reuse_key_fingerprint=reuse_key.fingerprint,
                validation_report=leakage.report,
                blocked_reason=(
                    "No challenger outperformed the best baseline on development validation data."
                ),
            )

        # Stage 12: Freeze selection BEFORE accessing protected test data
        selection = FrozenModelSelection.from_training_run(
            training_run=training_run,
            policy=policy,
        )

        # Stage 13: Protected Final Evaluation (single use, no refitting)
        evaluator = ProtectedFinalModelEvaluator(
            contract=contract,
            evaluation_registry=self._evaluation_registry,
            test_data_provider=self._test_data_provider,
            model_storage=self._model_storage,
        )
        final_eval = evaluator.evaluate(selection=selection, manifest=manifest)

        # Stage 14: Register immutable candidate ModelVersion & apply evaluation decision
        best_challenger_res = training_run.best_challenger
        assert best_challenger_res is not None
        assert best_challenger_res.model_reference is not None
        selected_cfg = training_run.configuration.get_candidate(selection.selected_candidate_id)

        version_num = self._model_registry.next_version_number(scope=scope)
        model_identity = PredictionModelIdentity(
            model_version_id=uuid4(),
            factory_id=factory_id,
            domain=contract.domain,
            process=contract.process,
            task_id=task.task_id,
            dataset_id=contract.dataset_id,
            contract_version=contract.version,
            task_version=task.version,
            target_version=contract.target.version,
        )
        feature_spec = ModelFeatureSpecification(
            feature_names=tuple(col for col in contract.predictor_names if col in eng_df.columns),
            target_name=target_col,
            target_unit=contract.target.feature.unit,
        )
        feature_spec.validate_contract(contract)

        model_version = ModelVersion(
            identity=model_identity,
            version_number=version_num,
            training_run_id=training_run.training_run_id,
            selection_id=selection.selection_id,
            training_dataset_revision_id=engineered_artifact.metadata.revision_id,
            split_id=manifest.split_id,
            candidate=selected_cfg,
            artifact=best_challenger_res.model_reference,
            feature_specification=feature_spec,
            contract_sha256=contract_sha,
            implementation_sha256=impl_sha,
            previous_model_version_id=(
                current_active.version.model_version_id if current_active is not None else None
            ),
        )
        model_version.validate_selection(selection)

        candidate_entry = self._model_registry.register_candidate(
            scope=scope, version=model_version
        )
        evaluated_entry = self._model_registry.apply_evaluation(
            scope=scope,
            model_version_id=candidate_entry.version.model_version_id,
            evaluation=final_eval,
        )

        # Optional fair comparison on same independent test records for RETRAINING
        promotion_comparison: PromotionComparison | None = None
        if (
            mode == TrainingMode.RETRAINING
            and current_active is not None
            and evaluated_entry.status == ModelStatus.VALIDATED
        ):
            promotion_comparison = self._compare_active_vs_challenger_on_test(
                scope=scope,
                active_entry=current_active,
                challenger_entry=evaluated_entry,
                manifest=manifest,
                eng_df=eng_df,
                lineage_summary=lineage_summary,
                contract=contract,
                minimum_improvement_pct=policy.minimum_improvement_pct,
            )

        # First activation ONLY on INITIAL accepted training when auto_activate_first=True
        final_entry = evaluated_entry
        active_after = current_active
        if (
            mode == TrainingMode.INITIAL
            and auto_activate_first
            and evaluated_entry.status == ModelStatus.VALIDATED
            and current_active is None
        ):
            activation_req = ModelTransitionRequest(
                scope=scope,
                action=ModelTransitionAction.FIRST_ACTIVATION,
                target_model_version_id=evaluated_entry.version.model_version_id,
                requested_by=requested_by,
                reason="Initial validated model activation after passing protected evaluation.",
            )
            final_entry = self._model_registry.transition(request=activation_req)
            active_after = final_entry

        outcome_status = (
            "succeeded" if final_eval.decision == EvaluationDecision.ACCEPTED else "rejected"
        )
        return TrainingWorkflowOutcome(
            status=outcome_status,
            mode=mode,
            execution_id=context.execution_id,
            source_file_sha256=source_sha,
            is_duplicate_source_file=is_dup_file,
            profiling_result=prof_res,
            lineage_summary=lineage_summary,
            readiness_result=readiness,
            split_manifest=manifest,
            leakage_validation=leakage,
            training_run=training_run,
            final_evaluation=final_eval,
            model_entry=final_entry,
            active_model_entry=active_after,
            reused_training_run=False,
            reuse_key_fingerprint=reuse_key.fingerprint,
            validation_report=leakage.report,
            blocked_reason=(
                None
                if outcome_status == "succeeded"
                else f"Model rejected by acceptance policy: {final_eval.rejection_reasons}"
            ),
            promotion_comparison=promotion_comparison,
        )

    def _compare_active_vs_challenger_on_test(
        self,
        *,
        scope: ModelScope,
        active_entry: ModelRegistryEntry,
        challenger_entry: ModelRegistryEntry,
        manifest: SplitManifest,
        eng_df: pd.DataFrame,
        lineage_summary: LineageResolutionSummary,
        contract: DatasetContract,
        minimum_improvement_pct: float,
    ) -> PromotionComparison:
        rec_to_row = {m.record_id: m.row_position for m in lineage_summary.manifest.memberships}
        test_rows = [rec_to_row[rid] for rid in manifest.test.record_ids]
        test_df = eng_df.iloc[test_rows]
        y_true = test_df[contract.target.feature.name].to_numpy(dtype=float)

        def _mae_for_entry(entry: ModelRegistryEntry) -> float:
            with self._model_storage.open_verified(
                factory_id=scope.factory_id,
                reference=entry.version.artifact,
            ) as stream:
                env = deserialize_trusted_bundle(stream.read())
            X_te = test_df.loc[:, list(env["feature_names"])]
            preds = np.asarray(env["pipeline"].predict(X_te), dtype=float)
            target_min = contract.target.feature.constraints.minimum
            if target_min is not None:
                preds = np.clip(preds, target_min, None)
            return float(np.mean(np.abs(y_true - preds)))

        return PromotionComparison(
            scope=scope,
            active_model_version_id=active_entry.version.model_version_id,
            challenger_model_version_id=challenger_entry.version.model_version_id,
            evaluation_dataset_revision_id=manifest.dataset_revision_id,
            evaluation_record_ids=manifest.test.record_ids,
            active_mae=_mae_for_entry(active_entry),
            challenger_mae=_mae_for_entry(challenger_entry),
            minimum_improvement_pct=minimum_improvement_pct,
        )

    def promote_model(
        self,
        *,
        scope: ModelScope,
        target_model_version_id: UUID,
        comparison: PromotionComparison,
        requested_by: str,
        reason: str,
    ) -> ModelRegistryEntry:
        """Explicitly promote a VALIDATED challenger over the current ACTIVE model."""
        active = self._model_registry.get_active(scope=scope)
        if active is None:
            raise ValueError("Cannot promote without an existing active model.")
        req = ModelTransitionRequest(
            scope=scope,
            action=ModelTransitionAction.PROMOTION,
            target_model_version_id=target_model_version_id,
            expected_active_model_version_id=active.version.model_version_id,
            requested_by=requested_by,
            reason=reason,
            comparison=comparison,
        )
        return self._model_registry.transition(request=req)

    def rollback_model(
        self,
        *,
        scope: ModelScope,
        target_archived_version_id: UUID,
        requested_by: str,
        reason: str,
    ) -> ModelRegistryEntry:
        """Explicitly rollback to a previously validated ARCHIVED model version."""
        active = self._model_registry.get_active(scope=scope)
        if active is None:
            raise ValueError("Cannot rollback without an existing active model.")
        req = ModelTransitionRequest(
            scope=scope,
            action=ModelTransitionAction.ROLLBACK,
            target_model_version_id=target_archived_version_id,
            expected_active_model_version_id=active.version.model_version_id,
            requested_by=requested_by,
            reason=reason,
        )
        return self._model_registry.transition(request=req)
