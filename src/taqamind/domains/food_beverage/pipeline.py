"""End-to-end Food & Beverage ML Pipeline Orchestrator wired onto Core P1.1-P1.5."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from taqamind.core.contracts import (
    ConfirmationStatus,
    DatasetContract,
    DatasetPurpose,
    TargetConfirmation,
)
from taqamind.core.evaluation import ProtectedFinalEvaluator
from taqamind.core.interfaces import (
    AcceptancePolicy,
    ApprovedMapping,
    ArtifactFormat,
    ArtifactReference,
    CandidateConfiguration,
    CandidateRole,
    DatasetArtifact,
    DatasetMetadata,
    DatasetState,
    EvaluationDecision,
    FinalEvaluationResult,
    FrozenModelSelection,
    ModelFamily,
    ModelFeatureSpecification,
    ModelParameter,
    ModelRegistryEntry,
    ModelScope,
    ModelTransitionAction,
    ModelTransitionRequest,
    ModelVersion,
    PipelineContext,
    PredictionBatchResult,
    PredictionModelIdentity,
    ProfilingResult,
    PromotionComparison,
    RecordScope,
    RegressionMetric,
    SplitConfiguration,
    SplitManifest,
    SplitStrategy,
    TrainingConfiguration,
    TrainingIntent,
    TrainingMode,
    TrainingReuseKey,
    TrainingRunResult,
    calculate_bytes_sha256,
    calculate_file_sha256,
    calculate_training_configuration_fingerprint,
)
from taqamind.core.lineage import MembershipBuildSummary
from taqamind.core.pipeline import CorePlatformServices
from taqamind.core.registry import DomainRegistry
from taqamind.domains.food_beverage.adapters.unit_mapper import UnitAwareColumnMapperAdapter
from taqamind.domains.food_beverage.config import DOMAIN_ID, FoodBeverageStorageConfig
from taqamind.domains.food_beverage.domain import register_food_beverage


@dataclass
class PreparedDatasetBundle:
    """Result of RAW -> INGESTED -> MAPPED -> VALIDATED -> CLEANED -> ENGINEERED + Lineage."""

    context: PipelineContext
    contract: DatasetContract
    engineered_artifact: DatasetArtifact
    profiling: ProfilingResult
    lineage: MembershipBuildSummary


@dataclass
class TrainingWorkflowResult:
    """Complete result of an initial training or explicit retraining workflow."""

    prepared: PreparedDatasetBundle
    split_manifest: SplitManifest
    training_run: TrainingRunResult
    evaluation: FinalEvaluationResult
    registry_entry: ModelRegistryEntry | None
    reused_existing: bool = False


def default_training_configuration(seed: int = 42) -> TrainingConfiguration:
    """Build a controlled default candidate search configuration."""

    return TrainingConfiguration(
        candidates=(
            CandidateConfiguration(
                family=ModelFamily.DUMMY_REGRESSOR,
                role=CandidateRole.BASELINE,
                parameters=(ModelParameter(name="strategy", value="median"),),
            ),
            CandidateConfiguration(
                family=ModelFamily.RIDGE,
                role=CandidateRole.CHALLENGER,
                parameters=(ModelParameter(name="alpha", value=1.0),),
            ),
            CandidateConfiguration(
                family=ModelFamily.RANDOM_FOREST,
                role=CandidateRole.CHALLENGER,
                parameters=(
                    ModelParameter(name="n_estimators", value=60),
                    ModelParameter(name="max_depth", value=6),
                ),
            ),
            CandidateConfiguration(
                family=ModelFamily.HIST_GRADIENT_BOOSTING,
                role=CandidateRole.CHALLENGER,
                parameters=(
                    ModelParameter(name="max_iter", value=60),
                    ModelParameter(name="max_depth", value=5),
                    ModelParameter(name="learning_rate", value=0.08),
                ),
            ),
        ),
        random_seed=seed,
        justification=(
            "Controlled Food & Beverage regression baseline vs. linear and tree challengers."
        ),
    )


class FoodBeveragePipeline:
    """Orchestrator wiring Core P1.1-P1.5 engines directly with P1.6-P1.8 adapters."""

    IMPLEMENTATION_VERSION = "0.1.0"

    def __init__(
        self,
        storage_config: FoodBeverageStorageConfig | None = None,
        domain_registry: DomainRegistry | None = None,
    ) -> None:
        self.storage_config = storage_config or FoodBeverageStorageConfig()
        self.storage_config.ensure_dirs()
        self.domain_registry = domain_registry or DomainRegistry()

        # Initialize all Core P1.1–P1.8 services via CorePlatformServices
        services = CorePlatformServices.create_local(self.storage_config.root_dir)
        self._services = services  # kept for create_evaluator access

        # P1.1–P1.5
        self.dataset_storage = services.dataset_storage
        self.model_storage = services.model_storage
        self.dataset_registry = services.dataset_registry
        self.ingestion = services.ingestion
        self.profiler = services.profiler
        # Use UnitAwareColumnMapperAdapter (AUD-001) wrapping the Core mapper
        self.mapper = UnitAwareColumnMapperAdapter(
            services.dataset_storage, core_mapper=services.mapper
        )
        self.validator = services.validator
        self.cleaner = services.cleaner
        self.feature_engineer = services.feature_engineer
        self.lineage_registry = services.lineage_registry

        # P1.6–P1.8
        self.membership_builder = services.membership_builder
        self.readiness = services.readiness_checker
        self.splitter = services.splitter
        self.leakage = services.leakage_validator
        self.split_registry = services.split_registry
        self.usage_registry = services.usage_registry
        self.training_registry = services.training_registry
        self.evaluation_registry = services.evaluation_registry
        self.trainer = services.trainer
        self.model_registry = services.model_registry
        self.active_model_provider = services.active_model_provider
        self.predictor = services.prediction_engine
        self._reuse_cache: dict[str, TrainingWorkflowResult] = {}

    def _get_evaluator(self, contract: DatasetContract) -> ProtectedFinalEvaluator:
        """Create a contract-bound ProtectedFinalEvaluator via CorePlatformServices."""
        return self._services.create_evaluator(contract)

    def get_contract(
        self,
        process_id: str,
        target_confirmation: TargetConfirmation | None = None,
    ) -> DatasetContract:
        confirmations = {process_id: target_confirmation} if target_confirmation else None
        reg = DomainRegistry()
        domain = register_food_beverage(reg, confirmations=confirmations)
        return domain.get_process(process_id).tasks[0].dataset_contract

    def prepare_dataset(
        self,
        *,
        csv_path: Path,
        factory_id: UUID,
        process_id: str,
        purpose: DatasetPurpose,
        target_confirmation: TargetConfirmation | None = None,
        explicit_source_units: dict[str, str] | None = None,
        model_version_id: UUID | None = None,
    ) -> PreparedDatasetBundle:
        """Run Ingestion, Profiling, Mapping, Validation, Cleaning, Engineering, and Lineage."""

        contract = self.get_contract(process_id, target_confirmation=target_confirmation)
        reg = DomainRegistry()
        domain = register_food_beverage(
            reg,
            confirmations={process_id: target_confirmation} if target_confirmation else None,
        )
        task = domain.get_process(process_id).tasks[0]

        raw_rev = uuid4()
        context = PipelineContext(
            execution_id=uuid4(),
            factory_id=factory_id,
            domain=DOMAIN_ID,
            process=process_id,
            dataset_id=contract.dataset_id,
            dataset_revision_id=raw_rev,
            contract_version=contract.version,
            task_id=task.task_id,
            task_version=task.version,
            target_version=contract.target.version,
            purpose=purpose,
            model_version_id=model_version_id,
        )

        raw_bytes = csv_path.read_bytes()
        raw_sha = calculate_file_sha256(csv_path)
        raw_ref = ArtifactReference(
            artifact_id=raw_rev,
            factory_id=factory_id,
            format=ArtifactFormat.CSV,
            content_sha256=raw_sha,
            size_bytes=len(raw_bytes),
        )
        self.dataset_storage.put(
            factory_id=factory_id,
            reference=raw_ref,
            stream=BytesIO(raw_bytes),
        )
        raw_meta = DatasetMetadata(
            revision_id=raw_rev,
            root_revision_id=raw_rev,
            factory_id=factory_id,
            domain=DOMAIN_ID,
            process=process_id,
            dataset_id=contract.dataset_id,
            contract_version=contract.version,
            task_id=task.task_id,
            task_version=task.version,
            target_version=contract.target.version,
            source_file_sha256=raw_sha,
            state=DatasetState.RAW,
            row_count=None,
            columns=(),
        )
        raw_artifact = DatasetArtifact(metadata=raw_meta, reference=raw_ref)
        self.dataset_registry.register(factory_id=factory_id, artifact=raw_artifact)

        ingest_res = self.ingestion.ingest(
            context=context, artifact=raw_artifact, contract=contract
        )
        if not ingest_res.is_successful or ingest_res.output_artifact is None:
            raise ValueError(f"Ingestion failed: {ingest_res.report or ingest_res.error}")
        self.dataset_registry.register(factory_id=factory_id, artifact=ingest_res.output_artifact)

        profiling = self.profiler.profile(
            context=context,
            artifact=ingest_res.output_artifact,
            contract=contract,
        )

        proposal = self.mapper.propose_mapping(
            context=context,
            artifact=ingest_res.output_artifact,
            contract=contract,
            explicit_source_units=explicit_source_units,
        )
        if purpose == DatasetPurpose.PREDICTION:
            target_tokens = {
                contract.target.feature.name.casefold(),
                *(a.casefold() for a in contract.target.feature.aliases),
            }
            leaked_targets = [
                c.name for c in ingest_res.output_artifact.metadata.columns
                if c.name.casefold() in target_tokens
            ]
            if leaked_targets:
                raise ValueError(
                    f"Prediction dataset must not include the target column(s): {leaked_targets}"
                )
        confirmed_entries = tuple(
            entry.model_copy(update={"source_unit": entry.canonical_unit})
            if entry.canonical_unit is not None and entry.source_unit is None
            else entry
            for entry in proposal.entries
        )
        confirmed_proposal = proposal.model_copy(update={"entries": confirmed_entries})
        confirmed_cols = tuple(
            e.source_column for e in confirmed_entries if e.requires_confirmation
        )
        approved = ApprovedMapping(
            proposal=confirmed_proposal,
            approved_by="food_beverage_pipeline",
            confirmed_source_columns=confirmed_cols,
        )
        map_res = self.mapper.apply_mapping(
            context=context,
            artifact=ingest_res.output_artifact,
            contract=contract,
            mapping=approved,
        )
        if not map_res.is_successful or map_res.output_artifact is None:
            raise ValueError(f"Mapping failed: {map_res.report}")
        self.dataset_registry.register(factory_id=factory_id, artifact=map_res.output_artifact)

        val_stage = self.validator.validate(
            context=context,
            artifact=map_res.output_artifact,
            contract=contract,
        )
        if not val_stage.is_successful or (
            val_stage.report is not None and val_stage.report.has_blocking_issues
        ):
            msgs = [i.message for i in val_stage.report.issues] if val_stage.report else []
            raise ValueError(f"Semantic validation blocked dataset: {msgs}")

        clean_res = self.cleaner.clean(
            context=context,
            artifact=map_res.output_artifact,
            contract=contract,
        )
        if not clean_res.is_successful or clean_res.output_artifact is None:
            raise ValueError(f"Cleaning failed: {clean_res.report}")
        self.dataset_registry.register(factory_id=factory_id, artifact=clean_res.output_artifact)

        eng_res = self.feature_engineer.engineer(
            context=context,
            artifact=clean_res.output_artifact,
            contract=contract,
        )
        if not eng_res.is_successful or eng_res.output_artifact is None:
            raise ValueError(f"Feature engineering failed: {eng_res.report}")
        self.dataset_registry.register(factory_id=factory_id, artifact=eng_res.output_artifact)

        with self.dataset_storage.open_verified(
            factory_id=factory_id,
            reference=eng_res.output_artifact.reference,
        ) as stream:
            eng_df = pd.read_csv(stream, keep_default_na=True)

        bkey_cols = list(contract.business_keys)
        if bkey_cols and eng_df.duplicated(subset=bkey_cols).any():
            raise ValueError(
                f"Duplicate business key detected in dataset for columns {bkey_cols}."
            )

        rec_scope = RecordScope(
            factory_id=factory_id,
            domain=DOMAIN_ID,
            process=process_id,
            dataset_id=contract.dataset_id,
            identity_namespace="production_record",
        )
        lineage = self.membership_builder.build_and_register(
            scope=rec_scope,
            artifact=eng_res.output_artifact,
            contract=contract,
            confirm_updates_by="food_beverage_pipeline",
        )

        return PreparedDatasetBundle(
            context=context,
            contract=contract,
            engineered_artifact=eng_res.output_artifact,
            profiling=profiling,
            lineage=lineage,
        )

    def train_model(
        self,
        *,
        csv_path: Path,
        factory_id: UUID,
        process_id: str,
        target_confirmation: TargetConfirmation | None = None,
        explicit_source_units: dict[str, str] | None = None,
        split_config: SplitConfiguration | None = None,
        training_config: TrainingConfiguration | None = None,
        acceptance_policy: AcceptancePolicy | None = None,
        intent: TrainingMode = TrainingMode.INITIAL,
        auto_activate_if_accepted: bool = True,
    ) -> TrainingWorkflowResult:
        """Execute controlled training, protected evaluation, and registry persistence."""

        prepared = self.prepare_dataset(
            csv_path=csv_path,
            factory_id=factory_id,
            process_id=process_id,
            purpose=DatasetPurpose.TRAINING,
            target_confirmation=target_confirmation,
            explicit_source_units=explicit_source_units,
        )

        cfg = training_config or default_training_configuration()
        policy = acceptance_policy or AcceptancePolicy(
            minimum_improvement_pct=1.0,
            justification="Model must beat baseline MAE by at least 1% on protected test data.",
            created_at=prepared.context.created_at,
        )
        cfg_sha = calculate_training_configuration_fingerprint(cfg, policy)
        src_sha = prepared.engineered_artifact.metadata.source_file_sha256
        reuse_id = f"{factory_id}:{process_id}:{src_sha}:{cfg_sha}"
        if intent == TrainingMode.RETRAINING and reuse_id in self._reuse_cache:
            cached = self._reuse_cache[reuse_id]
            return TrainingWorkflowResult(
                prepared=cached.prepared,
                split_manifest=cached.split_manifest,
                training_run=cached.training_run,
                evaluation=cached.evaluation,
                registry_entry=cached.registry_entry,
                reused_existing=True,
            )

        readiness = self.readiness.assess(
            context=prepared.context,
            artifact=prepared.engineered_artifact,
            contract=prepared.contract,
        )
        readiness.require_split_eligibility()

        s_cfg = split_config or SplitConfiguration(
            strategy=SplitStrategy.TEMPORAL,
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
            justification="Temporal holdout for pre-production industrial batch forecasting.",
        )

        manifest = self.splitter.split(
            readiness=readiness,
            configuration=s_cfg,
        )
        self.split_registry.register_split(factory_id=factory_id, manifest=manifest)

        leakage = self.leakage.validate(
            readiness=readiness,
            manifest=manifest,
        )
        self.split_registry.register_leakage_validation(
            factory_id=factory_id,
            validation=leakage,
        )
        auth = self.split_registry.issue_training_authorization(
            factory_id=factory_id,
            validation_id=leakage.validation_id,
        )

        contract_sha = calculate_bytes_sha256(
            prepared.contract.model_dump_json().encode("utf-8")
        )
        impl_sha = calculate_bytes_sha256(self.IMPLEMENTATION_VERSION.encode("utf-8"))
        split_sha = calculate_bytes_sha256(manifest.model_dump_json().encode("utf-8"))
        model_scope = ModelScope(
            factory_id=factory_id,
            domain=DOMAIN_ID,
            process=process_id,
            task_id=prepared.context.task_id,
        )
        prev_model_id: UUID | None = None
        if intent == TrainingMode.RETRAINING:
            prev_active = self.model_registry.get_active(scope=model_scope)
            if prev_active is not None:
                prev_model_id = prev_active.version.model_version_id
            else:
                versions = self.model_registry.list_versions(scope=model_scope)
                if versions:
                    prev_model_id = versions[-1].version.model_version_id

        training_intent = TrainingIntent(
            context=prepared.context,
            mode=intent,
            requested_by="food_beverage_pipeline",
            reason="Food & Beverage model training.",
            previous_model_version_id=prev_model_id,
        )
        self.training_registry.register_intent(factory_id=factory_id, intent=training_intent)

        reuse_key = TrainingReuseKey(
            factory_id=factory_id,
            domain=DOMAIN_ID,
            process=process_id,
            task_id=prepared.context.task_id,
            dataset_id=prepared.contract.dataset_id,
            contract_version=prepared.contract.version,
            task_version=prepared.context.task_version,
            target_version=prepared.contract.target.version,
            dataset_content_sha256=prepared.engineered_artifact.reference.content_sha256,
            contract_sha256=contract_sha,
            split_sha256=split_sha,
            training_configuration_sha256=cfg_sha,
            implementation_sha256=impl_sha,
            mode=intent,
            previous_model_version_id=prev_model_id,
        )

        training_run = self.trainer.train(
            authorization=auth,
            contract=prepared.contract,
            configuration=cfg,
            policy=policy,
        )
        self.training_registry.register_result(
            factory_id=factory_id,
            key=reuse_key,
            result=training_run,
        )

        selection = FrozenModelSelection.from_training_run(
            training_run=training_run,
            policy=policy,
        )

        evaluation = self._get_evaluator(prepared.contract).evaluate(
            selection=selection,
            manifest=manifest,
        )

        selected_res = next(
            r
            for r in training_run.successful_results
            if r.candidate_id == selection.selected_candidate_id
        )
        selected_cfg = next(
            c for c in cfg.candidates if c.candidate_id == selection.selected_candidate_id
        )
        assert selected_res.model_reference is not None

        scope = ModelScope(
            factory_id=factory_id,
            domain=DOMAIN_ID,
            process=process_id,
            task_id=prepared.context.task_id,
        )
        model_ver_id = uuid4()
        identity = PredictionModelIdentity(
            model_version_id=model_ver_id,
            factory_id=factory_id,
            domain=DOMAIN_ID,
            process=process_id,
            task_id=prepared.context.task_id,
            dataset_id=prepared.contract.dataset_id,
            contract_version=prepared.contract.version,
            task_version=prepared.context.task_version,
            target_version=prepared.contract.target.version,
        )

        active_before = self.model_registry.get_active(scope=scope)
        existing_versions = self.model_registry.list_versions(scope=scope)
        next_version_number = (
            max((v.version.version_number for v in existing_versions), default=0) + 1
        )
        predictor_names = tuple(
            f.name
            for f in prepared.contract.features
            if f.role.value == "predictor" and prepared.contract.is_available_at_prediction(f)
        )
        version = ModelVersion(
            identity=identity,
            version_number=next_version_number,
            training_run_id=training_run.training_run_id,
            selection_id=selection.selection_id,
            training_dataset_revision_id=prepared.engineered_artifact.metadata.revision_id,
            split_id=manifest.split_id,
            candidate=selected_cfg,
            artifact=selected_res.model_reference,
            feature_specification=ModelFeatureSpecification(
                feature_names=predictor_names,
                target_name=prepared.contract.target.feature.name,
                target_unit=prepared.contract.target.feature.unit,
            ),
            contract_sha256=contract_sha,
            implementation_sha256=impl_sha,
            previous_model_version_id=(
                active_before.version.model_version_id if active_before else None
            ),
        )

        self.model_registry.register_candidate(scope=scope, version=version)
        entry = self.model_registry.apply_evaluation(
            scope=scope,
            model_version_id=version.model_version_id,
            evaluation=evaluation,
        )
        if (
            evaluation.decision == EvaluationDecision.ACCEPTED
            and auto_activate_if_accepted
            and (intent == TrainingMode.INITIAL or active_before is None)
        ):
            entry = self.model_registry.transition(
                request=ModelTransitionRequest(
                    scope=scope,
                    target_model_version_id=model_ver_id,
                    expected_active_model_version_id=None,
                    action=ModelTransitionAction.FIRST_ACTIVATION,
                    requested_by="food_beverage_pipeline",
                    reason="Initial validated model activated.",
                    comparison=None,
                )
            )

        res = TrainingWorkflowResult(
            prepared=prepared,
            split_manifest=manifest,
            training_run=training_run,
            evaluation=evaluation,
            registry_entry=entry,
            reused_existing=False,
        )
        self._reuse_cache[reuse_id] = res
        return res

    def promote_model(
        self,
        *,
        factory_id: UUID,
        process_id: str,
        challenger_result: TrainingWorkflowResult,
        minimum_improvement_pct: float = 1.0,
        requested_by: str = "food_beverage_pipeline",
        reason: str = "Challenger outperformed active model on protected test data.",
    ) -> ModelRegistryEntry:
        """Promote a validated challenger model while enforcing PromotionComparison."""
        if challenger_result.registry_entry is None:
            raise ValueError("Challenger training result has no registry entry.")
        scope = challenger_result.registry_entry.version.scope
        active = self.model_registry.get_active(scope=scope)
        if active is None:
            raise ValueError(
                f"No ACTIVE model exists for scope={scope}; use FIRST_ACTIVATION instead."
            )
        comparison = PromotionComparison(
            scope=scope,
            active_model_version_id=active.version.model_version_id,
            challenger_model_version_id=challenger_result.registry_entry.version.model_version_id,
            evaluation_dataset_revision_id=(
                challenger_result.prepared.engineered_artifact.metadata.revision_id
            ),
            evaluation_record_ids=challenger_result.split_manifest.test.record_ids,
            metric=RegressionMetric.MAE,
            active_mae=challenger_result.evaluation.baseline_mae,
            challenger_mae=challenger_result.evaluation.selected_mae,
            minimum_improvement_pct=minimum_improvement_pct,
        )
        return self.model_registry.transition(
            request=ModelTransitionRequest(
                scope=scope,
                action=ModelTransitionAction.PROMOTION,
                target_model_version_id=(
                    challenger_result.registry_entry.version.model_version_id
                ),
                expected_active_model_version_id=active.version.model_version_id,
                requested_by=requested_by,
                reason=reason,
                comparison=comparison,
            )
        )

    def predict(
        self,
        *,
        csv_path: Path,
        factory_id: UUID,
        process_id: str,
        explicit_source_units: dict[str, str] | None = None,
        target_confirmation: TargetConfirmation | None = None,
    ) -> PredictionBatchResult:
        """Run inference-only prediction using the factory's saved ACTIVE model."""

        reg = DomainRegistry()
        domain = register_food_beverage(reg)
        task_id = domain.get_process(process_id).tasks[0].task_id
        scope = ModelScope(
            factory_id=factory_id,
            domain=DOMAIN_ID,
            process=process_id,
            task_id=task_id,
        )
        active = self.model_registry.get_active(scope=scope)
        if active is None:
            raise ValueError(
                f"No ACTIVE model exists for factory={factory_id}, process={process_id}."
            )

        effective_conf = target_confirmation or TargetConfirmation(
            status=ConfirmationStatus.CONFIRMED,
            confirmed_by=f"active_model_{active.version.model_version_id}",
            confirmed_at=active.version.created_at,
            notes="Target semantics confirmed during active model training.",
        )

        prepared = self.prepare_dataset(
            csv_path=csv_path,
            factory_id=factory_id,
            process_id=process_id,
            purpose=DatasetPurpose.PREDICTION,
            target_confirmation=effective_conf,
            explicit_source_units=explicit_source_units,
            model_version_id=active.version.model_version_id,
        )
        return self.predictor.predict(
            context=prepared.context,
            artifact=prepared.engineered_artifact,
            contract=prepared.contract,
            model=active.version.identity,
        )

    def get_model_info(self, *, factory_id: UUID, process_id: str) -> dict[str, Any]:
        """Return active and registered model versions for a factory and process."""

        contract = self.get_contract(process_id)
        reg = DomainRegistry()
        domain = register_food_beverage(reg)
        task_id = domain.get_process(process_id).tasks[0].task_id
        scope = ModelScope(
            factory_id=factory_id,
            domain=DOMAIN_ID,
            process=process_id,
            task_id=task_id,
        )
        active = self.model_registry.get_active(scope=scope)
        all_entries = self.model_registry.list_versions(scope=scope)
        return {
            "factory_id": str(factory_id),
            "process_id": process_id,
            "dataset_id": contract.dataset_id,
            "active_model_version_id": (
                str(active.version.model_version_id) if active else None
            ),
            "active_version_number": active.version.version_number if active else None,
            "registered_versions": [
                {
                    "model_version_id": str(e.version.model_version_id),
                    "version_number": e.version.version_number,
                    "status": e.status.value,
                    "family": e.version.candidate.family.value,
                }
                for e in all_entries
            ],
        }
