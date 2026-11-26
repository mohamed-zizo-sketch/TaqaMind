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
from taqamind.core.feature_engineering import DeterministicFeatureEngineer
from taqamind.core.ingestion import CsvDataIngestion
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
    RecordUsage,
    RecordUsageRole,
    SplitConfiguration,
    SplitManifest,
    SplitRole,
    SplitStrategy,
    TrainingConfiguration,
    TrainingMode,
    TrainingReuseKey,
    TrainingRunResult,
    calculate_bytes_sha256,
    calculate_file_sha256,
    calculate_split_fingerprint,
    calculate_training_configuration_fingerprint,
)
from taqamind.core.lineage import LocalRecordLineageRegistry
from taqamind.core.preprocessing import ContractDataCleaner
from taqamind.core.profiling import PandasDataProfiler
from taqamind.core.registry import (
    DomainRegistry,
    LocalDatasetArtifactStorage,
    LocalDatasetRegistry,
    LocalModelArtifactStorage,
)
from taqamind.core.validation import ContractSemanticValidator
from taqamind.domains.food_beverage.adapters import (
    FoodBeverageControlledAutoMLTrainer,
    FoodBeverageDatasetSplitter,
    FoodBeverageLeakageValidator,
    FoodBeveragePredictionEngine,
    FoodBeverageProtectedFinalEvaluator,
    FoodBeverageRowMembershipBuilder,
    FoodBeverageTrainingReadinessChecker,
    LocalActiveModelProvider,
    LocalDevelopmentDataProvider,
    LocalEvaluationRegistry,
    LocalModelRegistry,
    LocalProtectedTestDataProvider,
    LocalRecordUsageRegistry,
    LocalSplitRegistry,
    LocalTrainingRegistry,
    RowLineageSummary,
    UnitAwareColumnMapperAdapter,
    hash_contract,
)
from taqamind.domains.food_beverage.config import DOMAIN_ID, FoodBeverageStorageConfig
from taqamind.domains.food_beverage.domain import register_food_beverage


@dataclass
class PreparedDatasetBundle:
    """Result of RAW -> INGESTED -> MAPPED -> VALIDATED -> CLEANED -> ENGINEERED + Lineage."""

    context: PipelineContext
    contract: DatasetContract
    engineered_artifact: DatasetArtifact
    profiling: ProfilingResult
    lineage: RowLineageSummary


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

        # Real Shared Core P1.1-P1.5 engines
        self.dataset_storage = LocalDatasetArtifactStorage(self.storage_config.root_dir / "blobs")
        self.model_storage = LocalModelArtifactStorage(self.storage_config.root_dir / "blobs")
        self.dataset_registry = LocalDatasetRegistry(
            self.storage_config.registries_dir, self.dataset_storage
        )
        self.ingestion = CsvDataIngestion(self.dataset_storage)
        self.profiler = PandasDataProfiler(self.dataset_storage)
        self.mapper = UnitAwareColumnMapperAdapter(self.dataset_storage)
        self.validator = ContractSemanticValidator(self.dataset_storage)
        self.cleaner = ContractDataCleaner(self.dataset_storage)
        self.engineer = DeterministicFeatureEngineer(self.dataset_storage)
        self.lineage_registry = LocalRecordLineageRegistry(
            self.storage_config.registries_dir,
            dataset_registry=self.dataset_registry,
        )

        # Protocol-compatible adapters for Core gaps (AUD-008 + P1.6-P1.8)
        self.membership_builder = FoodBeverageRowMembershipBuilder(self.lineage_registry)
        self.readiness = FoodBeverageTrainingReadinessChecker(self.dataset_storage)
        self.splitter = FoodBeverageDatasetSplitter(self.dataset_storage)
        self.leakage = FoodBeverageLeakageValidator()
        self.split_registry = LocalSplitRegistry(self.storage_config.registries_dir)
        self.usage_registry = LocalRecordUsageRegistry(self.storage_config.registries_dir)
        self.training_registry = LocalTrainingRegistry(self.storage_config.registries_dir)
        self.evaluation_registry = LocalEvaluationRegistry(self.storage_config.registries_dir)
        self.trainer = FoodBeverageControlledAutoMLTrainer(self.model_storage)
        self.evaluator = FoodBeverageProtectedFinalEvaluator(self.model_storage)
        self.model_registry = LocalModelRegistry(self.storage_config.registries_dir)
        self.active_model_provider = LocalActiveModelProvider(self.model_registry)
        self.predictor = FoodBeveragePredictionEngine(
            dataset_storage=self.dataset_storage,
            model_storage=self.model_storage,
            active_provider=self.active_model_provider,
        )
        self._reuse_cache: dict[str, TrainingWorkflowResult] = {}

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
        confirmed_cols = tuple(
            e.source_column for e in proposal.entries if e.requires_confirmation
        )
        approved = ApprovedMapping(
            proposal=proposal,
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

        eng_res = self.engineer.engineer(
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

        lineage = self.membership_builder.register_rows(
            context=context,
            contract=contract,
            dataset_revision_id=eng_res.output_artifact.metadata.revision_id,
            df=eng_df,
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
            lineage=prepared.lineage,
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

        exposed_test_ids = self.usage_registry.get_exposed_record_ids(
            scope=prepared.lineage.scope,
            role=RecordUsageRole.PROTECTED_TEST,
        )
        manifest, partitions = self.splitter.split(
            context=prepared.context,
            artifact=prepared.engineered_artifact,
            contract=prepared.contract,
            configuration=s_cfg,
            lineage=prepared.lineage,
            exposed_test_record_ids=exposed_test_ids,
        )
        self.split_registry.register(manifest)

        _, auth = self.leakage.validate(
            readiness=readiness,
            manifest=manifest,
            partitions=partitions,
            exposed_test_record_ids=exposed_test_ids,
        )

        contract_sha = hash_contract(prepared.contract)
        impl_sha = calculate_bytes_sha256(self.IMPLEMENTATION_VERSION.encode("utf-8"))
        split_sha = calculate_split_fingerprint(
            manifest,
            membership=prepared.lineage.manifest,
        )
        model_scope = ModelScope(
            factory_id=factory_id,
            domain=DOMAIN_ID,
            process=process_id,
            task_id=prepared.context.task_id,
        )
        prev_model_id: UUID | None = None
        if intent == TrainingMode.RETRAINING:
            prev_active = self.model_registry.get_active_model(model_scope)
            if prev_active is not None:
                prev_model_id = prev_active.version.model_version_id
            else:
                versions = self.model_registry.list_versions(model_scope)
                if versions:
                    prev_model_id = versions[-1].version.model_version_id
        _ = TrainingReuseKey(
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

        dev_provider = LocalDevelopmentDataProvider(partitions)
        training_run = self.trainer.train(
            authorization=auth,
            configuration=cfg,
            contract=prepared.contract,
            dev_provider=dev_provider,
        )
        self.training_registry.register(training_run)

        selection = FrozenModelSelection.from_training_run(
            training_run=training_run,
            policy=policy,
        )

        rec_ver_map = {
            m.record_id: m.record_version_id
            for m in prepared.lineage.manifest.memberships
        }
        test_usages = [
            RecordUsage(
                scope=prepared.lineage.scope,
                record_id=rid,
                record_version_id=rec_ver_map[rid],
                execution_id=prepared.context.execution_id,
                split_id=manifest.split_id,
                role=RecordUsageRole.PROTECTED_TEST,
            )
            for rid in manifest.test.record_ids
        ]
        self.usage_registry.record(test_usages)

        test_provider = LocalProtectedTestDataProvider(partitions[SplitRole.TEST])
        evaluation = self.evaluator.evaluate(
            selection=selection,
            manifest=manifest,
            contract=prepared.contract,
            test_provider=test_provider,
        )
        self.evaluation_registry.register(evaluation)

        selected_res = next(
            r for r in training_run.results if r.candidate_id == selection.selected_candidate_id
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

        active_before = self.model_registry.get_active_model(scope)
        predictor_names = tuple(
            f.name
            for f in prepared.contract.features
            if f.role.value == "predictor" and prepared.contract.is_available_at_prediction(f)
        )
        version = ModelVersion(
            identity=identity,
            version_number=self.model_registry.next_version_number(scope),
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

        entry = self.model_registry.register_evaluated(
            version=version,
            evaluation_id=evaluation.evaluation_id,
            decision=evaluation.decision,
        )
        if (
            evaluation.decision == EvaluationDecision.ACCEPTED
            and auto_activate_if_accepted
            and (intent == TrainingMode.INITIAL or active_before is None)
        ):
            entry = self.model_registry.transition(
                ModelTransitionRequest(
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
        active = self.model_registry.get_active_model(scope)
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
            lineage=prepared.lineage,
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
        active = self.model_registry.get_active_model(scope)
        all_entries = self.model_registry.list_entries(scope)
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
