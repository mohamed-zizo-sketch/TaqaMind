"""End-to-end textile pipeline orchestrator wiring Core P1.1-P1.5 and P1.6-P1.8 adapters."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from taqamind.core.contracts import (
    DatasetContract,
    DatasetPurpose,
    TargetConfirmation,
)
from taqamind.core.feature_engineering import DerivedFeatureSpec, DeterministicFeatureEngineer
from taqamind.core.ingestion import CsvDataIngestion
from taqamind.core.interfaces import (
    AcceptancePolicy,
    ApprovedMapping,
    ArtifactFormat,
    ArtifactReference,
    DatasetArtifact,
    DatasetMetadata,
    DatasetState,
    EvaluationDecision,
    ExecutionStatus,
    FinalEvaluationResult,
    FrozenModelSelection,
    LeakageValidationResult,
    ModelFeatureSpecification,
    ModelScope,
    ModelTransitionAction,
    ModelTransitionRequest,
    ModelVersion,
    PipelineContext,
    PredictionBatchResult,
    PredictionModelIdentity,
    SplitConfiguration,
    SplitManifest,
    StageResult,
    TrainingAuthorization,
    TrainingConfiguration,
    TrainingIntent,
    TrainingMode,
    TrainingReadinessResult,
    TrainingReuseKey,
    TrainingRunResult,
    calculate_bytes_sha256,
    calculate_file_sha256,
)
from taqamind.core.lineage import LocalRecordLineageRegistry
from taqamind.core.preprocessing import ContractDataCleaner
from taqamind.core.registry import (
    DomainRegistry,
    LocalDatasetArtifactStorage,
    LocalDatasetRegistry,
    LocalModelArtifactStorage,
)
from taqamind.core.validation import ContractSemanticValidator
from taqamind.domains.textile.adapters.data_access_training import (
    LocalDevelopmentDataProvider,
    LocalEvaluationRegistry,
    LocalProtectedTestDataProvider,
    LocalTrainingRegistry,
    TextileControlledAutoMLTrainer,
    TextileProtectedFinalEvaluator,
)
from taqamind.domains.textile.adapters.membership import (
    RowLineageSummary,
    TextileRowMembershipBuilder,
)
from taqamind.domains.textile.adapters.model_registry_prediction import (
    LocalActiveModelProvider,
    LocalModelRegistry,
    TextilePredictionEngine,
)
from taqamind.domains.textile.adapters.readiness_splitting import (
    LocalRecordUsageRegistry,
    LocalSplitRegistry,
    TextileDatasetSplitter,
    TextileLeakageValidator,
    TextileTrainingReadinessChecker,
    hash_contract,
)
from taqamind.domains.textile.adapters.unit_mapper import UnitAwareColumnMapperAdapter
from taqamind.domains.textile.config import (
    build_default_acceptance_policy,
    build_default_split_configuration,
    build_default_training_configuration,
)
from taqamind.domains.textile.domain import (
    TEXTILE_DOMAIN_ID,
    register_textile,
)
from taqamind.domains.textile.knitting import (
    KNITTING_DATASET_ID,
    KNITTING_PROCESS_ID,
    KNITTING_TASK_ID,
    build_knitting_dataset_contract,
    build_knitting_derived_specs,
    build_synthetic_knitting_confirmation,
)
from taqamind.domains.textile.weaving import (
    WEAVING_DATASET_ID,
    WEAVING_PROCESS_ID,
    WEAVING_TASK_ID,
    build_synthetic_weaving_confirmation,
    build_weaving_dataset_contract,
    build_weaving_derived_specs,
)

IMPLEMENTATION_SHA256 = hashlib.sha256(b"taqamind-textile-v1.0.0").hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedDatasetResult:
    """Outputs of RAW -> INGESTED -> MAPPED -> VALIDATED -> CLEANED -> ENGINEERED."""

    context: PipelineContext
    raw_artifact: DatasetArtifact
    ingested_result: StageResult
    mapped_result: StageResult
    validated_result: StageResult
    cleaned_result: StageResult
    engineered_result: StageResult
    engineered_artifact: DatasetArtifact
    lineage_summary: RowLineageSummary | None = None


@dataclass(frozen=True, slots=True)
class TrainingWorkflowResult:
    """Outputs of readiness -> split -> leakage -> training -> evaluation -> registry."""

    readiness: TrainingReadinessResult
    split_manifest: SplitManifest
    leakage_validation: LeakageValidationResult
    authorization: TrainingAuthorization
    training_run: TrainingRunResult
    selection: FrozenModelSelection
    evaluation: FinalEvaluationResult
    model_version: ModelVersion | None
    activated: bool


class TextilePlatformStores:
    """Dependency container wiring Core P1.1-P1.5 engines and P1.6-P1.8 textile adapters."""

    def __init__(self, storage_dir: str | Path) -> None:
        self.root = Path(storage_dir).resolve()
        self.dataset_storage = LocalDatasetArtifactStorage(self.root / "datasets")
        self.model_storage = LocalModelArtifactStorage(self.root / "models")
        self.registry_dir = self.root / "registries"

        self.domain_registry = DomainRegistry()
        self.dataset_registry = LocalDatasetRegistry(self.registry_dir, self.dataset_storage)
        self.lineage_registry = LocalRecordLineageRegistry(
            self.registry_dir, dataset_registry=self.dataset_registry
        )
        self.split_registry = LocalSplitRegistry(self.registry_dir)
        self.usage_registry = LocalRecordUsageRegistry(self.registry_dir)
        self.training_registry = LocalTrainingRegistry(self.registry_dir)
        self.evaluation_registry = LocalEvaluationRegistry(
            self.registry_dir,
            split_registry=self.split_registry,
            usage_registry=self.usage_registry,
        )
        self.model_registry = LocalModelRegistry(
            self.registry_dir, model_storage=self.model_storage
        )

        # Core P1.2-P1.5 engines + AUD-001 unit-aware mapping adapter.
        self.ingestion = CsvDataIngestion(self.dataset_storage)
        self.mapper = UnitAwareColumnMapperAdapter(self.dataset_storage)
        self.validator = ContractSemanticValidator(self.dataset_storage)
        self.cleaner = ContractDataCleaner(self.dataset_storage)

        # Textile P1.6-P1.8 adapters.
        self.membership_builder = TextileRowMembershipBuilder(
            self.dataset_storage, self.lineage_registry
        )
        self.readiness_checker = TextileTrainingReadinessChecker(
            self.dataset_storage, self.lineage_registry
        )
        self.splitter = TextileDatasetSplitter(self.dataset_storage, self.lineage_registry)
        self.leakage_validator = TextileLeakageValidator(
            self.dataset_storage, self.lineage_registry, self.usage_registry
        )
        self.dev_provider = LocalDevelopmentDataProvider(
            registry_dir=self.registry_dir,
            dataset_storage=self.dataset_storage,
            dataset_registry=self.dataset_registry,
            lineage_registry=self.lineage_registry,
            split_registry=self.split_registry,
            usage_registry=self.usage_registry,
        )
        self.test_provider = LocalProtectedTestDataProvider(
            registry_dir=self.registry_dir,
            dataset_storage=self.dataset_storage,
            dataset_registry=self.dataset_registry,
            lineage_registry=self.lineage_registry,
            split_registry=self.split_registry,
            usage_registry=self.usage_registry,
            evaluation_registry=self.evaluation_registry,
        )
        self.trainer = TextileControlledAutoMLTrainer(self.dev_provider, self.model_storage)
        self.active_model_provider = LocalActiveModelProvider(
            self.model_registry, self.model_storage
        )
        self.prediction_engine = TextilePredictionEngine(
            dataset_storage=self.dataset_storage,
            model_registry=self.model_registry,
            active_model_provider=self.active_model_provider,
            lineage_registry=self.lineage_registry,
        )


def resolve_process_spec(
    process_id: str,
    *,
    confirmation: TargetConfirmation | None = None,
    use_synthetic_confirmation: bool = False,
) -> tuple[str, str, DatasetContract, tuple[DerivedFeatureSpec, ...]]:
    """Return (task_id, dataset_id, DatasetContract, derived_specs) for a textile process."""
    if process_id == WEAVING_PROCESS_ID:
        conf = confirmation or (
            build_synthetic_weaving_confirmation() if use_synthetic_confirmation else None
        )
        return (
            WEAVING_TASK_ID,
            WEAVING_DATASET_ID,
            build_weaving_dataset_contract(confirmation=conf),
            build_weaving_derived_specs(),
        )
    if process_id == KNITTING_PROCESS_ID:
        conf = confirmation or (
            build_synthetic_knitting_confirmation() if use_synthetic_confirmation else None
        )
        return (
            KNITTING_TASK_ID,
            KNITTING_DATASET_ID,
            build_knitting_dataset_contract(confirmation=conf),
            build_knitting_derived_specs(),
        )
    raise ValueError(f"Unsupported textile process '{process_id}'.")


class TextilePipelineService:
    """High-level orchestrator for Textile V1 training, retraining, and prediction."""

    def __init__(self, storage_dir: str | Path) -> None:
        self.stores = TextilePlatformStores(storage_dir)
        register_textile(self.stores.domain_registry)

    def _build_raw_artifact(
        self,
        *,
        factory_id: UUID,
        process_id: str,
        task_id: str,
        dataset_id: str,
        contract: DatasetContract,
        csv_path: Path,
        revision_id: UUID,
    ) -> DatasetArtifact:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Input CSV not found: '{csv_path}'.")
        sha256 = calculate_file_sha256(csv_path)
        size_bytes = csv_path.stat().st_size
        metadata = DatasetMetadata(
            revision_id=revision_id,
            root_revision_id=revision_id,
            factory_id=factory_id,
            domain=TEXTILE_DOMAIN_ID,
            process=process_id,
            dataset_id=dataset_id,
            contract_version=contract.version,
            task_id=task_id,
            task_version="1.0.0",
            target_version=contract.target.version,
            source_file_sha256=sha256,
            state=DatasetState.RAW,
            row_count=None,
            columns=(),
        )
        ref = ArtifactReference(
            factory_id=factory_id,
            format=ArtifactFormat.CSV,
            content_sha256=sha256,
            size_bytes=size_bytes,
        )
        with csv_path.open("rb") as stream:
            self.stores.dataset_storage.put(
                factory_id=factory_id, reference=ref, stream=stream
            )
        artifact = DatasetArtifact(metadata=metadata, reference=ref)
        self.stores.dataset_registry.register(factory_id=factory_id, artifact=artifact)
        return artifact

    def ingest_and_prepare(
        self,
        *,
        factory_id: UUID,
        process_id: str,
        csv_path: str | Path,
        purpose: DatasetPurpose,
        contract: DatasetContract | None = None,
        use_synthetic_confirmation: bool = False,
        approved_by: str = "textile_pipeline",
        register_lineage: bool = True,
        model_version_id: UUID | None = None,
    ) -> PreparedDatasetResult:
        """Execute RAW -> INGESTED -> MAPPED -> VALIDATED -> CLEANED -> ENGINEERED."""
        task_id, dataset_id, default_contract, derived_specs = resolve_process_spec(
            process_id, use_synthetic_confirmation=use_synthetic_confirmation
        )
        active_contract = contract or default_contract

        raw_revision_id = uuid4()
        raw_artifact = self._build_raw_artifact(
            factory_id=factory_id,
            process_id=process_id,
            task_id=task_id,
            dataset_id=dataset_id,
            contract=active_contract,
            csv_path=Path(csv_path),
            revision_id=raw_revision_id,
        )
        context = PipelineContext(
            factory_id=factory_id,
            dataset_revision_id=raw_artifact.metadata.revision_id,
            domain=TEXTILE_DOMAIN_ID,
            process=process_id,
            task_id=task_id,
            dataset_id=dataset_id,
            contract_version=active_contract.version,
            task_version="1.0.0",
            target_version=active_contract.target.version,
            purpose=purpose,
            model_version_id=model_version_id,
        )

        ingested = self.stores.ingestion.ingest(
            context=context, artifact=raw_artifact, contract=active_contract
        )
        if ingested.status != ExecutionStatus.SUCCEEDED or ingested.output_artifact is None:
            raise RuntimeError(f"Ingestion failed: {ingested.error or ingested.report}")
        self.stores.dataset_registry.register(
            factory_id=factory_id, artifact=ingested.output_artifact
        )

        proposal = self.stores.mapper.propose_mapping(
            context=context, artifact=ingested.output_artifact, contract=active_contract
        )
        approved = ApprovedMapping(
            proposal=proposal,
            approved_by=approved_by,
            confirmed_source_columns=tuple(e.source_column for e in proposal.entries),
        )
        mapped = self.stores.mapper.apply_mapping(
            context=context,
            artifact=ingested.output_artifact,
            contract=active_contract,
            mapping=approved,
        )
        if mapped.status != ExecutionStatus.SUCCEEDED or mapped.output_artifact is None:
            raise RuntimeError(f"Mapping failed: {mapped.error or mapped.report}")
        self.stores.dataset_registry.register(
            factory_id=factory_id, artifact=mapped.output_artifact
        )

        validated = self.stores.validator.validate(
            context=context, artifact=mapped.output_artifact, contract=active_contract
        )
        if validated.status in {ExecutionStatus.FAILED, ExecutionStatus.BLOCKED}:
            raise RuntimeError(f"Semantic validation blocked/failed: {validated.report}")

        cleaned = self.stores.cleaner.clean(
            context=context, artifact=mapped.output_artifact, contract=active_contract
        )
        if cleaned.status != ExecutionStatus.SUCCEEDED or cleaned.output_artifact is None:
            raise RuntimeError(f"Cleaning failed: {cleaned.error or cleaned.report}")
        self.stores.dataset_registry.register(
            factory_id=factory_id, artifact=cleaned.output_artifact
        )

        engineer = DeterministicFeatureEngineer(
            self.stores.dataset_storage, derived_specs=derived_specs
        )
        engineered = engineer.engineer(
            context=context, artifact=cleaned.output_artifact, contract=active_contract
        )
        if engineered.status != ExecutionStatus.SUCCEEDED or engineered.output_artifact is None:
            raise RuntimeError(
                f"Feature engineering failed: {engineered.error or engineered.report}"
            )
        engineered_artifact = engineered.output_artifact
        self.stores.dataset_registry.register(
            factory_id=factory_id, artifact=engineered_artifact
        )

        lineage_summary: RowLineageSummary | None = None
        if register_lineage:
            lineage_summary = self.stores.membership_builder.build_and_register(
                context=context,
                artifact=engineered_artifact,
                contract=active_contract,
            )

        return PreparedDatasetResult(
            context=context,
            raw_artifact=raw_artifact,
            ingested_result=ingested,
            mapped_result=mapped,
            validated_result=validated,
            cleaned_result=cleaned,
            engineered_result=engineered,
            engineered_artifact=engineered_artifact,
            lineage_summary=lineage_summary,
        )

    def train(
        self,
        *,
        factory_id: UUID,
        process_id: str,
        prepared: PreparedDatasetResult,
        contract: DatasetContract,
        mode: TrainingMode = TrainingMode.INITIAL,
        previous_model_version_id: UUID | None = None,
        requested_by: str = "textile_pipeline",
        reason: str = "Initial factory model training",
        split_config: SplitConfiguration | None = None,
        training_config: TrainingConfiguration | None = None,
        policy: AcceptancePolicy | None = None,
        activate_if_accepted: bool = True,
    ) -> TrainingWorkflowResult:
        """Execute readiness -> split -> leakage -> training -> evaluation -> model registry."""
        context = prepared.context
        readiness = self.stores.readiness_checker.assess(
            context=context,
            artifact=prepared.engineered_artifact,
            contract=contract,
        )
        readiness.require_split_eligibility()

        s_cfg = split_config or build_default_split_configuration(process_id)
        t_cfg = training_config or build_default_training_configuration()
        a_policy = policy or build_default_acceptance_policy().model_copy(
            update={"created_at": context.created_at}
        )

        manifest = self.stores.splitter.split(readiness=readiness, configuration=s_cfg)
        self.stores.split_registry.register_split(factory_id=factory_id, manifest=manifest)

        leakage = self.stores.leakage_validator.validate(readiness=readiness, manifest=manifest)
        self.stores.split_registry.register_leakage_validation(
            factory_id=factory_id, validation=leakage
        )
        authorization = self.stores.split_registry.issue_training_authorization(
            factory_id=factory_id, validation_id=leakage.validation_id
        )

        intent = TrainingIntent(
            context=context,
            mode=mode,
            requested_by=requested_by,
            reason=reason,
            previous_model_version_id=previous_model_version_id,
        )
        self.stores.training_registry.register_intent(factory_id=factory_id, intent=intent)

        reuse_key = TrainingReuseKey(
            factory_id=factory_id,
            domain=context.domain,
            process=context.process,
            task_id=context.task_id,
            dataset_id=context.dataset_id,
            contract_version=contract.version,
            task_version=context.task_version,
            target_version=contract.target.version,
            mode=mode,
            previous_model_version_id=previous_model_version_id,
            dataset_content_sha256=prepared.engineered_artifact.reference.content_sha256,
            contract_sha256=hash_contract(contract),
            split_sha256=calculate_bytes_sha256(
                manifest.model_dump_json().encode("utf-8")
            ),
            training_configuration_sha256=calculate_bytes_sha256(
                t_cfg.model_dump_json().encode("utf-8")
            ),
            implementation_sha256=IMPLEMENTATION_SHA256,
        )
        training_run = self.stores.trainer.train(
            authorization=authorization,
            contract=contract,
            configuration=t_cfg,
            policy=a_policy,
        )
        self.stores.training_registry.register_result(
            factory_id=factory_id, key=reuse_key, result=training_run
        )

        selection = FrozenModelSelection.from_training_run(
            training_run=training_run, policy=a_policy
        )
        evaluator = TextileProtectedFinalEvaluator(
            test_provider=self.stores.test_provider,
            evaluation_registry=self.stores.evaluation_registry,
            model_storage=self.stores.model_storage,
            contract=contract,
        )
        evaluation = evaluator.evaluate(selection=selection, manifest=manifest)

        scope = ModelScope(
            factory_id=factory_id,
            domain=context.domain,
            process=context.process,
            task_id=context.task_id,
        )
        existing_versions = self.stores.model_registry.list_versions(scope=scope)
        next_version_number = (
            max((v.version.version_number for v in existing_versions), default=0) + 1
        )
        selected_result = next(
            r
            for r in training_run.successful_results
            if r.candidate_id == selection.selected_candidate_id
        )
        selected_candidate = next(
            c for c in t_cfg.candidates if c.candidate_id == selection.selected_candidate_id
        )
        assert selected_result.model_reference is not None

        identity = PredictionModelIdentity(
            model_version_id=uuid4(),
            factory_id=factory_id,
            domain=context.domain,
            process=context.process,
            task_id=context.task_id,
            dataset_id=context.dataset_id,
            contract_version=contract.version,
            task_version=context.task_version,
            target_version=contract.target.version,
        )
        version = ModelVersion(
            identity=identity,
            version_number=next_version_number,
            training_run_id=training_run.training_run_id,
            selection_id=selection.selection_id,
            training_dataset_revision_id=prepared.engineered_artifact.metadata.revision_id,
            split_id=manifest.split_id,
            candidate=selected_candidate,
            artifact=selected_result.model_reference,
            feature_specification=ModelFeatureSpecification(
                feature_names=contract.predictor_names,
                target_name=contract.target.feature.name,
                target_unit=contract.target.feature.unit,
            ),
            contract_sha256=hash_contract(contract),
            implementation_sha256=IMPLEMENTATION_SHA256,
            previous_model_version_id=previous_model_version_id,
        )
        self.stores.model_registry.register_candidate(scope=scope, version=version)
        entry = self.stores.model_registry.apply_evaluation(
            scope=scope, model_version_id=version.model_version_id, evaluation=evaluation
        )

        activated = False
        if activate_if_accepted and evaluation.decision == EvaluationDecision.ACCEPTED:
            active_entry = self.stores.model_registry.get_active(scope=scope)
            action = (
                ModelTransitionAction.FIRST_ACTIVATION
                if active_entry is None
                else ModelTransitionAction.PROMOTION
            )
            req = ModelTransitionRequest(
                scope=scope,
                action=action,
                target_model_version_id=entry.version.model_version_id,
                expected_active_model_version_id=(
                    active_entry.version.model_version_id if active_entry else None
                ),
                requested_by=requested_by,
                reason=reason,
                comparison=None,
            )
            self.stores.model_registry.transition(request=req)
            activated = True

        return TrainingWorkflowResult(
            readiness=readiness,
            split_manifest=manifest,
            leakage_validation=leakage,
            authorization=authorization,
            training_run=training_run,
            selection=selection,
            evaluation=evaluation,
            model_version=version,
            activated=activated,
        )

    def predict(
        self,
        *,
        factory_id: UUID,
        process_id: str,
        csv_path: str | Path,
        contract: DatasetContract | None = None,
        model_version_id: UUID | None = None,
        register_lineage: bool = True,
    ) -> PredictionBatchResult:
        """Prepare prediction CSV and score using the active model with ZERO fitting."""
        task_id, _, default_contract, _ = resolve_process_spec(process_id)
        active_contract = contract or default_contract
        scope = ModelScope(
            factory_id=factory_id,
            domain=TEXTILE_DOMAIN_ID,
            process=process_id,
            task_id=task_id,
        )
        active_entry = self.stores.active_model_provider.get_active(scope=scope)
        if active_entry is None:
            raise RuntimeError(
                f"No active model found for factory '{factory_id}' and process '{process_id}'."
            )
        target_model = active_entry.version.identity
        if model_version_id is not None and target_model.model_version_id != model_version_id:
            raise ValueError(
                f"Requested model_version_id '{model_version_id}' is not active "
                f"(active is '{target_model.model_version_id}')."
            )

        prepared = self.ingest_and_prepare(
            factory_id=factory_id,
            process_id=process_id,
            csv_path=csv_path,
            purpose=DatasetPurpose.PREDICTION,
            contract=active_contract,
            register_lineage=register_lineage,
            model_version_id=target_model.model_version_id,
        )
        return self.stores.prediction_engine.predict(
            context=prepared.context,
            artifact=prepared.engineered_artifact,
            contract=active_contract,
            model=target_model,
        )
