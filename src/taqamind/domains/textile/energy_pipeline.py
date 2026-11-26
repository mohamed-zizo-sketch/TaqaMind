"""Textile V2 Energy pipeline wired directly to Shared CorePlatformServices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

from taqamind.core.contracts import DatasetContract, DatasetPurpose
from taqamind.core.feature_engineering import DerivedFeatureSpec, DeterministicFeatureEngineer
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
    ModelFeatureSpecification,
    ModelRegistryEntry,
    ModelScope,
    ModelTransitionAction,
    ModelTransitionRequest,
    ModelVersion,
    PipelineContext,
    PredictionBatchResult,
    PredictionModelIdentity,
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
    calculate_split_fingerprint,
    calculate_training_configuration_fingerprint,
)
from taqamind.core.pipeline import CorePlatformServices
from taqamind.core.registry import DomainRegistry
from taqamind.domains.textile.domain import TEXTILE_DOMAIN_ID, register_textile
from taqamind.domains.textile.knitting.contracts import KNITTING_PROCESS_ID
from taqamind.domains.textile.knitting.energy_contracts import (
    KNITTING_ENERGY_DATASET_ID,
    KNITTING_ENERGY_TASK_ID,
    build_knitting_energy_derived_specs,
)
from taqamind.domains.textile.weaving.contracts import WEAVING_PROCESS_ID
from taqamind.domains.textile.weaving.energy_contracts import (
    WEAVING_ENERGY_DATASET_ID,
    WEAVING_ENERGY_TASK_ID,
    build_weaving_energy_derived_specs,
)


@dataclass(frozen=True, slots=True)
class PreparedEnergyDataset:
    """Result of preparing a Weaving or Knitting Energy CSV through ENGINEERED."""

    context: PipelineContext
    artifact: DatasetArtifact


@dataclass(frozen=True, slots=True)
class EnergyTrainingOutcome:
    """Result of a controlled V2 Energy training + protected evaluation run."""

    training_run: TrainingRunResult
    manifest: SplitManifest
    evaluation: FinalEvaluationResult
    model_entry: ModelRegistryEntry
    activated: bool


class TextileEnergyPipelineService:
    """Pre-production electrical energy pipeline powered by CorePlatformServices."""

    def __init__(self, root_dir: Path) -> None:
        self._root = Path(root_dir)
        self.core = CorePlatformServices.create_local(self._root)
        self.domain_registry = DomainRegistry()
        register_textile(self.domain_registry)

    def _derived_specs_for(self, process_id: str) -> tuple[DerivedFeatureSpec, ...]:
        if process_id == WEAVING_PROCESS_ID:
            return build_weaving_energy_derived_specs()
        if process_id == KNITTING_PROCESS_ID:
            return build_knitting_energy_derived_specs()
        raise ValueError(f"Unsupported textile process '{process_id}'.")

    def _task_and_dataset_for(self, process_id: str) -> tuple[str, str]:
        if process_id == WEAVING_PROCESS_ID:
            return WEAVING_ENERGY_TASK_ID, WEAVING_ENERGY_DATASET_ID
        if process_id == KNITTING_PROCESS_ID:
            return KNITTING_ENERGY_TASK_ID, KNITTING_ENERGY_DATASET_ID
        raise ValueError(f"Unsupported textile process '{process_id}'.")

    def _default_split_config(self, process_id: str, seed: int = 42) -> SplitConfiguration:
        return SplitConfiguration(
            strategy=SplitStrategy.GROUP_TEMPORAL,
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            random_seed=seed,
            justification=f"Group-temporal split for {process_id} energy prediction.",
        )

    def default_training_config(self, *, random_seed: int = 42) -> TrainingConfiguration:
        return TrainingConfiguration(
            random_seed=random_seed,
            candidates=(
                CandidateConfiguration(
                    family="dummy_regressor",
                    role=CandidateRole.BASELINE,
                ),
                CandidateConfiguration(
                    family="historical_rate",
                    role=CandidateRole.BASELINE,
                ),
                CandidateConfiguration(
                    family="ridge",
                    role=CandidateRole.CHALLENGER,
                ),
                CandidateConfiguration(
                    family="hist_gradient_boosting",
                    role=CandidateRole.CHALLENGER,
                ),
            ),
            justification="Controlled V2 Textile Energy AutoML candidates.",
        )

    def ingest_and_prepare(
        self,
        *,
        factory_id: UUID,
        process_id: str,
        csv_path: Path,
        purpose: DatasetPurpose,
        contract: DatasetContract,
        model_version_id: UUID | None = None,
        explicit_source_units: dict[str, str] | None = None,
        created_at: datetime | None = None,
    ) -> PreparedEnergyDataset:
        task_id, dataset_id = self._task_and_dataset_for(process_id)
        rev_id = uuid4()
        ctx = PipelineContext(
            factory_id=factory_id,
            dataset_revision_id=rev_id,
            domain=TEXTILE_DOMAIN_ID,
            process=process_id,
            task_id=task_id,
            dataset_id=dataset_id,
            contract_version=contract.version,
            task_version="1.0.0",
            target_version=contract.target.version,
            purpose=purpose,
            model_version_id=model_version_id,
            created_at=created_at or datetime.now(UTC),
        )
        raw_bytes = Path(csv_path).read_bytes()
        ref = ArtifactReference(
            factory_id=factory_id,
            format=ArtifactFormat.CSV,
            content_sha256=calculate_file_sha256(csv_path),
            size_bytes=len(raw_bytes),
        )
        self.core.dataset_storage.put(
            factory_id=factory_id,
            reference=ref,
            stream=BytesIO(raw_bytes),
        )
        raw = DatasetArtifact(
            metadata=DatasetMetadata(
                revision_id=rev_id,
                root_revision_id=rev_id,
                factory_id=factory_id,
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
        self.core.dataset_registry.register(factory_id=factory_id, artifact=raw)

        ing = self.core.ingestion.ingest(context=ctx, artifact=raw, contract=contract)
        if not ing.is_successful or ing.output_artifact is None:
            raise ValueError(f"Ingestion failed: {ing.report}")
        self.core.dataset_registry.register(factory_id=factory_id, artifact=ing.output_artifact)

        if purpose == DatasetPurpose.PREDICTION:
            target_name = contract.target.feature.name.casefold()
            leaked = [
                c.name
                for c in ing.output_artifact.metadata.columns
                if c.name.casefold() == target_name
            ]
            if leaked:
                raise ValueError(f"Prediction CSV must not contain target column(s): {leaked}")

        self.core.profiler.profile(
            context=ctx,
            artifact=ing.output_artifact,
            contract=contract,
        )
        prop = self.core.mapper.propose_mapping(
            context=ctx,
            artifact=ing.output_artifact,
            contract=contract,
        )
        unit_overrides = explicit_source_units or {}
        entries = tuple(
            e.model_copy(
                update={
                    "source_unit": unit_overrides.get(e.source_column, e.canonical_unit)
                }
            )
            if e.canonical_unit is not None and e.source_unit is None
            else e
            for e in prop.entries
        )
        confirmed_prop = prop.model_copy(update={"entries": entries})
        mapped = self.core.mapper.apply_mapping(
            context=ctx,
            artifact=ing.output_artifact,
            contract=contract,
            mapping=ApprovedMapping(
                proposal=confirmed_prop,
                approved_by="textile_energy_v2",
                confirmed_source_columns=tuple(
                    e.source_column for e in entries if e.requires_confirmation
                ),
            ),
        )
        if not mapped.is_successful or mapped.output_artifact is None:
            raise ValueError(f"Mapping failed: {mapped.report}")
        self.core.dataset_registry.register(factory_id=factory_id, artifact=mapped.output_artifact)

        val = self.core.validator.validate(
            context=ctx,
            artifact=mapped.output_artifact,
            contract=contract,
        )
        if not val.is_successful or (val.report is not None and val.report.has_blocking_issues):
            raise ValueError(f"Semantic validation failed: {val.report}")

        cleaned = self.core.cleaner.clean(
            context=ctx,
            artifact=mapped.output_artifact,
            contract=contract,
        )
        if not cleaned.is_successful or cleaned.output_artifact is None:
            raise ValueError(f"Cleaning failed: {cleaned.report}")
        self.core.dataset_registry.register(factory_id=factory_id, artifact=cleaned.output_artifact)

        feature_engineer = DeterministicFeatureEngineer(
            self.core.dataset_storage,
            derived_specs=self._derived_specs_for(process_id),
        )
        engineered = feature_engineer.engineer(
            context=ctx,
            artifact=cleaned.output_artifact,
            contract=contract,
        )
        if not engineered.is_successful or engineered.output_artifact is None:
            raise ValueError(f"Feature engineering failed: {engineered.report}")
        self.core.dataset_registry.register(
            factory_id=factory_id,
            artifact=engineered.output_artifact,
        )

        rec_scope = RecordScope(
            factory_id=factory_id,
            domain=ctx.domain,
            process=ctx.process,
            dataset_id=ctx.dataset_id,
            identity_namespace="production_record",
        )
        self.core.membership_builder.build_and_register(
            scope=rec_scope,
            artifact=engineered.output_artifact,
            contract=contract,
        )
        return PreparedEnergyDataset(context=ctx, artifact=engineered.output_artifact)

    def train_and_evaluate(
        self,
        *,
        factory_id: UUID,
        process_id: str,
        prepared: PreparedEnergyDataset,
        contract: DatasetContract,
        training_config: TrainingConfiguration | None = None,
        policy: AcceptancePolicy | None = None,
        mode: TrainingMode = TrainingMode.INITIAL,
        previous_model_version_id: UUID | None = None,
        auto_activate: bool = True,
    ) -> EnergyTrainingOutcome:
        ctx = prepared.context
        artifact = prepared.artifact
        cfg = training_config or self.default_training_config()
        acc_policy = policy or AcceptancePolicy(
            minimum_improvement_pct=1.0,
            justification="Textile Energy V2 default policy.",
            created_at=ctx.created_at,
        )

        readiness = self.core.readiness_checker.assess(
            context=ctx,
            artifact=artifact,
            contract=contract,
        )
        readiness.require_split_eligibility()

        manifest = self.core.splitter.split(
            readiness=readiness,
            configuration=self._default_split_config(process_id, seed=cfg.random_seed),
        )
        self.core.split_registry.register_split(factory_id=factory_id, manifest=manifest)

        leakage = self.core.leakage_validator.validate(readiness=readiness, manifest=manifest)
        leakage.require_training_eligibility()
        self.core.split_registry.register_leakage_validation(
            factory_id=factory_id,
            validation=leakage,
        )
        auth = self.core.split_registry.issue_training_authorization(
            factory_id=factory_id,
            validation_id=leakage.validation_id,
        )

        intent = TrainingIntent(
            context=ctx,
            mode=mode,
            requested_by="textile_energy_v2",
            reason=f"{mode.value} training for {process_id} energy.",
            previous_model_version_id=previous_model_version_id,
        )
        self.core.training_registry.register_intent(factory_id=factory_id, intent=intent)

        rec_scope = RecordScope(
            factory_id=factory_id,
            domain=ctx.domain,
            process=ctx.process,
            dataset_id=ctx.dataset_id,
            identity_namespace="production_record",
        )
        membership = self.core.lineage_registry.get_membership(
            scope=rec_scope,
            dataset_revision_id=artifact.metadata.revision_id,
        )
        reuse_key = TrainingReuseKey.from_intent(
            intent=intent,
            dataset_content_sha256=artifact.reference.content_sha256,
            contract_sha256=calculate_bytes_sha256(contract.model_dump_json().encode("utf-8")),
            split_sha256=calculate_split_fingerprint(manifest, membership),
            training_configuration_sha256=calculate_training_configuration_fingerprint(
                cfg, acc_policy
            ),
            implementation_sha256=calculate_bytes_sha256(b"core-automl-v1"),
        )
        reused = self.core.training_registry.find_reusable_result(key=reuse_key)
        if reused is not None:
            run_result = reused
        else:
            run_result = self.core.trainer.train(
                authorization=auth,
                contract=contract,
                configuration=cfg,
                policy=acc_policy,
            )
            self.core.training_registry.register_result(
                factory_id=factory_id,
                key=reuse_key,
                result=run_result,
            )

        selection = FrozenModelSelection.from_training_run(
            training_run=run_result,
            policy=acc_policy,
        )
        evaluator = self.core.create_evaluator(contract)
        evaluation = evaluator.evaluate(selection=selection, manifest=manifest)

        challenger_res = run_result.best_challenger
        if challenger_res is None or challenger_res.model_reference is None:
            raise ValueError("No successful challenger model produced.")
        challenger_cfg = cfg.get_candidate(challenger_res.candidate_id)

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
        scope = ModelScope.from_identity(identity)
        existing_versions = self.core.model_registry.list_versions(scope=scope)
        version = ModelVersion(
            identity=identity,
            version_number=len(existing_versions) + 1,
            training_run_id=run_result.training_run_id,
            selection_id=selection.selection_id,
            split_id=manifest.split_id,
            training_dataset_revision_id=artifact.metadata.revision_id,
            contract_sha256=calculate_bytes_sha256(contract.model_dump_json().encode("utf-8")),
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
        self.core.model_registry.register_candidate(scope=scope, version=version)
        entry = self.core.model_registry.apply_evaluation(
            scope=scope,
            model_version_id=version.model_version_id,
            evaluation=evaluation,
        )

        activated = False
        if auto_activate and evaluation.decision == EvaluationDecision.ACCEPTED:
            current_active = self.core.model_registry.get_active(scope=scope)
            if current_active is None:
                entry = self.core.model_registry.transition(
                    request=ModelTransitionRequest(
                        scope=scope,
                        action=ModelTransitionAction.FIRST_ACTIVATION,
                        target_model_version_id=version.model_version_id,
                        expected_active_model_version_id=None,
                        requested_by="textile_energy_v2",
                        reason="Initial V2 energy model activation.",
                    )
                )
            else:
                entry = self.core.model_registry.transition(
                    request=ModelTransitionRequest(
                        scope=scope,
                        action=ModelTransitionAction.PROMOTION,
                        target_model_version_id=version.model_version_id,
                        expected_active_model_version_id=current_active.version.model_version_id,
                        comparison=PromotionComparison(
                            scope=scope,
                            active_model_version_id=current_active.version.model_version_id,
                            challenger_model_version_id=version.model_version_id,
                            evaluation_dataset_revision_id=artifact.metadata.revision_id,
                            evaluation_record_ids=manifest.test.record_ids,
                            metric=RegressionMetric.MAE,
                            active_mae=evaluation.baseline_mae,
                            challenger_mae=evaluation.selected_mae,
                            minimum_improvement_pct=acc_policy.minimum_improvement_pct,
                        ),
                        requested_by="textile_energy_v2",
                        reason="Promote retrained V2 energy model.",
                    )
                )
            activated = True

        return EnergyTrainingOutcome(
            training_run=run_result,
            manifest=manifest,
            evaluation=evaluation,
            model_entry=entry,
            activated=activated,
        )

    def predict(
        self,
        *,
        factory_id: UUID,
        process_id: str,
        csv_path: Path,
        contract: DatasetContract,
        explicit_source_units: dict[str, str] | None = None,
    ) -> PredictionBatchResult:
        task_id, _ = self._task_and_dataset_for(process_id)
        scope = ModelScope(
            factory_id=factory_id,
            domain=TEXTILE_DOMAIN_ID,
            process=process_id,
            task_id=task_id,
        )
        active = self.core.active_model_provider.get_active(scope=scope)
        if active is None:
            raise ValueError("No active energy model registered for this factory and process.")
        prepared = self.ingest_and_prepare(
            factory_id=factory_id,
            process_id=process_id,
            csv_path=csv_path,
            purpose=DatasetPurpose.PREDICTION,
            contract=contract,
            model_version_id=active.version.model_version_id,
            explicit_source_units=explicit_source_units,
        )
        return self.core.prediction_engine.predict(
            context=prepared.context,
            artifact=prepared.artifact,
            contract=contract,
            model=active.version.identity,
        )


__all__ = [
    "EnergyTrainingOutcome",
    "PreparedEnergyDataset",
    "TextileEnergyPipelineService",
]
