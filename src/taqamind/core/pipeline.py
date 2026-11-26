"""Unified Core ML pipeline orchestrator wiring P1.1-P1.8 components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from taqamind.core.automl import (
    ControlledAutoMLTrainer,
    LocalDevelopmentDataProvider,
    LocalEvaluationRegistry,
    LocalProtectedTestDataProvider,
    LocalTrainingRegistry,
)
from taqamind.core.evaluation import ProtectedFinalEvaluator
from taqamind.core.feature_engineering import DeterministicFeatureEngineer
from taqamind.core.ingestion import CsvDataIngestion
from taqamind.core.lineage import LocalRecordLineageRegistry, RecordMembershipBuilder
from taqamind.core.mapping import DeterministicColumnMapper
from taqamind.core.prediction import CorePredictionEngine, LocalActiveModelProvider
from taqamind.core.preprocessing import ContractDataCleaner
from taqamind.core.profiling import PandasDataProfiler
from taqamind.core.readiness import CoreTrainingReadinessChecker
from taqamind.core.registry import (
    LocalDatasetArtifactStorage,
    LocalDatasetRegistry,
    LocalModelArtifactStorage,
    LocalModelRegistry,
)
from taqamind.core.splitting import (
    CoreDatasetSplitter,
    CoreLeakageValidator,
    LocalRecordUsageRegistry,
    LocalSplitRegistry,
)
from taqamind.core.validation import ContractSemanticValidator


@dataclass(frozen=True, slots=True)
class CorePlatformServices:
    """Injectable bundle of all Shared Core local storage, registries, and ML engines."""

    root_dir: Path
    dataset_storage: LocalDatasetArtifactStorage
    model_storage: LocalModelArtifactStorage
    dataset_registry: LocalDatasetRegistry
    lineage_registry: LocalRecordLineageRegistry
    split_registry: LocalSplitRegistry
    usage_registry: LocalRecordUsageRegistry
    training_registry: LocalTrainingRegistry
    evaluation_registry: LocalEvaluationRegistry
    model_registry: LocalModelRegistry
    ingestion: CsvDataIngestion
    profiler: PandasDataProfiler
    mapper: DeterministicColumnMapper
    validator: ContractSemanticValidator
    cleaner: ContractDataCleaner
    feature_engineer: DeterministicFeatureEngineer
    membership_builder: RecordMembershipBuilder
    readiness_checker: CoreTrainingReadinessChecker
    splitter: CoreDatasetSplitter
    leakage_validator: CoreLeakageValidator
    development_data_provider: LocalDevelopmentDataProvider
    protected_test_data_provider: LocalProtectedTestDataProvider
    trainer: ControlledAutoMLTrainer
    active_model_provider: LocalActiveModelProvider
    prediction_engine: CorePredictionEngine

    @classmethod
    def create_local(cls, root_dir: Path) -> CorePlatformServices:
        """Construct a complete factory-isolated local Core platform stack under root_dir."""
        root = Path(root_dir)
        dataset_storage = LocalDatasetArtifactStorage(root / "dataset_storage")
        model_storage = LocalModelArtifactStorage(root / "model_storage")
        dataset_registry = LocalDatasetRegistry(root / "dataset_registry", dataset_storage)
        lineage_registry = LocalRecordLineageRegistry(
            root / "lineage_registry",
            dataset_registry=dataset_registry,
        )
        split_registry = LocalSplitRegistry(root / "split_registry")
        usage_registry = LocalRecordUsageRegistry(root / "usage_registry")
        training_registry = LocalTrainingRegistry(root / "training_registry")
        evaluation_registry = LocalEvaluationRegistry(
            root_dir=root / "evaluation_registry",
            split_registry=split_registry,
            usage_registry=usage_registry,
            lineage_registry=lineage_registry,
        )
        model_registry = LocalModelRegistry(
            root_dir=root / "model_registry",
            model_storage=model_storage,
        )
        development_data_provider = LocalDevelopmentDataProvider(
            partition_root=root / "dev_partitions",
            storage=dataset_storage,
            dataset_registry=dataset_registry,
            lineage_registry=lineage_registry,
            split_registry=split_registry,
            usage_registry=usage_registry,
        )
        protected_test_data_provider = LocalProtectedTestDataProvider(
            partition_root=root / "test_partitions",
            storage=dataset_storage,
            dataset_registry=dataset_registry,
            lineage_registry=lineage_registry,
            split_registry=split_registry,
            evaluation_registry=evaluation_registry,
        )
        active_model_provider = LocalActiveModelProvider(
            model_registry=model_registry,
            model_storage=model_storage,
        )
        return cls(
            root_dir=root,
            dataset_storage=dataset_storage,
            model_storage=model_storage,
            dataset_registry=dataset_registry,
            lineage_registry=lineage_registry,
            split_registry=split_registry,
            usage_registry=usage_registry,
            training_registry=training_registry,
            evaluation_registry=evaluation_registry,
            model_registry=model_registry,
            ingestion=CsvDataIngestion(dataset_storage),
            profiler=PandasDataProfiler(dataset_storage),
            mapper=DeterministicColumnMapper(dataset_storage),
            validator=ContractSemanticValidator(dataset_storage),
            cleaner=ContractDataCleaner(dataset_storage),
            feature_engineer=DeterministicFeatureEngineer(dataset_storage),
            membership_builder=RecordMembershipBuilder(dataset_storage, lineage_registry),
            readiness_checker=CoreTrainingReadinessChecker(dataset_storage, lineage_registry),
            splitter=CoreDatasetSplitter(dataset_storage, lineage_registry),
            leakage_validator=CoreLeakageValidator(
                dataset_storage,
                lineage_registry,
                usage_registry,
            ),
            development_data_provider=development_data_provider,
            protected_test_data_provider=protected_test_data_provider,
            trainer=ControlledAutoMLTrainer(
                development_data_provider=development_data_provider,
                model_storage=model_storage,
            ),
            active_model_provider=active_model_provider,
            prediction_engine=CorePredictionEngine(
                active_model_provider=active_model_provider,
                dataset_storage=dataset_storage,
                lineage_registry=lineage_registry,
            ),
        )

    def create_evaluator(self, contract) -> ProtectedFinalEvaluator:
        """Create a contract-bound ProtectedFinalEvaluator."""
        return ProtectedFinalEvaluator(
            contract=contract,
            evaluation_registry=self.evaluation_registry,
            test_data_provider=self.protected_test_data_provider,
            model_storage=self.model_storage,
        )


__all__ = ["CorePlatformServices"]
