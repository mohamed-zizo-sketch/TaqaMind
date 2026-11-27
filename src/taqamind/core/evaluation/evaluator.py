"""Core P1.7 ProtectedFinalEvaluator implementation."""

from __future__ import annotations

import pickle
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from taqamind.core.automl._metrics import compute_regression_metrics
from taqamind.core.automl.data_access import (
    convert_dataframe_by_contract,
    read_partition_dataframe,
)
from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.ingestion.resource_policy import CsvResourcePolicy, resolve_internal_csv_policy
from taqamind.core.interfaces.data_access import ProtectedTestDataProvider
from taqamind.core.interfaces.evaluation import (
    FinalEvaluationResult,
    FrozenModelSelection,
    TestMetricResult,
)
from taqamind.core.interfaces.lineage import RecordScope
from taqamind.core.interfaces.ml import FinalModelEvaluator
from taqamind.core.interfaces.results import StageTiming
from taqamind.core.interfaces.splitting import SplitManifest
from taqamind.core.interfaces.storage import ModelArtifactStorage
from taqamind.core.interfaces.training_registry import (
    EvaluationRegistry,
    EvaluationReservation,
)
from taqamind.core.preprocessing.type_conversion import ContractTypeConverter


class ProtectedFinalEvaluator(FinalModelEvaluator):
    """Evaluates frozen baseline & selected challenger once on the protected test partition."""

    def __init__(
        self,
        *,
        contract: DatasetContract,
        evaluation_registry: EvaluationRegistry,
        test_data_provider: ProtectedTestDataProvider,
        model_storage: ModelArtifactStorage,
        type_converter: ContractTypeConverter | None = None,
        resource_policy: CsvResourcePolicy | None = None,
    ) -> None:
        self._contract = contract
        self._evaluation_registry = evaluation_registry
        self._test_data_provider = test_data_provider
        self._model_storage = model_storage
        self._converter = type_converter or ContractTypeConverter()
        self._resource_policy = resolve_internal_csv_policy(resource_policy)

    def _score_candidate(
        self,
        *,
        factory_id,
        candidate_result,
        test_df: pd.DataFrame,
        y_test: np.ndarray,
        test_record_ids: tuple,
        metrics: tuple,
    ) -> TestMetricResult:
        ref = candidate_result.model_reference
        if ref is None:
            raise ValueError("Candidate model reference is required for evaluation.")
        with self._model_storage.open_verified(
            factory_id=factory_id,
            reference=ref,
        ) as stream:
            bundle = pickle.loads(stream.read())

        preprocessor = bundle["preprocessor"]
        estimator = bundle["estimator"]
        X_test = preprocessor.transform(test_df)
        y_pred = np.asarray(estimator.predict(X_test), dtype=float)
        if bundle.get("must_be_non_negative", False):
            y_pred = np.maximum(0.0, y_pred)

        computed = compute_regression_metrics(
            y_true=y_test,
            y_pred=y_pred,
            metrics=metrics,
        )
        return TestMetricResult(
            candidate_id=candidate_result.candidate_id,
            model_artifact_id=ref.artifact_id,
            record_ids=test_record_ids,
            metrics=computed,
        )

    def evaluate(
        self,
        *,
        selection: FrozenModelSelection,
        manifest: SplitManifest,
    ) -> FinalEvaluationResult:
        started_at = datetime.now(UTC)
        context = manifest.context
        factory_id = context.factory_id
        scope = RecordScope(
            factory_id=context.factory_id,
            domain=context.domain,
            process=context.process,
            dataset_id=context.dataset_id,
            identity_namespace="production_record",
        )

        self._evaluation_registry.register_selection(
            factory_id=factory_id,
            selection=selection,
        )
        reservation = EvaluationReservation.from_selection(
            scope=scope,
            selection=selection,
            manifest=manifest,
        )
        reserved = self._evaluation_registry.reserve_test_access(
            factory_id=factory_id,
            reservation=reservation,
        )
        access = self._test_data_provider.prepare(
            reservation=reserved,
            selection=selection,
            contract=self._contract,
        )

        with self._test_data_provider.open_test(access=access) as test_stream:
            test_raw = read_partition_dataframe(test_stream, resource_policy=self._resource_policy)

        test_df = convert_dataframe_by_contract(
            test_raw,
            contract=self._contract,
            converter=self._converter,
        )
        target_col = self._contract.target.feature.name
        y_test = pd.to_numeric(test_df[target_col], errors="raise").to_numpy(dtype=float)

        run = selection.training_run
        requested_metrics = (run.configuration.primary_metric, *run.configuration.secondary_metrics)
        results_by_id = {r.candidate_id: r for r in run.successful_results}

        baseline_eval = self._score_candidate(
            factory_id=factory_id,
            candidate_result=results_by_id[selection.baseline_candidate_id],
            test_df=test_df,
            y_test=y_test,
            test_record_ids=reserved.test_record_ids,
            metrics=requested_metrics,
        )
        selected_eval = self._score_candidate(
            factory_id=factory_id,
            candidate_result=results_by_id[selection.selected_candidate_id],
            test_df=test_df,
            y_test=y_test,
            test_record_ids=reserved.test_record_ids,
            metrics=requested_metrics,
        )
        finished_at = datetime.now(UTC)

        evaluation = FinalEvaluationResult(
            selection=selection,
            split_manifest=manifest,
            baseline_result=baseline_eval,
            selected_result=selected_eval,
            timing=StageTiming(started_at=started_at, finished_at=finished_at),
        )
        self._evaluation_registry.register_evaluation(
            factory_id=factory_id,
            reservation_id=reserved.reservation_id,
            evaluation=evaluation,
        )
        return evaluation


__all__ = ["ProtectedFinalEvaluator"]
