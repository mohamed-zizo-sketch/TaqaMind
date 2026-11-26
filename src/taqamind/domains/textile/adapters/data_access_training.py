"""Data access providers, training/evaluation registries, AutoML trainer, and evaluator adapters."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO
from uuid import UUID, uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from taqamind.core.contracts import DatasetContract, DataType
from taqamind.core.interfaces import (
    AcceptancePolicy,
    ArtifactFormat,
    ArtifactReference,
    CandidateConfiguration,
    CandidateModelReference,
    CandidateResult,
    CandidateStatus,
    DataAccessDeniedError,
    DatasetRegistry,
    DevelopmentDataAccess,
    DevelopmentDataProvider,
    EvaluationRegistry,
    EvaluationReservation,
    ExecutionError,
    FinalEvaluationResult,
    FinalModelEvaluator,
    FrozenModelSelection,
    ModelArtifactStorage,
    ModelFamily,
    ModelTrainer,
    PartitionDataReference,
    ProtectedTestDataAccess,
    ProtectedTestDataProvider,
    RecordLineageRegistry,
    RecordScope,
    RecordUsage,
    RecordUsageRegistry,
    RecordUsageRole,
    RegistryConflictError,
    RegistryNotFoundError,
    RegistryScopeError,
    RegressionMetric,
    SplitManifest,
    SplitPartition,
    SplitRegistry,
    StageTiming,
    TestMetricResult,
    TrainingAuthorization,
    TrainingConfiguration,
    TrainingIntent,
    TrainingRegistry,
    TrainingReuseKey,
    TrainingRunResult,
    ValidationMetric,
    calculate_bytes_sha256,
    find_previously_exposed_test_records,
    validate_training_request,
)
from taqamind.core.preprocessing import TabularPreprocessor
from taqamind.core.registry import LocalDatasetArtifactStorage
from taqamind.domains.textile.adapters._store import (
    atomic_write_text,
    dump_json_model,
    iter_json_models,
    load_json_model,
    safe_resolve,
)


def _scope_for_context(context: Any, identity_namespace: str = "production_record") -> RecordScope:
    return RecordScope(
        factory_id=context.factory_id,
        domain=context.domain,
        process=context.process,
        dataset_id=context.dataset_id,
        identity_namespace=identity_namespace,
    )


class _PartitionStorageMixin:
    """Shared partition materialization & verified opening helper."""

    def __init__(
        self,
        *,
        registry_dir: str | Path,
        dataset_storage: LocalDatasetArtifactStorage,
        dataset_registry: DatasetRegistry,
        lineage_registry: RecordLineageRegistry,
        usage_registry: RecordUsageRegistry,
        identity_namespace: str = "production_record",
    ) -> None:
        self._root = Path(registry_dir).resolve()
        self._storage = dataset_storage
        self._datasets = dataset_registry
        self._lineage = lineage_registry
        self._usage = usage_registry
        self._identity_namespace = identity_namespace

    def _materialize_partition(
        self,
        *,
        factory_id: UUID,
        dataset_revision_id: UUID,
        split_id: UUID,
        scope: RecordScope,
        partition: SplitPartition,
        contract: DatasetContract,
    ) -> PartitionDataReference:
        artifact = self._datasets.get(
            factory_id=factory_id, dataset_revision_id=dataset_revision_id
        )
        membership = self._lineage.get_membership(
            scope=scope, dataset_revision_id=dataset_revision_id
        )
        membership.validate_artifact(artifact)
        row_by_id = {m.record_id: m.row_position for m in membership.memberships}

        dtype_hints = {
            f.name: "string"
            for f in contract.all_columns
            if f.data_type in {DataType.STRING, DataType.CATEGORY, DataType.DATE, DataType.DATETIME}
        }
        with self._storage.open_verified(
            factory_id=factory_id, reference=artifact.reference
        ) as stream:
            df = pd.read_csv(stream, keep_default_na=True, dtype=dtype_hints)

        learning_cols = tuple(contract.predictor_names) + (contract.target.feature.name,)
        indices = [row_by_id[rid] for rid in partition.record_ids]
        part_df = df.iloc[indices][list(learning_cols)].reset_index(drop=True)

        csv_bytes = part_df.to_csv(index=False).encode("utf-8")
        digest = calculate_bytes_sha256(csv_bytes)
        part_id = uuid4()
        ref = ArtifactReference(
            artifact_id=part_id,
            factory_id=factory_id,
            format=ArtifactFormat.CSV,
            content_sha256=digest,
            size_bytes=len(csv_bytes),
        )
        self._storage.put(factory_id=factory_id, reference=ref, stream=BytesIO(csv_bytes))

        part_ref = PartitionDataReference(
            partition_artifact_id=part_id,
            factory_id=factory_id,
            dataset_revision_id=dataset_revision_id,
            split_id=split_id,
            role=partition.role,
            record_ids=partition.record_ids,
            columns=learning_cols,
            content_sha256=digest,
            size_bytes=len(csv_bytes),
        )
        meta_path = safe_resolve(self._root, "partitions", str(factory_id), f"{part_id}.json")
        atomic_write_text(
            meta_path,
            json.dumps(
                {
                    "partition": part_ref.model_dump(mode="json"),
                    "artifact": ref.model_dump(mode="json"),
                },
                indent=2,
            ),
        )
        return part_ref

    @contextmanager
    def _open_partition(
        self,
        *,
        factory_id: UUID,
        partition_ref: PartitionDataReference,
        scope: RecordScope,
        execution_id: UUID,
        usage_role: RecordUsageRole,
    ) -> Iterator[BinaryIO]:
        membership = self._lineage.get_membership(
            scope=scope, dataset_revision_id=partition_ref.dataset_revision_id
        )
        ver_by_id = {m.record_id: m.record_version_id for m in membership.memberships}
        usages = tuple(
            RecordUsage(
                scope=scope,
                record_id=rid,
                record_version_id=ver_by_id[rid],
                execution_id=execution_id,
                split_id=partition_ref.split_id,
                role=usage_role,
            )
            for rid in partition_ref.record_ids
        )
        self._usage.record_usage(scope=scope, usages=usages)

        meta_path = safe_resolve(
            self._root, "partitions", str(factory_id), f"{partition_ref.partition_artifact_id}.json"
        )
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        art_ref = ArtifactReference.model_validate(payload["artifact"])
        with self._storage.open_verified(factory_id=factory_id, reference=art_ref) as stream:
            yield stream


class LocalDevelopmentDataProvider(_PartitionStorageMixin, DevelopmentDataProvider):
    """Materialize train/validation partitions only, recording ML exposure before access.

    Migration path: Replace with Shared Core P1.7 `DevelopmentDataProvider`.
    """

    def __init__(
        self,
        *,
        registry_dir: str | Path,
        dataset_storage: LocalDatasetArtifactStorage,
        dataset_registry: DatasetRegistry,
        lineage_registry: RecordLineageRegistry,
        split_registry: SplitRegistry,
        usage_registry: RecordUsageRegistry,
        identity_namespace: str = "production_record",
    ) -> None:
        super().__init__(
            registry_dir=registry_dir,
            dataset_storage=dataset_storage,
            dataset_registry=dataset_registry,
            lineage_registry=lineage_registry,
            usage_registry=usage_registry,
            identity_namespace=identity_namespace,
        )
        self._splits = split_registry

    def prepare(
        self, *, authorization: TrainingAuthorization, contract: DatasetContract
    ) -> DevelopmentDataAccess:
        dev = authorization.development_split
        context = dev.context
        self._splits.verify_training_authorization(
            factory_id=context.factory_id, authorization=authorization
        )
        scope = _scope_for_context(context, self._identity_namespace)
        train_ref = self._materialize_partition(
            factory_id=context.factory_id,
            dataset_revision_id=dev.dataset_revision_id,
            split_id=dev.split_id,
            scope=scope,
            partition=dev.train,
            contract=contract,
        )
        val_ref = self._materialize_partition(
            factory_id=context.factory_id,
            dataset_revision_id=dev.dataset_revision_id,
            split_id=dev.split_id,
            scope=scope,
            partition=dev.validation,
            contract=contract,
        )
        access = DevelopmentDataAccess(
            authorization=authorization, train=train_ref, validation=val_ref
        )
        access.validate_contract(contract)
        return access

    @contextmanager
    def open_train(self, *, access: DevelopmentDataAccess) -> Iterator[BinaryIO]:
        dev = access.authorization.development_split
        scope = _scope_for_context(dev.context, self._identity_namespace)
        with self._open_partition(
            factory_id=dev.context.factory_id,
            partition_ref=access.train,
            scope=scope,
            execution_id=dev.context.execution_id,
            usage_role=RecordUsageRole.TRAIN,
        ) as stream:
            yield stream

    @contextmanager
    def open_validation(self, *, access: DevelopmentDataAccess) -> Iterator[BinaryIO]:
        dev = access.authorization.development_split
        scope = _scope_for_context(dev.context, self._identity_namespace)
        with self._open_partition(
            factory_id=dev.context.factory_id,
            partition_ref=access.validation,
            scope=scope,
            execution_id=dev.context.execution_id,
            usage_role=RecordUsageRole.VALIDATION,
        ) as stream:
            yield stream


class LocalProtectedTestDataProvider(_PartitionStorageMixin, ProtectedTestDataProvider):
    """Materialize protected test partition only for registered reservations.

    Migration path: Replace with Shared Core P1.7 `ProtectedTestDataProvider`.
    """

    def __init__(
        self,
        *,
        registry_dir: str | Path,
        dataset_storage: LocalDatasetArtifactStorage,
        dataset_registry: DatasetRegistry,
        lineage_registry: RecordLineageRegistry,
        split_registry: SplitRegistry,
        usage_registry: RecordUsageRegistry,
        evaluation_registry: EvaluationRegistry,
        identity_namespace: str = "production_record",
    ) -> None:
        super().__init__(
            registry_dir=registry_dir,
            dataset_storage=dataset_storage,
            dataset_registry=dataset_registry,
            lineage_registry=lineage_registry,
            usage_registry=usage_registry,
            identity_namespace=identity_namespace,
        )
        self._splits = split_registry
        self._evals = evaluation_registry

    def prepare(
        self,
        *,
        reservation: EvaluationReservation,
        selection: FrozenModelSelection,
        contract: DatasetContract,
    ) -> ProtectedTestDataAccess:
        dev = selection.training_run.development_split
        context = dev.context
        manifest = self._splits.get_split(factory_id=context.factory_id, split_id=dev.split_id)
        if set(reservation.test_record_ids) != set(manifest.test.record_ids):
            raise DataAccessDeniedError("Reservation test record IDs do not match split manifest.")
        test_ref = self._materialize_partition(
            factory_id=context.factory_id,
            dataset_revision_id=manifest.dataset_revision_id,
            split_id=manifest.split_id,
            scope=reservation.scope,
            partition=manifest.test,
            contract=contract,
        )
        access = ProtectedTestDataAccess(
            reservation=reservation, selection=selection, test=test_ref
        )
        access.validate_contract(contract)
        return access

    @contextmanager
    def open_test(self, *, access: ProtectedTestDataAccess) -> Iterator[BinaryIO]:
        context = access.selection.training_run.development_split.context
        self._evals.mark_test_exposed(
            factory_id=context.factory_id,
            reservation_id=access.reservation.reservation_id,
        )
        with self._open_partition(
            factory_id=context.factory_id,
            partition_ref=access.test,
            scope=access.reservation.scope,
            execution_id=context.execution_id,
            usage_role=RecordUsageRole.PROTECTED_TEST,
        ) as stream:
            yield stream


class LocalTrainingRegistry(TrainingRegistry):
    """Persist explicit training intents and reusable completed runs.

    Migration path: Replace with Shared Core P1.7 `LocalTrainingRegistry`.
    """

    def __init__(self, registry_dir: str | Path) -> None:
        self._root = Path(registry_dir).resolve()

    def register_intent(self, *, factory_id: UUID, intent: TrainingIntent) -> None:
        if intent.context.factory_id != factory_id:
            raise RegistryScopeError("TrainingIntent factory mismatch.")
        path = safe_resolve(
            self._root, "training_intents", str(factory_id), f"{intent.context.execution_id}.json"
        )
        if path.is_file():
            existing = load_json_model(path, TrainingIntent)
            if existing == intent:
                return
            raise RegistryConflictError("Conflicting training intent for execution_id.")
        atomic_write_text(path, dump_json_model(intent))

    def find_reusable_result(self, *, key: TrainingReuseKey) -> TrainingRunResult | None:
        index_path = safe_resolve(
            self._root, "training_reuse", str(key.factory_id), f"{key.fingerprint}.json"
        )
        if not index_path.is_file():
            return None
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        run_id = UUID(payload["training_run_id"])
        return self.get_result(factory_id=key.factory_id, training_run_id=run_id)

    def register_result(
        self, *, factory_id: UUID, key: TrainingReuseKey, result: TrainingRunResult
    ) -> None:
        if (
            key.factory_id != factory_id
            or result.development_split.context.factory_id != factory_id
        ):
            raise RegistryScopeError("Training result factory mismatch.")
        res_path = safe_resolve(
            self._root, "training_runs", str(factory_id), f"{result.training_run_id}.json"
        )
        if res_path.is_file():
            existing = load_json_model(res_path, TrainingRunResult)
            if existing != result:
                raise RegistryConflictError("Conflicting TrainingRunResult for training_run_id.")
        else:
            atomic_write_text(res_path, dump_json_model(result))

        reuse_path = safe_resolve(
            self._root, "training_reuse", str(factory_id), f"{key.fingerprint}.json"
        )
        if not reuse_path.is_file():
            atomic_write_text(
                reuse_path,
                json.dumps(
                    {
                        "fingerprint": key.fingerprint,
                        "training_run_id": str(result.training_run_id),
                        "key": key.model_dump(mode="json"),
                    },
                    indent=2,
                ),
            )

    def get_result(self, *, factory_id: UUID, training_run_id: UUID) -> TrainingRunResult:
        path = safe_resolve(
            self._root, "training_runs", str(factory_id), f"{training_run_id}.json"
        )
        if not path.is_file():
            raise RegistryNotFoundError(f"Training run '{training_run_id}' not found.")
        return load_json_model(path, TrainingRunResult)


class LocalEvaluationRegistry(EvaluationRegistry):
    """Control test reservations, exposure records, and immutable FinalEvaluationResult.

    Migration path: Replace with Shared Core P1.7 `LocalEvaluationRegistry`.
    """

    def __init__(
        self,
        registry_dir: str | Path,
        *,
        split_registry: SplitRegistry,
        usage_registry: RecordUsageRegistry,
    ) -> None:
        self._root = Path(registry_dir).resolve()
        self._splits = split_registry
        self._usage = usage_registry

    def register_selection(
        self, *, factory_id: UUID, selection: FrozenModelSelection
    ) -> None:
        if selection.training_run.development_split.context.factory_id != factory_id:
            raise RegistryScopeError("Selection factory mismatch.")
        path = safe_resolve(
            self._root, "selections", str(factory_id), f"{selection.selection_id}.json"
        )
        if path.is_file():
            existing = load_json_model(path, FrozenModelSelection)
            if existing == selection:
                return
            raise RegistryConflictError("Conflicting selection for selection_id.")
        atomic_write_text(path, dump_json_model(selection))

    def reserve_test_access(
        self, *, factory_id: UUID, reservation: EvaluationReservation
    ) -> EvaluationReservation:
        if reservation.scope.factory_id != factory_id:
            raise RegistryScopeError("Reservation factory mismatch.")
        manifest = self._splits.get_split(factory_id=factory_id, split_id=reservation.split_id)
        history = self._usage.get_history(
            scope=reservation.scope, record_ids=reservation.test_record_ids
        )
        exposed = find_previously_exposed_test_records(
            manifest=manifest, scope=reservation.scope, history=history
        )
        if exposed:
            raise RegistryConflictError(
                "Protected test records have already been exposed in historical runs."
            )

        res_dir = safe_resolve(self._root, "reservations", str(factory_id))
        existing_reservations = iter_json_models(res_dir, EvaluationReservation)
        requested_test = set(reservation.test_record_ids)
        for existing in existing_reservations:
            if existing.reservation_id == reservation.reservation_id:
                if existing == reservation:
                    return existing
                raise RegistryConflictError("Conflicting reservation metadata.")
            if set(existing.test_record_ids) & requested_test:
                raise RegistryConflictError(
                    "Competing reservation already covers one or more protected test records."
                )

        path = safe_resolve(res_dir, f"{reservation.reservation_id}.json")
        atomic_write_text(path, dump_json_model(reservation))
        return reservation

    def mark_test_exposed(self, *, factory_id: UUID, reservation_id: UUID) -> None:
        res_path = safe_resolve(
            self._root, "reservations", str(factory_id), f"{reservation_id}.json"
        )
        if not res_path.is_file():
            raise RegistryNotFoundError(f"Reservation '{reservation_id}' not found.")
        exp_path = safe_resolve(
            self._root, "exposed_reservations", str(factory_id), f"{reservation_id}.json"
        )
        if not exp_path.is_file():
            atomic_write_text(
                exp_path,
                json.dumps(
                    {
                        "reservation_id": str(reservation_id),
                        "exposed_at": datetime.now(UTC).isoformat(),
                    },
                    indent=2,
                ),
            )

    def register_evaluation(
        self,
        *,
        factory_id: UUID,
        reservation_id: UUID,
        evaluation: FinalEvaluationResult,
    ) -> None:
        if evaluation.split_manifest.context.factory_id != factory_id:
            raise RegistryScopeError("Evaluation factory mismatch.")
        res_path = safe_resolve(
            self._root, "reservations", str(factory_id), f"{reservation_id}.json"
        )
        if not res_path.is_file():
            raise RegistryNotFoundError("Evaluation references unknown reservation.")
        ev_path = safe_resolve(
            self._root, "evaluations", str(factory_id), f"{evaluation.evaluation_id}.json"
        )
        if ev_path.is_file():
            existing = load_json_model(ev_path, FinalEvaluationResult)
            if existing == evaluation:
                return
            raise RegistryConflictError("Conflicting evaluation result.")
        atomic_write_text(ev_path, dump_json_model(evaluation))

    def get_evaluation(self, *, factory_id: UUID, evaluation_id: UUID) -> FinalEvaluationResult:
        path = safe_resolve(
            self._root, "evaluations", str(factory_id), f"{evaluation_id}.json"
        )
        if not path.is_file():
            raise RegistryNotFoundError(f"Evaluation '{evaluation_id}' not found.")
        return load_json_model(path, FinalEvaluationResult)


class _HistoricalRateEstimator(BaseEstimator, RegressorMixin):
    """Historical mean target/denominator rate baseline estimator."""

    def __init__(self, denominator_idx: int = 0) -> None:
        self.denominator_idx = denominator_idx
        self.mean_rate_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> _HistoricalRateEstimator:
        denom = np.asarray(X[:, self.denominator_idx], dtype=float)
        valid = np.isfinite(denom) & (denom > 0)
        if not bool(valid.any()):
            self.mean_rate_ = float(np.mean(y))
        else:
            self.mean_rate_ = float(np.sum(y[valid]) / np.sum(denom[valid]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        denom = np.asarray(X[:, self.denominator_idx], dtype=float)
        denom = np.where(np.isfinite(denom) & (denom > 0), denom, 0.0)
        return np.maximum(0.0, denom * self.mean_rate_)


class _TwoStageHurdleEstimator(BaseEstimator, RegressorMixin):
    """Two-stage hurdle regressor: P(y > 0 | X) * E[y | y > 0, X]."""

    def __init__(self, random_state: int = 42, max_iter: int = 100) -> None:
        self.random_state = random_state
        self.max_iter = max_iter
        self.classifier_: HistGradientBoostingRegressor | Any = None
        self.regressor_: HistGradientBoostingRegressor | None = None
        self.constant_positive_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> _TwoStageHurdleEstimator:
        from sklearn.linear_model import LogisticRegression

        y_arr = np.asarray(y, dtype=float)
        is_pos = (y_arr > 0).astype(int)
        if len(np.unique(is_pos)) < 2:
            self.classifier_ = None
            self.constant_prob_ = float(is_pos[0]) if len(is_pos) else 0.0
        else:
            clf = LogisticRegression(max_iter=self.max_iter, random_state=self.random_state)
            clf.fit(X, is_pos)
            self.classifier_ = clf

        pos_mask = y_arr > 0
        if int(pos_mask.sum()) >= 5:
            reg = HistGradientBoostingRegressor(
                random_state=self.random_state, max_iter=self.max_iter
            )
            reg.fit(X[pos_mask], y_arr[pos_mask])
            self.regressor_ = reg
        elif bool(pos_mask.any()):
            self.regressor_ = None
            self.constant_positive_ = float(np.mean(y_arr[pos_mask]))
        else:
            self.regressor_ = None
            self.constant_positive_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classifier_ is None:
            prob = np.full(len(X), getattr(self, "constant_prob_", 0.0), dtype=float)
        else:
            prob = self.classifier_.predict_proba(X)[:, 1]

        if self.regressor_ is None:
            mag = np.full(len(X), self.constant_positive_, dtype=float)
        else:
            mag = np.maximum(0.0, self.regressor_.predict(X))
        return np.maximum(0.0, prob * mag)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: tuple[RegressionMetric, ...],
) -> tuple[ValidationMetric, ...]:
    """Compute MAE, RMSE, WAPE, and R2 with explicit undefined guards."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    out: list[ValidationMetric] = []
    for metric in metrics:
        if metric == RegressionMetric.MAE:
            out.append(
                ValidationMetric(metric=metric, value=float(mean_absolute_error(y_t, y_p)))
            )
        elif metric == RegressionMetric.RMSE:
            out.append(
                ValidationMetric(metric=metric, value=float(root_mean_squared_error(y_t, y_p)))
            )
        elif metric == RegressionMetric.WAPE:
            denom = float(np.sum(np.abs(y_t)))
            if denom == 0.0:
                out.append(
                    ValidationMetric(
                        metric=metric,
                        value=None,
                        undefined_reason="Sum of |y_true| is zero; WAPE is undefined.",
                    )
                )
            else:
                wape = float(np.sum(np.abs(y_t - y_p)) / denom)
                out.append(ValidationMetric(metric=metric, value=wape))
        elif metric == RegressionMetric.R2:
            if len(y_t) < 2 or float(np.var(y_t)) == 0.0:
                out.append(
                    ValidationMetric(
                        metric=metric,
                        value=None,
                        undefined_reason="Zero variance or <2 samples; R2 is undefined.",
                    )
                )
            else:
                r2_val = min(1.0, float(r2_score(y_t, y_p)))
                out.append(ValidationMetric(metric=metric, value=r2_val))
    return tuple(out)


class TextileControlledAutoMLTrainer(ModelTrainer):
    """Train approved baseline + challenger regression models strictly on development data.

    Migration path: Replace with Shared Core P1.7 `ControlledAutoMLTrainer`.
    """

    _ALLOWLISTED_PARAMS: dict[ModelFamily, set[str]] = {
        ModelFamily.DUMMY_REGRESSOR: {"strategy"},
        ModelFamily.HISTORICAL_RATE: set(),
        ModelFamily.RIDGE: {"alpha"},
        ModelFamily.ELASTIC_NET: {"alpha", "l1_ratio", "max_iter"},
        ModelFamily.RANDOM_FOREST: {"n_estimators", "max_depth", "min_samples_leaf"},
        ModelFamily.HIST_GRADIENT_BOOSTING: {"max_iter", "max_depth", "learning_rate"},
        ModelFamily.XGBOOST: {"n_estimators", "max_depth", "learning_rate", "subsample"},
        ModelFamily.HURDLE_REGRESSION: {"max_iter"},
    }

    def __init__(
        self,
        data_provider: DevelopmentDataProvider,
        model_storage: ModelArtifactStorage,
    ) -> None:
        self._data_provider = data_provider
        self._model_storage = model_storage

    def train(
        self,
        *,
        authorization: TrainingAuthorization,
        contract: DatasetContract,
        configuration: TrainingConfiguration,
        policy: AcceptancePolicy,
    ) -> TrainingRunResult:
        validate_training_request(
            authorization=authorization,
            contract=contract,
            configuration=configuration,
            policy=policy,
        )
        dev = authorization.development_split
        context = dev.context
        access = self._data_provider.prepare(authorization=authorization, contract=contract)

        with self._data_provider.open_train(access=access) as stream:
            train_df = pd.read_csv(stream, keep_default_na=True)
        with self._data_provider.open_validation(access=access) as stream:
            val_df = pd.read_csv(stream, keep_default_na=True)

        target_col = contract.target.feature.name
        y_train = pd.to_numeric(train_df[target_col], errors="raise").to_numpy(dtype=float)
        y_val = pd.to_numeric(val_df[target_col], errors="raise").to_numpy(dtype=float)

        metrics_requested = (configuration.primary_metric, *configuration.secondary_metrics)
        contract_sha = calculate_bytes_sha256(
            json.dumps(
                contract.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )

        results: list[CandidateResult] = []
        run_started = datetime.now(UTC)

        for candidate in configuration.candidates:
            trial_start = datetime.now(UTC)
            elapsed = (trial_start - run_started).total_seconds()
            if elapsed > configuration.budget.max_training_seconds:
                results.append(
                    CandidateResult(
                        candidate_id=candidate.candidate_id,
                        status=CandidateStatus.SKIPPED,
                        timing=StageTiming(started_at=trial_start, finished_at=datetime.now(UTC)),
                        skip_reason="Training wall-clock budget exceeded.",
                    )
                )
                continue

            try:
                # Fit a fresh TabularPreprocessor strictly on train_df inside each trial.
                preprocessor = TabularPreprocessor(contract)
                if candidate.family == ModelFamily.HISTORICAL_RATE:
                    assert contract.target.derived_rate is not None
                    denom_name = contract.target.derived_rate.denominator
                    X_train = np.asarray(
                        pd.to_numeric(train_df[denom_name], errors="raise"), dtype=float
                    ).reshape(-1, 1)
                    X_val = np.asarray(
                        pd.to_numeric(val_df[denom_name], errors="raise"), dtype=float
                    ).reshape(-1, 1)
                    estimator: Any = _HistoricalRateEstimator(denominator_idx=0)
                    estimator.fit(X_train, y_train)
                    preds = estimator.predict(X_val)
                    bundle = {
                        "bundle_kind": "historical_rate",
                        "denominator_column": denom_name,
                        "estimator": estimator,
                        "feature_names": preprocessor.ordered_predictors,
                        "contract_sha256": contract_sha,
                    }
                else:
                    X_train = preprocessor.fit_transform(train_df)
                    X_val = preprocessor.transform(val_df)
                    estimator = self._build_estimator(candidate, configuration.random_seed)
                    estimator.fit(X_train, y_train)
                    preds = np.asarray(estimator.predict(X_val), dtype=float)
                    if contract.target.feature.constraints.minimum is not None:
                        min_val = float(contract.target.feature.constraints.minimum)
                        preds = np.maximum(min_val, preds)
                    bundle = {
                        "bundle_kind": "tabular_sklearn",
                        "preprocessor": preprocessor,
                        "estimator": estimator,
                        "feature_names": preprocessor.ordered_predictors,
                        "contract_sha256": contract_sha,
                        "family": candidate.family.value,
                    }

                val_metrics = compute_regression_metrics(y_val, preds, metrics_requested)
                buf = BytesIO()
                joblib.dump(bundle, buf)
                bundle_bytes = buf.getvalue()
                model_ref = CandidateModelReference(
                    factory_id=context.factory_id,
                    execution_id=context.execution_id,
                    candidate_id=candidate.candidate_id,
                    content_sha256=calculate_bytes_sha256(bundle_bytes),
                    size_bytes=len(bundle_bytes),
                )
                self._model_storage.put(
                    factory_id=context.factory_id,
                    reference=model_ref,
                    stream=BytesIO(bundle_bytes),
                )
                results.append(
                    CandidateResult(
                        candidate_id=candidate.candidate_id,
                        status=CandidateStatus.SUCCEEDED,
                        timing=StageTiming(started_at=trial_start, finished_at=datetime.now(UTC)),
                        validation_metrics=val_metrics,
                        evaluated_row_count=len(y_val),
                        model_reference=model_ref,
                    )
                )
            except Exception as exc:
                results.append(
                    CandidateResult(
                        candidate_id=candidate.candidate_id,
                        status=CandidateStatus.FAILED,
                        timing=StageTiming(started_at=trial_start, finished_at=datetime.now(UTC)),
                        error=ExecutionError(code="CANDIDATE_TRAINING_FAILED", message=str(exc)),
                    )
                )

        return TrainingRunResult(
            configuration=configuration,
            development_split=dev,
            results=tuple(results),
        )

    def _build_estimator(self, candidate: CandidateConfiguration, seed: int) -> Any:
        params = dict(candidate.parameter_map)
        allowed = self._ALLOWLISTED_PARAMS[candidate.family]
        unknown = set(params.keys()) - allowed
        if unknown:
            raise ValueError(
                f"Disallowed hyperparameters for {candidate.family.value}: {sorted(unknown)}"
            )
        if candidate.family == ModelFamily.DUMMY_REGRESSOR:
            strategy = str(params.get("strategy", "median"))
            return DummyRegressor(strategy=strategy)
        if candidate.family == ModelFamily.RIDGE:
            return Ridge(alpha=float(params.get("alpha", 1.0)), random_state=seed)
        if candidate.family == ModelFamily.ELASTIC_NET:
            return ElasticNet(
                alpha=float(params.get("alpha", 0.5)),
                l1_ratio=float(params.get("l1_ratio", 0.5)),
                max_iter=int(params.get("max_iter", 2000)),
                random_state=seed,
            )
        if candidate.family == ModelFamily.RANDOM_FOREST:
            return RandomForestRegressor(
                n_estimators=int(params.get("n_estimators", 50)),
                max_depth=params.get("max_depth", 8),
                min_samples_leaf=int(params.get("min_samples_leaf", 2)),
                random_state=seed,
                n_jobs=1,
            )
        if candidate.family == ModelFamily.HIST_GRADIENT_BOOSTING:
            return HistGradientBoostingRegressor(
                max_iter=int(params.get("max_iter", 80)),
                max_depth=params.get("max_depth", 5),
                learning_rate=float(params.get("learning_rate", 0.08)),
                random_state=seed,
            )
        if candidate.family == ModelFamily.XGBOOST:
            return XGBRegressor(
                n_estimators=int(params.get("n_estimators", 60)),
                max_depth=int(params.get("max_depth", 4)),
                learning_rate=float(params.get("learning_rate", 0.08)),
                subsample=float(params.get("subsample", 0.9)),
                random_state=seed,
                n_jobs=1,
            )
        if candidate.family == ModelFamily.HURDLE_REGRESSION:
            return _TwoStageHurdleEstimator(
                random_state=seed,
                max_iter=int(params.get("max_iter", 300)),
            )
        raise ValueError(f"Unsupported model family: {candidate.family}")


class TextileProtectedFinalEvaluator(FinalModelEvaluator):
    """Evaluate frozen baseline + challenger models once on protected test records.

    Migration path: Replace with Shared Core P1.7 `ProtectedFinalEvaluator`.
    """

    def __init__(
        self,
        *,
        test_provider: ProtectedTestDataProvider,
        evaluation_registry: EvaluationRegistry,
        model_storage: ModelArtifactStorage,
        contract: DatasetContract,
        identity_namespace: str = "production_record",
    ) -> None:
        self._test_provider = test_provider
        self._eval_registry = evaluation_registry
        self._model_storage = model_storage
        self._contract = contract
        self._identity_namespace = identity_namespace

    def evaluate(
        self, *, selection: FrozenModelSelection, manifest: SplitManifest
    ) -> FinalEvaluationResult:
        context = manifest.context
        scope = _scope_for_context(context, self._identity_namespace)
        self._eval_registry.register_selection(factory_id=context.factory_id, selection=selection)

        reservation = EvaluationReservation.from_selection(
            scope=scope, selection=selection, manifest=manifest
        )
        reserved = self._eval_registry.reserve_test_access(
            factory_id=context.factory_id, reservation=reservation
        )
        started_at = max(datetime.now(UTC), reserved.reserved_at)
        access = self._test_provider.prepare(
            reservation=reserved, selection=selection, contract=self._contract
        )
        with self._test_provider.open_test(access=access) as stream:
            test_df = pd.read_csv(stream, keep_default_na=True)

        target_col = self._contract.target.feature.name
        y_test = pd.to_numeric(test_df[target_col], errors="raise").to_numpy(dtype=float)

        run = selection.training_run
        metrics_requested = (
            run.configuration.primary_metric,
            *run.configuration.secondary_metrics,
        )
        successful = {r.candidate_id: r for r in run.successful_results}

        baseline_res = self._score_candidate(
            factory_id=context.factory_id,
            candidate_result=successful[selection.baseline_candidate_id],
            test_df=test_df,
            y_test=y_test,
            test_record_ids=manifest.test.record_ids,
            metrics_requested=metrics_requested,
        )
        challenger_res = self._score_candidate(
            factory_id=context.factory_id,
            candidate_result=successful[selection.selected_candidate_id],
            test_df=test_df,
            y_test=y_test,
            test_record_ids=manifest.test.record_ids,
            metrics_requested=metrics_requested,
        )

        finished_at = datetime.now(UTC)
        if finished_at < started_at:
            finished_at = started_at
        evaluation = FinalEvaluationResult(
            selection=selection,
            split_manifest=manifest,
            baseline_result=baseline_res,
            selected_result=challenger_res,
            timing=StageTiming(started_at=started_at, finished_at=finished_at),
        )
        self._eval_registry.register_evaluation(
            factory_id=context.factory_id,
            reservation_id=reserved.reservation_id,
            evaluation=evaluation,
        )
        return evaluation

    def _score_candidate(
        self,
        *,
        factory_id: UUID,
        candidate_result: CandidateResult,
        test_df: pd.DataFrame,
        y_test: np.ndarray,
        test_record_ids: tuple[UUID, ...],
        metrics_requested: tuple[RegressionMetric, ...],
    ) -> TestMetricResult:
        ref = candidate_result.model_reference
        assert ref is not None
        with self._model_storage.open_verified(factory_id=factory_id, reference=ref) as stream:
            bundle = joblib.load(stream)

        if bundle.get("bundle_kind") == "historical_rate":
            denom_col = bundle["denominator_column"]
            X_test = np.asarray(
                pd.to_numeric(test_df[denom_col], errors="raise"), dtype=float
            ).reshape(-1, 1)
            preds = np.asarray(bundle["estimator"].predict(X_test), dtype=float)
        else:
            preprocessor: TabularPreprocessor = bundle["preprocessor"]
            X_test = preprocessor.transform(test_df)
            preds = np.asarray(bundle["estimator"].predict(X_test), dtype=float)
            if self._contract.target.feature.constraints.minimum is not None:
                min_val = float(self._contract.target.feature.constraints.minimum)
                preds = np.maximum(min_val, preds)

        computed = compute_regression_metrics(y_test, preds, metrics_requested)
        # Ensure MAE is finite and defined.
        if any(m.metric == RegressionMetric.MAE and m.value is None for m in computed):
            raise ValueError("MAE cannot be undefined during protected test evaluation.")

        return TestMetricResult(
            candidate_id=candidate_result.candidate_id,
            model_artifact_id=ref.artifact_id,
            record_ids=test_record_ids,
            metrics=computed,
        )
