"""Restricted local data access providers implementing Shared Core protocols.

Implements:
    - `DevelopmentDataProvider` -> `LocalDevelopmentDataProvider`
    - `ProtectedTestDataProvider` -> `LocalProtectedTestDataProvider`
    - `ActiveModelProvider` -> `LocalActiveModelProvider`

Physically materializes train, validation, and protected test partitions as
separate verified files so model trainers never read protected test bytes.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

from taqamind.core.contracts import DatasetContract
from taqamind.core.interfaces.artifacts import calculate_bytes_sha256
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.data_access import (
    DataAccessDeniedError,
    DevelopmentDataAccess,
    PartitionDataReference,
    ProtectedTestDataAccess,
    validate_active_model_access,
)
from taqamind.core.interfaces.dataset_registry import (
    DatasetRegistry,
    RecordLineageRegistry,
)
from taqamind.core.interfaces.evaluation import FrozenModelSelection
from taqamind.core.interfaces.leakage import TrainingAuthorization
from taqamind.core.interfaces.model_registry import (
    ModelRegistry,
    ModelRegistryEntry,
    ModelScope,
)
from taqamind.core.interfaces.splitting import SplitPartition, SplitRole
from taqamind.core.interfaces.storage import (
    DatasetArtifactStorage,
    ModelArtifactStorage,
)
from taqamind.core.interfaces.training_registry import (
    EvaluationReservation,
    RecordUsage,
    RecordUsageRole,
)
from taqamind.domains.iron_manufacturing.adapters.processing import (
    _df_to_csv_bytes,
    _read_artifact_df,
)
from taqamind.domains.iron_manufacturing.adapters.readiness_and_splitting import (
    build_record_scope,
)
from taqamind.domains.iron_manufacturing.adapters.training_registry import (
    LocalEvaluationRegistry,
    LocalRecordUsageRegistry,
    LocalSplitRegistry,
)


class _PartitionMaterializer:
    """Internal helper to materialize verified partition CSV slices by record ID."""

    def __init__(
        self,
        *,
        root: Path | str,
        storage: DatasetArtifactStorage,
        dataset_registry: DatasetRegistry,
        lineage_registry: RecordLineageRegistry,
    ) -> None:
        self._root = Path(root).resolve() / "partitions"
        self._storage = storage
        self._dataset_registry = dataset_registry
        self._lineage_registry = lineage_registry
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, ref: PartitionDataReference) -> Path:
        return (
            self._root
            / str(ref.factory_id)
            / str(ref.split_id)
            / f"{ref.role.value}_{ref.partition_artifact_id}.csv"
        )

    def materialize(
        self,
        *,
        context: PipelineContext,
        split_id: UUID,
        dataset_revision_id: UUID,
        partition: SplitPartition,
        contract: DatasetContract,
    ) -> PartitionDataReference:
        artifact = self._dataset_registry.get(
            factory_id=context.factory_id,
            dataset_revision_id=dataset_revision_id,
        )
        scope = build_record_scope(context)
        membership = self._lineage_registry.get_membership(
            scope=scope,
            dataset_revision_id=dataset_revision_id,
        )
        membership.validate_artifact(artifact)

        df = _read_artifact_df(self._storage, context.factory_id, artifact)
        rec_to_row = {m.record_id: m.row_position for m in membership.memberships}
        row_positions = [rec_to_row[rid] for rid in partition.record_ids]

        learning_cols = tuple(
            col
            for col in (*contract.predictor_names, contract.target.feature.name)
            if col in df.columns
        )
        sliced = df.iloc[row_positions].loc[:, list(learning_cols)].copy()
        csv_bytes = _df_to_csv_bytes(sliced)
        sha = calculate_bytes_sha256(csv_bytes)

        ref = PartitionDataReference(
            partition_artifact_id=uuid4(),
            factory_id=context.factory_id,
            dataset_revision_id=dataset_revision_id,
            split_id=split_id,
            role=partition.role,
            record_ids=partition.record_ids,
            columns=learning_cols,
            content_sha256=sha,
            size_bytes=len(csv_bytes),
        )
        ref.validate_columns(contract)

        dest = self._path(ref)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(csv_bytes)
        return ref

    @contextmanager
    def open_verified_partition(self, ref: PartitionDataReference) -> Iterator[BinaryIO]:
        path = self._path(ref)
        if not path.exists():
            raise DataAccessDeniedError(f"Partition artifact missing: {ref.partition_artifact_id}.")
        raw = path.read_bytes()
        if len(raw) != ref.size_bytes or calculate_bytes_sha256(raw) != ref.content_sha256:
            raise DataAccessDeniedError("Partition artifact failed SHA-256 integrity verification.")
        stream = io.BytesIO(raw)
        try:
            yield stream
        finally:
            stream.close()


class LocalDevelopmentDataProvider:
    """Provides verified TRAIN and VALIDATION partitions only (`DevelopmentDataProvider`)."""

    def __init__(
        self,
        *,
        root: Path | str,
        storage: DatasetArtifactStorage,
        dataset_registry: DatasetRegistry,
        lineage_registry: RecordLineageRegistry,
        split_registry: LocalSplitRegistry,
        usage_registry: LocalRecordUsageRegistry,
    ) -> None:
        self._materializer = _PartitionMaterializer(
            root=root,
            storage=storage,
            dataset_registry=dataset_registry,
            lineage_registry=lineage_registry,
        )
        self._lineage_registry = lineage_registry
        self._split_registry = split_registry
        self._usage_registry = usage_registry

    def prepare(
        self,
        *,
        authorization: TrainingAuthorization,
        contract: DatasetContract,
    ) -> DevelopmentDataAccess:
        dev = authorization.development_split
        context = dev.context
        self._split_registry.verify_training_authorization(
            factory_id=context.factory_id,
            authorization=authorization,
        )
        train_ref = self._materializer.materialize(
            context=context,
            split_id=dev.split_id,
            dataset_revision_id=dev.dataset_revision_id,
            partition=dev.train,
            contract=contract,
        )
        val_ref = self._materializer.materialize(
            context=context,
            split_id=dev.split_id,
            dataset_revision_id=dev.dataset_revision_id,
            partition=dev.validation,
            contract=contract,
        )
        access = DevelopmentDataAccess(
            authorization=authorization,
            train=train_ref,
            validation=val_ref,
        )
        access.validate_contract(contract)
        return access

    def _record_partition_exposure(
        self,
        access: DevelopmentDataAccess,
        ref: PartitionDataReference,
        role: RecordUsageRole,
    ) -> None:
        context = access.authorization.development_split.context
        scope = build_record_scope(context)
        membership = self._lineage_registry.get_membership(
            scope=scope,
            dataset_revision_id=ref.dataset_revision_id,
        )
        rec_to_ver = {m.record_id: m.record_version_id for m in membership.memberships}
        usages = tuple(
            RecordUsage(
                scope=scope,
                record_id=rid,
                record_version_id=rec_to_ver[rid],
                execution_id=context.execution_id,
                split_id=ref.split_id,
                role=role,
            )
            for rid in ref.record_ids
        )
        self._usage_registry.record_usage(scope=scope, usages=usages)

    def open_train(
        self,
        *,
        access: DevelopmentDataAccess,
    ) -> AbstractContextManager[BinaryIO]:
        self._record_partition_exposure(access, access.train, RecordUsageRole.TRAIN)
        return self._materializer.open_verified_partition(access.train)

    def open_validation(
        self,
        *,
        access: DevelopmentDataAccess,
    ) -> AbstractContextManager[BinaryIO]:
        self._record_partition_exposure(access, access.validation, RecordUsageRole.VALIDATION)
        return self._materializer.open_verified_partition(access.validation)


class LocalProtectedTestDataProvider:
    """Provides protected TEST partition only after reservation (`ProtectedTestDataProvider`)."""

    def __init__(
        self,
        *,
        root: Path | str,
        storage: DatasetArtifactStorage,
        dataset_registry: DatasetRegistry,
        lineage_registry: RecordLineageRegistry,
        split_registry: LocalSplitRegistry,
        evaluation_registry: LocalEvaluationRegistry,
    ) -> None:
        self._materializer = _PartitionMaterializer(
            root=root,
            storage=storage,
            dataset_registry=dataset_registry,
            lineage_registry=lineage_registry,
        )
        self._split_registry = split_registry
        self._evaluation_registry = evaluation_registry

    def prepare(
        self,
        *,
        reservation: EvaluationReservation,
        selection: FrozenModelSelection,
        contract: DatasetContract,
    ) -> ProtectedTestDataAccess:
        dev = selection.training_run.development_split
        context = dev.context
        registered_res = self._evaluation_registry.get_reservation(
            factory_id=context.factory_id,
            reservation_id=reservation.reservation_id,
        )
        if registered_res != reservation:
            raise DataAccessDeniedError("Reservation does not match registered evaluation record.")

        manifest = self._split_registry.get_split(
            factory_id=context.factory_id,
            split_id=reservation.split_id,
        )
        if set(manifest.test.record_ids) != set(reservation.test_record_ids):
            raise DataAccessDeniedError("Reservation does not match registered test partition.")

        test_ref = self._materializer.materialize(
            context=context,
            split_id=manifest.split_id,
            dataset_revision_id=manifest.dataset_revision_id,
            partition=SplitPartition(role=SplitRole.TEST, record_ids=reservation.test_record_ids),
            contract=contract,
        )
        access = ProtectedTestDataAccess(
            reservation=reservation,
            selection=selection,
            test=test_ref,
        )
        access.validate_contract(contract)
        return access

    def open_test(
        self,
        *,
        access: ProtectedTestDataAccess,
    ) -> AbstractContextManager[BinaryIO]:
        context = access.selection.training_run.development_split.context
        # Persist exposure BEFORE releasing bytes; survives subsequent crashes.
        self._evaluation_registry.mark_test_exposed(
            factory_id=context.factory_id,
            reservation_id=access.reservation.reservation_id,
        )
        return self._materializer.open_verified_partition(access.test)


class LocalActiveModelProvider:
    """Read-only active model bundle provider (`ActiveModelProvider`)."""

    def __init__(
        self,
        *,
        model_registry: ModelRegistry,
        model_storage: ModelArtifactStorage,
    ) -> None:
        self._model_registry = model_registry
        self._model_storage = model_storage

    def get_active(
        self,
        *,
        scope: ModelScope,
    ) -> ModelRegistryEntry | None:
        return self._model_registry.get_active(scope=scope)

    def open_bundle(
        self,
        *,
        context: PipelineContext,
        entry: ModelRegistryEntry,
        contract: DatasetContract,
    ) -> AbstractContextManager[BinaryIO]:
        registered = self._model_registry.get(
            scope=entry.version.scope,
            model_version_id=entry.version.model_version_id,
        )
        if registered != entry:
            raise DataAccessDeniedError("Requested model entry does not match trusted registry.")
        validate_active_model_access(
            context=context,
            entry=registered,
            contract=contract,
        )
        return self._model_storage.open_verified(
            factory_id=context.factory_id,
            reference=registered.version.artifact,
        )
