"""Core P1.7 restricted development and protected test data providers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

import pandas as pd

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.ingestion.safe_csv import SafeCsvReader
from taqamind.core.interfaces.artifacts import (
    ArtifactFormat,
    ArtifactReference,
    calculate_bytes_sha256,
)
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.data_access import (
    DataAccessDeniedError,
    DevelopmentDataAccess,
    DevelopmentDataProvider,
    PartitionDataReference,
    ProtectedTestDataAccess,
    ProtectedTestDataProvider,
)
from taqamind.core.interfaces.dataset_registry import (
    DatasetMembershipManifest,
    DatasetRegistry,
    RecordLineageRegistry,
)
from taqamind.core.interfaces.evaluation import FrozenModelSelection
from taqamind.core.interfaces.leakage import TrainingAuthorization
from taqamind.core.interfaces.lineage import RecordScope
from taqamind.core.interfaces.splitting import SplitManifest, SplitPartition, SplitRole
from taqamind.core.interfaces.storage import DatasetArtifactStorage
from taqamind.core.interfaces.training_registry import (
    EvaluationRegistry,
    EvaluationReservation,
    RecordUsage,
    RecordUsageRegistry,
    RecordUsageRole,
    SplitRegistry,
)
from taqamind.core.preprocessing.type_conversion import ContractTypeConverter
from taqamind.core.registry._json_store import safe_resolve


class LocalDevelopmentDataProvider(DevelopmentDataProvider):
    """Materializes only approved train/validation partitions from trusted registries."""

    def __init__(
        self,
        *,
        partition_root: Path,
        storage: DatasetArtifactStorage,
        dataset_registry: DatasetRegistry,
        lineage_registry: RecordLineageRegistry,
        split_registry: SplitRegistry,
        usage_registry: RecordUsageRegistry,
        csv_reader: SafeCsvReader | None = None,
        identity_namespace: str = "production_record",
    ) -> None:
        self._partition_root = Path(partition_root)
        self._partition_root.mkdir(parents=True, exist_ok=True)
        self._storage = storage
        self._dataset_registry = dataset_registry
        self._lineage_registry = lineage_registry
        self._split_registry = split_registry
        self._usage_registry = usage_registry
        self._csv_reader = csv_reader or SafeCsvReader()
        self._identity_namespace = identity_namespace

    def _scope_for(self, context: PipelineContext) -> RecordScope:
        return RecordScope(
            factory_id=context.factory_id,
            domain=context.domain,
            process=context.process,
            dataset_id=context.dataset_id,
            identity_namespace=self._identity_namespace,
        )

    def _partition_path(self, *, factory_id: UUID, partition_artifact_id: UUID) -> Path:
        return safe_resolve(self._partition_root, f"{factory_id}/{partition_artifact_id}.csv")

    def _load_source_dataframe(
        self,
        *,
        context: PipelineContext,
        dataset_revision_id: UUID,
        contract: DatasetContract,
    ) -> tuple[pd.DataFrame, DatasetMembershipManifest]:
        dataset_artifact = self._dataset_registry.get(
            factory_id=context.factory_id,
            dataset_revision_id=dataset_revision_id,
        )
        with self._storage.open_verified(
            factory_id=context.factory_id,
            reference=dataset_artifact.reference,
        ) as stream:
            df = self._csv_reader.read(stream.read())
        membership = self._lineage_registry.get_membership(
            scope=self._scope_for(context),
            dataset_revision_id=dataset_revision_id,
        )
        if len(df) != len(membership.memberships):
            raise DataAccessDeniedError(
                "Dataset row count does not match registered membership manifest."
            )
        return df, membership

    def _materialize_partition(
        self,
        *,
        df: pd.DataFrame,
        membership: DatasetMembershipManifest,
        partition: SplitPartition,
        split_id: UUID,
        dataset_revision_id: UUID,
        factory_id: UUID,
        contract: DatasetContract,
    ) -> PartitionDataReference:
        record_to_index = {
            item.record_id: idx for idx, item in enumerate(membership.memberships)
        }
        indices: list[int] = []
        for rec_id in partition.record_ids:
            if rec_id not in record_to_index:
                raise DataAccessDeniedError(f"Record '{rec_id}' missing from membership.")
            indices.append(record_to_index[rec_id])

        allowed = set(contract.predictor_names) | {contract.target.feature.name}
        selected_cols = tuple(col for col in df.columns if col in allowed)
        sub_df = df.iloc[indices][list(selected_cols)].copy()
        payload = sub_df.to_csv(index=False).encode("utf-8")
        sha256 = calculate_bytes_sha256(payload)
        partition_artifact_id = uuid4()
        path = self._partition_path(
            factory_id=factory_id,
            partition_artifact_id=partition_artifact_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

        return PartitionDataReference(
            partition_artifact_id=partition_artifact_id,
            factory_id=factory_id,
            dataset_revision_id=dataset_revision_id,
            split_id=split_id,
            role=partition.role,
            record_ids=partition.record_ids,
            columns=selected_cols,
            content_sha256=sha256,
            size_bytes=len(payload),
        )

    def prepare(
        self,
        *,
        authorization: TrainingAuthorization,
        contract: DatasetContract,
    ) -> DevelopmentDataAccess:
        context = authorization.development_split.context
        factory_id = context.factory_id
        self._split_registry.verify_training_authorization(
            factory_id=factory_id,
            authorization=authorization,
        )
        manifest = self._split_registry.get_split(
            factory_id=factory_id,
            split_id=authorization.development_split.split_id,
        )
        df, membership = self._load_source_dataframe(
            context=context,
            dataset_revision_id=manifest.dataset_revision_id,
            contract=contract,
        )
        train_ref = self._materialize_partition(
            df=df,
            membership=membership,
            partition=manifest.train,
            split_id=manifest.split_id,
            dataset_revision_id=manifest.dataset_revision_id,
            factory_id=factory_id,
            contract=contract,
        )
        val_ref = self._materialize_partition(
            df=df,
            membership=membership,
            partition=manifest.validation,
            split_id=manifest.split_id,
            dataset_revision_id=manifest.dataset_revision_id,
            factory_id=factory_id,
            contract=contract,
        )
        access = DevelopmentDataAccess(
            authorization=authorization,
            train=train_ref,
            validation=val_ref,
        )
        access.validate_contract(contract)
        return access

    def _record_and_open(
        self,
        *,
        context: PipelineContext,
        partition: PartitionDataReference,
        role: RecordUsageRole,
    ) -> BytesIO:
        membership = self._lineage_registry.get_membership(
            scope=self._scope_for(context),
            dataset_revision_id=partition.dataset_revision_id,
        )
        versions = {item.record_id: item.record_version_id for item in membership.memberships}
        scope = self._scope_for(context)
        usages = tuple(
            RecordUsage(
                scope=scope,
                record_id=rec_id,
                record_version_id=versions[rec_id],
                execution_id=context.execution_id,
                split_id=partition.split_id,
                role=role,
            )
            for rec_id in partition.record_ids
        )
        self._usage_registry.record_usage(scope=scope, usages=usages)

        path = self._partition_path(
            factory_id=partition.factory_id,
            partition_artifact_id=partition.partition_artifact_id,
        )
        payload = path.read_bytes()
        if calculate_bytes_sha256(payload) != partition.content_sha256:
            raise DataAccessDeniedError("Partition integrity check failed.")
        return BytesIO(payload)

    @contextmanager
    def open_train(self, *, access: DevelopmentDataAccess) -> Iterator[BinaryIO]:
        stream = self._record_and_open(
            context=access.authorization.development_split.context,
            partition=access.train,
            role=RecordUsageRole.TRAIN,
        )
        try:
            yield stream
        finally:
            stream.close()

    @contextmanager
    def open_validation(self, *, access: DevelopmentDataAccess) -> Iterator[BinaryIO]:
        stream = self._record_and_open(
            context=access.authorization.development_split.context,
            partition=access.validation,
            role=RecordUsageRole.VALIDATION,
        )
        try:
            yield stream
        finally:
            stream.close()


class LocalProtectedTestDataProvider(ProtectedTestDataProvider):
    """Provides protected test partition only for registered reservations."""

    def __init__(
        self,
        *,
        partition_root: Path,
        storage: DatasetArtifactStorage,
        dataset_registry: DatasetRegistry,
        lineage_registry: RecordLineageRegistry,
        split_registry: SplitRegistry,
        evaluation_registry: EvaluationRegistry,
        csv_reader: SafeCsvReader | None = None,
        identity_namespace: str = "production_record",
    ) -> None:
        self._partition_root = Path(partition_root)
        self._partition_root.mkdir(parents=True, exist_ok=True)
        self._storage = storage
        self._dataset_registry = dataset_registry
        self._lineage_registry = lineage_registry
        self._split_registry = split_registry
        self._evaluation_registry = evaluation_registry
        self._csv_reader = csv_reader or SafeCsvReader()
        self._identity_namespace = identity_namespace

    def _partition_path(self, *, factory_id: UUID, partition_artifact_id: UUID) -> Path:
        return safe_resolve(self._partition_root, f"{factory_id}/{partition_artifact_id}.csv")

    def prepare(
        self,
        *,
        reservation: EvaluationReservation,
        selection: FrozenModelSelection,
        contract: DatasetContract,
    ) -> ProtectedTestDataAccess:
        development = selection.training_run.development_split
        context = development.context
        factory_id = context.factory_id
        manifest: SplitManifest = self._split_registry.get_split(
            factory_id=factory_id,
            split_id=development.split_id,
        )
        if set(manifest.test.record_ids) != set(reservation.test_record_ids):
            raise DataAccessDeniedError("Reservation test records do not match split manifest.")

        dataset_artifact = self._dataset_registry.get(
            factory_id=factory_id,
            dataset_revision_id=manifest.dataset_revision_id,
        )
        with self._storage.open_verified(
            factory_id=factory_id,
            reference=dataset_artifact.reference,
        ) as stream:
            df = self._csv_reader.read(stream.read())

        membership = self._lineage_registry.get_membership(
            scope=reservation.scope,
            dataset_revision_id=manifest.dataset_revision_id,
        )
        record_to_index = {
            item.record_id: idx for idx, item in enumerate(membership.memberships)
        }
        indices = [record_to_index[r] for r in reservation.test_record_ids]
        allowed = set(contract.predictor_names) | {contract.target.feature.name}
        selected_cols = tuple(col for col in df.columns if col in allowed)
        sub_df = df.iloc[indices][list(selected_cols)].copy()
        payload = sub_df.to_csv(index=False).encode("utf-8")
        sha256 = calculate_bytes_sha256(payload)

        partition_artifact_id = uuid4()
        path = self._partition_path(
            factory_id=factory_id,
            partition_artifact_id=partition_artifact_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

        test_ref = PartitionDataReference(
            partition_artifact_id=partition_artifact_id,
            factory_id=factory_id,
            dataset_revision_id=manifest.dataset_revision_id,
            split_id=manifest.split_id,
            role=SplitRole.TEST,
            record_ids=reservation.test_record_ids,
            columns=selected_cols,
            content_sha256=sha256,
            size_bytes=len(payload),
        )
        access = ProtectedTestDataAccess(
            reservation=reservation,
            selection=selection,
            test=test_ref,
        )
        access.validate_contract(contract)
        return access

    @contextmanager
    def open_test(self, *, access: ProtectedTestDataAccess) -> Iterator[BinaryIO]:
        factory_id = access.test.factory_id
        self._evaluation_registry.mark_test_exposed(
            factory_id=factory_id,
            reservation_id=access.reservation.reservation_id,
        )
        path = self._partition_path(
            factory_id=factory_id,
            partition_artifact_id=access.test.partition_artifact_id,
        )
        payload = path.read_bytes()
        if calculate_bytes_sha256(payload) != access.test.content_sha256:
            raise DataAccessDeniedError("Protected test partition integrity check failed.")
        stream = BytesIO(payload)
        try:
            yield stream
        finally:
            stream.close()


def read_partition_dataframe(
    stream: BinaryIO,
    csv_reader: SafeCsvReader | None = None,
) -> pd.DataFrame:
    """Read a materialized partition CSV stream using SafeCsvReader."""
    reader = csv_reader or SafeCsvReader()
    return reader.read(stream.read())


def convert_dataframe_by_contract(
    df: pd.DataFrame,
    *,
    contract: DatasetContract,
    converter: ContractTypeConverter | None = None,
) -> pd.DataFrame:
    """Convert string-typed CSV columns according to their contract DataType."""
    conv = converter or ContractTypeConverter()
    col_map = contract.column_map
    out = df.copy()
    for col in out.columns:
        if col in col_map:
            out[col] = conv.convert(out[col], data_type=col_map[col].data_type).values
    return out


__all__ = [
    "ArtifactFormat",
    "ArtifactReference",
    "LocalDevelopmentDataProvider",
    "LocalProtectedTestDataProvider",
    "convert_dataframe_by_contract",
    "read_partition_dataframe",
]
