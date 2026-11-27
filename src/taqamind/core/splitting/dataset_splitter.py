"""Core P1.6 DatasetSplitter implementation."""

from __future__ import annotations

from uuid import UUID

import numpy as np
import pandas as pd

from taqamind.core.ingestion.artifact_csv import read_verified_csv
from taqamind.core.ingestion.resource_policy import (
    CsvResourcePolicy,
    resolve_internal_csv_policy,
)
from taqamind.core.ingestion.safe_csv import SafeCsvReader
from taqamind.core.interfaces.dataset_registry import RecordLineageRegistry
from taqamind.core.interfaces.lineage import RecordScope
from taqamind.core.interfaces.ml import DatasetSplitter
from taqamind.core.interfaces.readiness import TrainingReadinessResult
from taqamind.core.interfaces.splitting import (
    SplitConfiguration,
    SplitManifest,
    SplitPartition,
    SplitRole,
    SplitStrategy,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage


class CoreDatasetSplitter(DatasetSplitter):
    """Create disjoint temporal, group, group-temporal, or random splits on record IDs."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        lineage_registry: RecordLineageRegistry,
        *,
        delimiter: str = ",",
        identity_namespace: str = "production_record",
        resource_policy: CsvResourcePolicy | None = None,
    ) -> None:
        self._storage = storage
        self._lineage = lineage_registry
        self._resource_policy = resolve_internal_csv_policy(resource_policy)
        self._reader = SafeCsvReader(delimiter=delimiter, resource_policy=self._resource_policy)
        self._identity_namespace = identity_namespace

    def split(
        self,
        *,
        readiness: TrainingReadinessResult,
        configuration: SplitConfiguration,
    ) -> SplitManifest:
        readiness.require_split_eligibility()
        context = readiness.result.context
        artifact = readiness.result.input_artifact
        contract = readiness.contract

        scope = RecordScope(
            factory_id=context.factory_id,
            domain=context.domain,
            process=context.process,
            dataset_id=context.dataset_id,
            identity_namespace=self._identity_namespace,
        )
        membership = self._lineage.get_membership(
            scope=scope,
            dataset_revision_id=artifact.metadata.revision_id,
        )
        membership.validate_artifact(artifact)
        if membership.has_repeated_record_ids:
            raise ValueError(
                "Dataset membership contains repeated record_ids; resolve duplicates before split."
            )

        df = read_verified_csv(
            storage=self._storage,
            factory_id=context.factory_id,
            reference=artifact.reference,
            reader=self._reader,
            policy=self._resource_policy,
        )

        row_record_ids = [item.record_id for item in membership.memberships]

        if configuration.strategy in {SplitStrategy.TEMPORAL, SplitStrategy.GROUP_TEMPORAL}:
            if contract.time_column is None:
                raise ValueError("Temporal split requires DatasetContract.time_column.")
        if configuration.strategy in {SplitStrategy.GROUP, SplitStrategy.GROUP_TEMPORAL}:
            if not contract.grouping_columns:
                raise ValueError("Group split requires DatasetContract.grouping_columns.")

        if configuration.strategy == SplitStrategy.GROUP:
            train_ids, val_ids, test_ids = self._split_group(
                df, row_record_ids, contract.grouping_columns, configuration
            )
        elif configuration.strategy == SplitStrategy.TEMPORAL:
            assert contract.time_column is not None
            train_ids, val_ids, test_ids = self._split_temporal(
                df, row_record_ids, contract.time_column, configuration
            )
        elif configuration.strategy == SplitStrategy.GROUP_TEMPORAL:
            assert contract.time_column is not None
            train_ids, val_ids, test_ids = self._split_group_temporal(
                df,
                row_record_ids,
                contract.grouping_columns,
                contract.time_column,
                configuration,
            )
        else:
            train_ids, val_ids, test_ids = self._split_random(row_record_ids, configuration)

        return SplitManifest(
            context=context,
            dataset_revision_id=artifact.metadata.revision_id,
            configuration=configuration,
            train=SplitPartition(role=SplitRole.TRAIN, record_ids=tuple(train_ids)),
            validation=SplitPartition(role=SplitRole.VALIDATION, record_ids=tuple(val_ids)),
            test=SplitPartition(role=SplitRole.TEST, record_ids=tuple(test_ids)),
        )

    @staticmethod
    def _allocate_blocks(
        blocks: list[list[UUID]],
        total_rows: int,
        configuration: SplitConfiguration,
    ) -> tuple[list[UUID], list[UUID], list[UUID]]:
        if len(blocks) < 3:
            raise ValueError("At least 3 distinct groups/timestamps are required for 3-way split.")
        target_train = max(1, round(total_rows * configuration.train_ratio))
        target_val = max(1, round(total_rows * configuration.validation_ratio))

        train: list[UUID] = []
        val: list[UUID] = []
        test: list[UUID] = []

        for idx, block in enumerate(blocks):
            remaining_blocks = len(blocks) - idx
            if not test and remaining_blocks == 1:
                test.extend(block)
            elif not val and remaining_blocks == 2:
                val.extend(block)
            elif len(train) < target_train and remaining_blocks > 2:
                train.extend(block)
            elif len(val) < target_val and remaining_blocks > 1:
                val.extend(block)
            else:
                test.extend(block)

        if not train or not val or not test:
            raise ValueError("Unable to populate non-empty train/validation/test partitions.")
        return train, val, test

    def _split_group(
        self,
        df: pd.DataFrame,
        record_ids: list[UUID],
        grouping_columns: tuple[str, ...],
        configuration: SplitConfiguration,
    ) -> tuple[list[UUID], list[UUID], list[UUID]]:
        groups: dict[tuple[str, ...], list[UUID]] = {}
        for idx, rid in enumerate(record_ids):
            key = tuple(str(df.iloc[idx][col]) for col in grouping_columns)
            groups.setdefault(key, []).append(rid)

        sorted_keys = sorted(groups.keys())
        rng = np.random.default_rng(configuration.random_seed)
        perm = rng.permutation(len(sorted_keys))
        ordered_blocks = [groups[sorted_keys[i]] for i in perm]
        return self._allocate_blocks(ordered_blocks, len(record_ids), configuration)

    def _split_temporal(
        self,
        df: pd.DataFrame,
        record_ids: list[UUID],
        time_column: str,
        configuration: SplitConfiguration,
    ) -> tuple[list[UUID], list[UUID], list[UUID]]:
        timestamps = pd.to_datetime(df[time_column], format="mixed", utc=True)
        buckets: dict[pd.Timestamp, list[UUID]] = {}
        for idx, rid in enumerate(record_ids):
            ts = timestamps.iloc[idx]
            buckets.setdefault(ts, []).append(rid)
        ordered_blocks = [buckets[ts] for ts in sorted(buckets.keys())]
        return self._allocate_blocks(ordered_blocks, len(record_ids), configuration)

    def _split_group_temporal(
        self,
        df: pd.DataFrame,
        record_ids: list[UUID],
        grouping_columns: tuple[str, ...],
        time_column: str,
        configuration: SplitConfiguration,
    ) -> tuple[list[UUID], list[UUID], list[UUID]]:
        timestamps = pd.to_datetime(df[time_column], format="mixed", utc=True)
        groups: dict[tuple[str, ...], tuple[pd.Timestamp, list[UUID]]] = {}
        for idx, rid in enumerate(record_ids):
            key = tuple(str(df.iloc[idx][col]) for col in grouping_columns)
            ts = timestamps.iloc[idx]
            if key not in groups:
                groups[key] = (ts, [rid])
            else:
                max_ts, ids = groups[key]
                ids.append(rid)
                groups[key] = (max(max_ts, ts), ids)
        ordered = sorted(groups.items(), key=lambda item: (item[1][0], item[0]))
        ordered_blocks = [item[1][1] for item in ordered]
        return self._allocate_blocks(ordered_blocks, len(record_ids), configuration)

    def _split_random(
        self,
        record_ids: list[UUID],
        configuration: SplitConfiguration,
    ) -> tuple[list[UUID], list[UUID], list[UUID]]:
        rng = np.random.default_rng(configuration.random_seed)
        perm = rng.permutation(len(record_ids))
        blocks = [[record_ids[i]] for i in perm]
        return self._allocate_blocks(blocks, len(record_ids), configuration)
