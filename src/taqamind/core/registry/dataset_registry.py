"""Local filesystem implementation of DatasetRegistry."""

from pathlib import Path
from uuid import UUID

from taqamind.core.contracts.feature import FeatureName
from taqamind.core.interfaces.artifacts import SHA256Digest
from taqamind.core.interfaces.dataset import DatasetArtifact, DatasetState
from taqamind.core.interfaces.dataset_registry import (
    DatasetRegistrationResult,
    DatasetRegistrationStatus,
    DatasetRegistry,
    RegistryConflictError,
    RegistryNotFoundError,
    RegistryScopeError,
    validate_dataset_parent,
)
from taqamind.core.interfaces.storage import (
    DatasetArtifactStorage,
    StorageError,
)
from taqamind.core.registry._concurrency import locked_path
from taqamind.core.registry._json_store import (
    atomic_write_text,
    iter_json_models,
    load_json_model,
    safe_resolve,
)


class LocalDatasetRegistry(DatasetRegistry):
    """Persist and validate immutable dataset revision metadata locally."""

    def __init__(
        self,
        registry_dir: str | Path,
        storage: DatasetArtifactStorage,
    ) -> None:
        self._root = Path(registry_dir).resolve()
        self._storage = storage

    def _factory_dir(self, factory_id: UUID) -> Path:
        return safe_resolve(self._root, "datasets", str(factory_id))

    def _revision_path(self, factory_id: UUID, dataset_revision_id: UUID) -> Path:
        return safe_resolve(
            self._root,
            "datasets",
            str(factory_id),
            f"{dataset_revision_id}.json",
        )

    def _verify_stored_artifact(
        self,
        *,
        factory_id: UUID,
        artifact: DatasetArtifact,
    ) -> None:
        try:
            with self._storage.open_verified(
                factory_id=factory_id,
                reference=artifact.reference,
            ):
                pass
        except StorageError as exc:
            raise RegistryConflictError(
                f"Dataset artifact failed storage verification: {exc}"
            ) from exc

    def _matching_raw_ids(
        self,
        *,
        factory_id: UUID,
        artifact: DatasetArtifact,
    ) -> tuple[UUID, ...]:
        meta = artifact.metadata
        if meta.state != DatasetState.RAW:
            return ()
        matches = self.find_raw_by_source_hash(
            factory_id=factory_id,
            domain=meta.domain,
            process=meta.process,
            dataset_id=meta.dataset_id,
            source_file_sha256=meta.source_file_sha256,
        )
        return tuple(
            item.metadata.revision_id
            for item in matches
            if item.metadata.revision_id != meta.revision_id
        )

    def register(
        self,
        *,
        factory_id: UUID,
        artifact: DatasetArtifact,
    ) -> DatasetRegistrationResult:
        if (
            artifact.metadata.factory_id != factory_id
            or artifact.reference.factory_id != factory_id
        ):
            raise RegistryScopeError(
                "Artifact factory_id does not match requested factory scope."
            )

        self._verify_stored_artifact(factory_id=factory_id, artifact=artifact)

        factory_lock = safe_resolve(self._root, "datasets", str(factory_id), ".factory.lock")
        with locked_path(factory_lock):
            target_path = self._revision_path(factory_id, artifact.metadata.revision_id)
            if target_path.is_file():
                existing = load_json_model(target_path, DatasetArtifact)
                if existing == artifact:
                    return DatasetRegistrationResult(
                        status=DatasetRegistrationStatus.ALREADY_REGISTERED,
                        artifact=existing,
                        matching_source_revision_ids=self._matching_raw_ids(
                            factory_id=factory_id,
                            artifact=existing,
                        ),
                    )
                raise RegistryConflictError(
                    "Conflicting metadata for an already-registered dataset revision."
                )

            if artifact.metadata.state != DatasetState.RAW:
                parent_id = artifact.metadata.parent_revision_id
                if parent_id is None:
                    raise RegistryConflictError(
                        "Processed dataset revision must declare a parent revision."
                    )
                parent = self.get(
                    factory_id=factory_id,
                    dataset_revision_id=parent_id,
                )
                validate_dataset_parent(parent=parent, child=artifact)
                # Ensure full root-to-parent chain is intact and acyclic.
                self.get_lineage(
                    factory_id=factory_id,
                    dataset_revision_id=parent_id,
                )

            atomic_write_text(target_path, artifact.model_dump_json(indent=2))
            return DatasetRegistrationResult(
                status=DatasetRegistrationStatus.REGISTERED,
                artifact=artifact,
                matching_source_revision_ids=self._matching_raw_ids(
                    factory_id=factory_id,
                    artifact=artifact,
                ),
            )

    def get(
        self,
        *,
        factory_id: UUID,
        dataset_revision_id: UUID,
    ) -> DatasetArtifact:
        path = self._revision_path(factory_id, dataset_revision_id)
        if not path.is_file():
            raise RegistryNotFoundError(
                f"Dataset revision '{dataset_revision_id}' not found for factory."
            )
        artifact = load_json_model(path, DatasetArtifact)
        if artifact.metadata.factory_id != factory_id:
            raise RegistryScopeError(
                "Stored dataset artifact factory_id mismatch."
            )
        return artifact

    def find_raw_by_source_hash(
        self,
        *,
        factory_id: UUID,
        domain: FeatureName,
        process: FeatureName,
        dataset_id: FeatureName,
        source_file_sha256: SHA256Digest,
    ) -> tuple[DatasetArtifact, ...]:
        factory_dir = self._factory_dir(factory_id)
        all_artifacts = iter_json_models(factory_dir, DatasetArtifact)
        matches = [
            item
            for item in all_artifacts
            if (
                item.metadata.factory_id == factory_id
                and item.metadata.state == DatasetState.RAW
                and item.metadata.domain == domain
                and item.metadata.process == process
                and item.metadata.dataset_id == dataset_id
                and item.metadata.source_file_sha256 == source_file_sha256
            )
        ]
        matches.sort(
            key=lambda item: (item.metadata.created_at, str(item.metadata.revision_id))
        )
        return tuple(matches)

    def get_lineage(
        self,
        *,
        factory_id: UUID,
        dataset_revision_id: UUID,
    ) -> tuple[DatasetArtifact, ...]:
        chain_reversed: list[DatasetArtifact] = []
        visited: set[UUID] = set()
        current_id: UUID | None = dataset_revision_id

        while current_id is not None:
            if current_id in visited:
                raise RegistryConflictError(
                    "Cycle detected in dataset revision lineage."
                )
            visited.add(current_id)
            artifact = self.get(
                factory_id=factory_id,
                dataset_revision_id=current_id,
            )
            chain_reversed.append(artifact)
            current_id = artifact.metadata.parent_revision_id

        chain = tuple(reversed(chain_reversed))
        if chain[0].metadata.state != DatasetState.RAW:
            raise RegistryConflictError(
                "Dataset lineage must originate from a RAW revision."
            )

        for index in range(len(chain) - 1):
            validate_dataset_parent(parent=chain[index], child=chain[index + 1])

        return chain
