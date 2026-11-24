"""Dataset and record registry protocols."""

from enum import StrEnum
from typing import Protocol, Self, runtime_checkable
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)

from taqamind.core.contracts.feature import FeatureName
from taqamind.core.contracts.target import ContractVersion
from taqamind.core.interfaces.artifacts import SHA256Digest
from taqamind.core.interfaces.dataset import (
    STATE_ORDER,
    DatasetArtifact,
    DatasetState,
)
from taqamind.core.interfaces.lineage import (
    DatasetRecordMembership,
    RecordIdentity,
    RecordMatchResult,
    RecordScope,
    RecordVersion,
)


class RegistryError(Exception):
    """Base exception for registry operations."""


class RegistryNotFoundError(RegistryError):
    """Requested registry entry does not exist."""


class RegistryConflictError(RegistryError):
    """Conflicting registry entry."""


class RegistryScopeError(RegistryError):
    """Registry entry belongs to an incompatible scope."""


class DatasetRegistrationStatus(StrEnum):
    """Outcome of registering an immutable dataset revision."""

    REGISTERED = "registered"

    ALREADY_REGISTERED = "already_registered"


class DatasetRegistrationResult(BaseModel):
    """Result of registering a dataset artifact."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: DatasetRegistrationStatus

    artifact: DatasetArtifact

    matching_source_revision_ids: tuple[UUID, ...] = ()

    @property
    def has_matching_source(self) -> bool:
        return bool(self.matching_source_revision_ids)

    @model_validator(mode="after")
    def validate_matches(self) -> Self:
        matches = self.matching_source_revision_ids

        if len(matches) != len(set(matches)):
            raise ValueError(
                "Matching source revisions must be unique."
            )

        if self.artifact.metadata.revision_id in matches:
            raise ValueError(
                "A revision cannot be its own source match."
            )

        return self


def validate_dataset_parent(
    *,
    parent: DatasetArtifact,
    child: DatasetArtifact,
) -> None:
    """Validate the relationship between two dataset revisions."""

    parent_meta = parent.metadata
    child_meta = child.metadata

    scope_fields = (
        "factory_id",
        "domain",
        "process",
        "dataset_id",
        "task_id",
        "contract_version",
        "task_version",
        "target_version",
    )

    for field in scope_fields:
        if getattr(parent_meta, field) != getattr(child_meta, field):
            raise RegistryScopeError(
                f"Dataset parent scope mismatch for '{field}'."
            )

    if child_meta.state == DatasetState.RAW:
        raise RegistryConflictError(
            "Raw dataset cannot have a parent."
        )

    if child_meta.parent_revision_id != parent_meta.revision_id:
        raise RegistryConflictError(
            "Dataset parent revision mismatch."
        )

    if child_meta.revision_id == parent_meta.revision_id:
        raise RegistryConflictError(
            "Child revision must have a unique identity."
        )

    if child_meta.root_revision_id != parent_meta.root_revision_id:
        raise RegistryConflictError(
            "Dataset root revision mismatch."
        )

    if child_meta.source_file_sha256 != parent_meta.source_file_sha256:
        raise RegistryConflictError(
            "Dataset source hash mismatch."
        )

    if (
        STATE_ORDER[child_meta.state]
        != STATE_ORDER[parent_meta.state] + 1
    ):
        raise RegistryConflictError(
            "Dataset child must advance one processing state."
        )

    if child_meta.created_at < parent_meta.created_at:
        raise RegistryConflictError(
            "Child dataset cannot precede its parent."
        )

    if child.reference.artifact_id == parent.reference.artifact_id:
        raise RegistryConflictError(
            "Child dataset requires a separate artifact identifier."
        )


class DatasetMembershipManifest(BaseModel):
    """Complete physical-row membership for a dataset revision."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    dataset_revision_id: UUID

    scope: RecordScope

    memberships: tuple[DatasetRecordMembership, ...] = ()

    @property
    def row_count(self) -> int:
        return len(self.memberships)

    @property
    def unique_record_count(self) -> int:
        return len({
            membership.record_id
            for membership in self.memberships
        })

    @property
    def has_repeated_record_ids(self) -> bool:
        return self.unique_record_count != self.row_count

    def validate_artifact(
        self,
        artifact: DatasetArtifact,
    ) -> None:
        metadata = artifact.metadata

        if metadata.state == DatasetState.RAW:
            raise ValueError(
                "Raw dataset does not have registered row membership."
            )

        if self.dataset_revision_id != metadata.revision_id:
            raise ValueError(
                "Membership dataset revision mismatch."
            )

        fields = (
            "factory_id",
            "domain",
            "process",
            "dataset_id",
        )

        for field in fields:
            if getattr(self.scope, field) != getattr(metadata, field):
                raise ValueError(
                    f"Membership scope mismatch for '{field}'."
                )

        if self.row_count != metadata.row_count:
            raise ValueError(
                "Membership count does not match dataset row count."
            )

    @model_validator(mode="after")
    def validate_memberships(self) -> Self:
        positions = []

        for membership in self.memberships:
            if (
                membership.dataset_revision_id
                != self.dataset_revision_id
            ):
                raise ValueError(
                    "Membership references another dataset revision."
                )

            if membership.scope != self.scope:
                raise ValueError(
                    "Membership record scope mismatch."
                )

            positions.append(membership.row_position)

        if len(positions) != len(set(positions)):
            raise ValueError(
                "Membership row positions must be unique."
            )

        if set(positions) != set(range(self.row_count)):
            raise ValueError(
                "Membership must cover every physical row position."
            )

        return self


@runtime_checkable
class DatasetRegistry(Protocol):
    """Registry for immutable dataset revisions."""

    def register(
        self,
        *,
        factory_id: UUID,
        artifact: DatasetArtifact,
    ) -> DatasetRegistrationResult:
        """Register a revision after verifying storage and lineage.

        Identical registration may be idempotent.
        Conflicting metadata must never overwrite existing entries.
        """
        ...

    def get(
        self,
        *,
        factory_id: UUID,
        dataset_revision_id: UUID,
    ) -> DatasetArtifact:
        """Retrieve one revision from the requested factory."""
        ...

    def find_raw_by_source_hash(
        self,
        *,
        factory_id: UUID,
        domain: FeatureName,
        process: FeatureName,
        dataset_id: FeatureName,
        source_file_sha256: SHA256Digest,
    ) -> tuple[DatasetArtifact, ...]:
        """Find raw revisions with the same original file hash."""
        ...

    def get_lineage(
        self,
        *,
        factory_id: UUID,
        dataset_revision_id: UUID,
    ) -> tuple[DatasetArtifact, ...]:
        """Return the verified root-to-current revision chain.

        Implementations must reject cycles, missing parents
        and inconsistent lineage.
        """
        ...


@runtime_checkable
class RecordLineageRegistry(Protocol):
    """Registry for stable business identities and record versions."""

    def match(
        self,
        *,
        scope: RecordScope,
        business_key_sha256: SHA256Digest | None,
        content_sha256: SHA256Digest,
        content_fingerprint_version: ContractVersion,
    ) -> RecordMatchResult:
        """Match normalized data without automatically modifying records.

        An unavailable or ambiguous business key must not be
        treated as proof that a record is new.
        """
        ...

    def register_identity(
        self,
        *,
        scope: RecordScope,
        identity: RecordIdentity,
    ) -> RecordIdentity:
        """Register an identity with scope and uniqueness enforcement."""
        ...

    def register_version(
        self,
        *,
        scope: RecordScope,
        version: RecordVersion,
    ) -> RecordVersion:
        """Register a new immutable version or reuse identical content.

        Updated versions require verified confirmation,
        consistent identity and validated parent lineage.
        """
        ...

    def get_identity(
        self,
        *,
        scope: RecordScope,
        record_id: UUID,
    ) -> RecordIdentity:
        ...

    def get_version(
        self,
        *,
        scope: RecordScope,
        record_version_id: UUID,
    ) -> RecordVersion:
        ...

    def get_versions(
        self,
        *,
        scope: RecordScope,
        record_id: UUID,
    ) -> tuple[RecordVersion, ...]:
        ...

    def register_membership(
        self,
        *,
        scope: RecordScope,
        manifest: DatasetMembershipManifest,
    ) -> None:
        """Register row membership after verifying every reference."""
        ...

    def get_membership(
        self,
        *,
        scope: RecordScope,
        dataset_revision_id: UUID,
    ) -> DatasetMembershipManifest:
        ...