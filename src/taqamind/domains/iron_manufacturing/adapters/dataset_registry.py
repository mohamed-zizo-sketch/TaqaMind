"""Local JSON-backed DatasetRegistry and RecordLineageRegistry adapters.

Migration note:
    Temporary local implementations of Shared Core `DatasetRegistry` and
    `RecordLineageRegistry` using atomic file writes and factory-scoped paths.
    Designed for migration to `taqamind.core.lineage` / `taqamind.core.registry`.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from taqamind.core.interfaces.dataset import DatasetArtifact, DatasetState
from taqamind.core.interfaces.dataset_registry import (
    DatasetMembershipManifest,
    DatasetRegistrationResult,
    DatasetRegistrationStatus,
    RegistryConflictError,
    RegistryNotFoundError,
    RegistryScopeError,
    validate_dataset_parent,
)
from taqamind.core.interfaces.lineage import (
    RecordIdentity,
    RecordIdentityStatus,
    RecordMatchResult,
    RecordMatchStatus,
    RecordScope,
    RecordVersion,
)
from taqamind.core.interfaces.serialization import (
    deserialize_contract,
    serialize_contract,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp_json_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


class LocalDatasetRegistry:
    """Durable factory-scoped implementation of Shared Core `DatasetRegistry`."""

    def __init__(
        self,
        root: Path | str,
        *,
        storage: DatasetArtifactStorage | None = None,
    ) -> None:
        self._root = Path(root).resolve() / "registries" / "datasets"
        self._storage = storage
        self._root.mkdir(parents=True, exist_ok=True)

    def _factory_file(self, factory_id: UUID) -> Path:
        return self._root / f"{factory_id}.json"

    def _load_factory(self, factory_id: UUID) -> dict[str, dict[str, Any]]:
        return _read_json(self._factory_file(factory_id), {})

    def _save_factory(self, factory_id: UUID, data: dict[str, dict[str, Any]]) -> None:
        _atomic_write_json(self._factory_file(factory_id), data)

    def register(
        self,
        *,
        factory_id: UUID,
        artifact: DatasetArtifact,
    ) -> DatasetRegistrationResult:
        if artifact.metadata.factory_id != factory_id:
            raise RegistryScopeError("Dataset artifact factory does not match requested factory.")

        if self._storage is not None and not self._storage.exists(
            factory_id=factory_id,
            reference=artifact.reference,
        ):
            raise RegistryConflictError(
                "Dataset artifact content must be stored before registration."
            )

        store = self._load_factory(factory_id)
        rev_key = str(artifact.metadata.revision_id)

        if rev_key in store:
            existing = deserialize_contract(DatasetArtifact, store[rev_key])
            if existing != artifact:
                raise RegistryConflictError(
                    f"Conflicting dataset registration for revision '{rev_key}'."
                )
            matching_raw = self._matching_raw_ids(store, artifact)
            return DatasetRegistrationResult(
                status=DatasetRegistrationStatus.ALREADY_REGISTERED,
                artifact=existing,
                matching_source_revision_ids=matching_raw,
            )

        if artifact.metadata.state != DatasetState.RAW:
            parent_id = artifact.metadata.parent_revision_id
            if parent_id is None or str(parent_id) not in store:
                raise RegistryConflictError(
                    f"Parent dataset revision '{parent_id}' is not registered."
                )
            parent = deserialize_contract(DatasetArtifact, store[str(parent_id)])
            validate_dataset_parent(parent=parent, child=artifact)

        matching_raw = self._matching_raw_ids(store, artifact)
        store[rev_key] = serialize_contract(artifact)
        self._save_factory(factory_id, store)

        return DatasetRegistrationResult(
            status=DatasetRegistrationStatus.REGISTERED,
            artifact=artifact,
            matching_source_revision_ids=matching_raw,
        )

    def _matching_raw_ids(
        self,
        store: dict[str, dict[str, Any]],
        artifact: DatasetArtifact,
    ) -> tuple[UUID, ...]:
        if artifact.metadata.state != DatasetState.RAW:
            return ()
        matches: list[UUID] = []
        meta = artifact.metadata
        for raw_payload in store.values():
            other = deserialize_contract(DatasetArtifact, raw_payload)
            om = other.metadata
            if (
                om.state == DatasetState.RAW
                and om.revision_id != meta.revision_id
                and om.domain == meta.domain
                and om.process == meta.process
                and om.dataset_id == meta.dataset_id
                and om.source_file_sha256 == meta.source_file_sha256
            ):
                matches.append(om.revision_id)
        return tuple(sorted(matches, key=str))

    def get(
        self,
        *,
        factory_id: UUID,
        dataset_revision_id: UUID,
    ) -> DatasetArtifact:
        store = self._load_factory(factory_id)
        key = str(dataset_revision_id)
        if key not in store:
            raise RegistryNotFoundError(
                f"Dataset revision '{dataset_revision_id}' not found for factory '{factory_id}'."
            )
        return deserialize_contract(DatasetArtifact, store[key])

    def find_raw_by_source_hash(
        self,
        *,
        factory_id: UUID,
        domain: str,
        process: str,
        dataset_id: str,
        source_file_sha256: str,
    ) -> tuple[DatasetArtifact, ...]:
        store = self._load_factory(factory_id)
        found: list[DatasetArtifact] = []
        for payload in store.values():
            artifact = deserialize_contract(DatasetArtifact, payload)
            m = artifact.metadata
            if (
                m.state == DatasetState.RAW
                and m.domain == domain
                and m.process == process
                and m.dataset_id == dataset_id
                and m.source_file_sha256 == source_file_sha256
            ):
                found.append(artifact)
        found.sort(key=lambda item: (item.metadata.created_at, str(item.metadata.revision_id)))
        return tuple(found)

    def get_lineage(
        self,
        *,
        factory_id: UUID,
        dataset_revision_id: UUID,
    ) -> tuple[DatasetArtifact, ...]:
        store = self._load_factory(factory_id)
        chain: list[DatasetArtifact] = []
        visited: set[UUID] = set()
        current_id: UUID | None = dataset_revision_id

        while current_id is not None:
            if current_id in visited:
                raise RegistryConflictError("Cycle detected in dataset lineage.")
            visited.add(current_id)
            key = str(current_id)
            if key not in store:
                raise RegistryNotFoundError(f"Missing lineage revision '{current_id}'.")
            artifact = deserialize_contract(DatasetArtifact, store[key])
            chain.append(artifact)
            current_id = artifact.metadata.parent_revision_id

        chain.reverse()
        if chain[0].metadata.state != DatasetState.RAW:
            raise RegistryConflictError("Lineage chain does not originate at a RAW revision.")
        for parent, child in zip(chain, chain[1:], strict=False):
            validate_dataset_parent(parent=parent, child=child)
        return tuple(chain)


class LocalRecordLineageRegistry:
    """Durable factory- and scope-isolated implementation of `RecordLineageRegistry`."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve() / "registries" / "lineage"
        self._root.mkdir(parents=True, exist_ok=True)

    def _scope_file(self, scope: RecordScope) -> Path:
        return (
            self._root
            / str(scope.factory_id)
            / scope.domain
            / scope.process
            / f"{scope.dataset_id}__{scope.identity_namespace}__{scope.identity_version}.json"
        )

    def _load(self, scope: RecordScope) -> dict[str, Any]:
        return _read_json(
            self._scope_file(scope),
            {
                "identities": {},
                "business_key_index": {},
                "versions": {},
                "versions_by_record": {},
                "memberships": {},
            },
        )

    def _save(self, scope: RecordScope, data: dict[str, Any]) -> None:
        _atomic_write_json(self._scope_file(scope), data)

    def match(
        self,
        *,
        scope: RecordScope,
        business_key_sha256: str | None,
        content_sha256: str,
        content_fingerprint_version: str,
    ) -> RecordMatchResult:
        if business_key_sha256 is None:
            return RecordMatchResult(
                status=RecordMatchStatus.UNVERIFIED,
                reason="Missing verified business key fingerprint; cannot match automatically.",
            )

        data = self._load(scope)
        record_ids: list[str] = data["business_key_index"].get(business_key_sha256, [])
        if not record_ids:
            return RecordMatchResult(status=RecordMatchStatus.NEW_RECORD)

        confirmed_records: list[RecordIdentity] = []
        for rec_id in record_ids:
            ident = deserialize_contract(RecordIdentity, data["identities"][rec_id])
            if ident.eligible_for_automatic_matching:
                confirmed_records.append(ident)

        if len(confirmed_records) > 1:
            return RecordMatchResult(
                status=RecordMatchStatus.AMBIGUOUS,
                reason="Multiple confirmed record identities share the same business key digest.",
            )

        if len(confirmed_records) == 0:
            return RecordMatchResult(
                status=RecordMatchStatus.AMBIGUOUS,
                reason=(
                    "Existing business key digest belongs to an unverified or ambiguous identity."
                ),
            )

        identity = confirmed_records[0]
        version_ids: list[str] = data["versions_by_record"].get(str(identity.record_id), [])
        if not version_ids:
            return RecordMatchResult(status=RecordMatchStatus.NEW_RECORD)

        versions = [
            deserialize_contract(RecordVersion, data["versions"][vid])
            for vid in version_ids
        ]
        versions.sort(key=lambda v: v.version_number)
        latest = versions[-1]

        if (
            latest.content_sha256 == content_sha256
            and latest.content_fingerprint_version == content_fingerprint_version
        ):
            return RecordMatchResult(
                status=RecordMatchStatus.EXACT_MATCH,
                matched_version=latest,
            )

        return RecordMatchResult(
            status=RecordMatchStatus.CONTENT_CHANGED,
            matched_version=latest,
            reason=(
                "Confirmed business key matched an existing record whose normalized content "
                "changed; explicit correction confirmation is required."
            ),
        )

    def register_identity(
        self,
        *,
        scope: RecordScope,
        identity: RecordIdentity,
    ) -> RecordIdentity:
        if identity.scope != scope:
            raise RegistryScopeError("Record identity scope does not match requested scope.")

        data = self._load(scope)
        rec_key = str(identity.record_id)
        if rec_key in data["identities"]:
            existing = deserialize_contract(RecordIdentity, data["identities"][rec_key])
            if existing != identity:
                raise RegistryConflictError(f"Conflicting identity for record '{rec_key}'.")
            return existing

        if (
            identity.status == RecordIdentityStatus.CONFIRMED
            and identity.business_key_sha256 is not None
        ):
            existing_for_bkey = data["business_key_index"].get(identity.business_key_sha256, [])
            for other_id in existing_for_bkey:
                other = deserialize_contract(RecordIdentity, data["identities"][other_id])
                if other.status == RecordIdentityStatus.CONFIRMED:
                    raise RegistryConflictError(
                        "A confirmed record identity already exists for this business key."
                    )

        data["identities"][rec_key] = serialize_contract(identity)
        if identity.business_key_sha256 is not None:
            bucket = data["business_key_index"].setdefault(identity.business_key_sha256, [])
            if rec_key not in bucket:
                bucket.append(rec_key)
        self._save(scope, data)
        return identity

    def register_version(
        self,
        *,
        scope: RecordScope,
        version: RecordVersion,
    ) -> RecordVersion:
        if version.identity.scope != scope:
            raise RegistryScopeError("Record version scope does not match requested scope.")

        data = self._load(scope)
        rec_key = str(version.identity.record_id)
        ver_key = str(version.record_version_id)

        if rec_key not in data["identities"]:
            raise RegistryNotFoundError(
                f"Record identity '{rec_key}' must be registered before registering a version."
            )
        registered_identity = deserialize_contract(RecordIdentity, data["identities"][rec_key])
        if registered_identity != version.identity:
            raise RegistryConflictError("Record version identity does not match registered record.")

        if ver_key in data["versions"]:
            existing = deserialize_contract(RecordVersion, data["versions"][ver_key])
            if existing != version:
                raise RegistryConflictError(f"Conflicting record version '{ver_key}'.")
            return existing

        existing_ver_ids: list[str] = data["versions_by_record"].get(rec_key, [])
        existing_versions = sorted(
            (
                deserialize_contract(RecordVersion, data["versions"][vid])
                for vid in existing_ver_ids
            ),
            key=lambda v: v.version_number,
        )

        if version.version_number == 1:
            if existing_versions:
                first = existing_versions[0]
                if (
                    first.content_sha256 == version.content_sha256
                    and first.content_fingerprint_version == version.content_fingerprint_version
                ):
                    return first
                raise RegistryConflictError("Version 1 already exists with different content.")
        else:
            if not existing_versions:
                raise RegistryConflictError("Cannot register updated version without version 1.")
            latest = existing_versions[-1]
            version.validate_previous(latest)

        data["versions"][ver_key] = serialize_contract(version)
        data["versions_by_record"].setdefault(rec_key, []).append(ver_key)
        self._save(scope, data)
        return version

    def get_identity(
        self,
        *,
        scope: RecordScope,
        record_id: UUID,
    ) -> RecordIdentity:
        data = self._load(scope)
        key = str(record_id)
        if key not in data["identities"]:
            raise RegistryNotFoundError(f"Record identity '{record_id}' not found.")
        return deserialize_contract(RecordIdentity, data["identities"][key])

    def get_version(
        self,
        *,
        scope: RecordScope,
        record_version_id: UUID,
    ) -> RecordVersion:
        data = self._load(scope)
        key = str(record_version_id)
        if key not in data["versions"]:
            raise RegistryNotFoundError(f"Record version '{record_version_id}' not found.")
        return deserialize_contract(RecordVersion, data["versions"][key])

    def get_versions(
        self,
        *,
        scope: RecordScope,
        record_id: UUID,
    ) -> tuple[RecordVersion, ...]:
        data = self._load(scope)
        rec_key = str(record_id)
        if rec_key not in data["identities"]:
            raise RegistryNotFoundError(f"Record identity '{record_id}' not found.")
        ver_ids = data["versions_by_record"].get(rec_key, [])
        versions = [
            deserialize_contract(RecordVersion, data["versions"][vid])
            for vid in ver_ids
        ]
        versions.sort(key=lambda v: v.version_number)
        return tuple(versions)

    def register_membership(
        self,
        *,
        scope: RecordScope,
        manifest: DatasetMembershipManifest,
    ) -> None:
        if manifest.scope != scope:
            raise RegistryScopeError("Membership manifest scope does not match requested scope.")

        data = self._load(scope)
        rev_key = str(manifest.dataset_revision_id)
        if rev_key in data["memberships"]:
            existing = deserialize_contract(DatasetMembershipManifest, data["memberships"][rev_key])
            if existing != manifest:
                raise RegistryConflictError(
                    f"Conflicting membership manifest for revision '{rev_key}'."
                )
            return

        for membership in manifest.memberships:
            ver_key = str(membership.record_version_id)
            if ver_key not in data["versions"]:
                raise RegistryNotFoundError(
                    f"Membership references unregistered version '{ver_key}'."
                )
            version = deserialize_contract(RecordVersion, data["versions"][ver_key])
            membership.validate_record_version(version)

        data["memberships"][rev_key] = serialize_contract(manifest)
        self._save(scope, data)

    def get_membership(
        self,
        *,
        scope: RecordScope,
        dataset_revision_id: UUID,
    ) -> DatasetMembershipManifest:
        data = self._load(scope)
        rev_key = str(dataset_revision_id)
        if rev_key not in data["memberships"]:
            raise RegistryNotFoundError(
                f"Membership manifest for dataset revision '{dataset_revision_id}' not found."
            )
        return deserialize_contract(DatasetMembershipManifest, data["memberships"][rev_key])
