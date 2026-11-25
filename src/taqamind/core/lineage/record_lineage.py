"""Local filesystem implementation of RecordLineageRegistry."""

from pathlib import Path
from uuid import UUID

from taqamind.core.contracts.target import ContractVersion
from taqamind.core.interfaces.artifacts import SHA256Digest
from taqamind.core.interfaces.dataset_registry import (
    DatasetMembershipManifest,
    DatasetRegistry,
    RecordLineageRegistry,
    RegistryConflictError,
    RegistryNotFoundError,
    RegistryScopeError,
)
from taqamind.core.interfaces.lineage import (
    RecordIdentity,
    RecordIdentityStatus,
    RecordMatchResult,
    RecordMatchStatus,
    RecordScope,
    RecordVersion,
)
from taqamind.core.registry._json_store import (
    atomic_write_text,
    iter_json_models,
    load_json_model,
    safe_resolve,
)


class LocalRecordLineageRegistry(RecordLineageRegistry):
    """Persist industrial record identities, versions, and dataset memberships."""

    def __init__(
        self,
        registry_dir: str | Path,
        *,
        dataset_registry: DatasetRegistry | None = None,
    ) -> None:
        self._root = Path(registry_dir).resolve()
        self._dataset_registry = dataset_registry

    def _scope_dir(self, scope: RecordScope) -> Path:
        return safe_resolve(
            self._root,
            "lineage",
            str(scope.factory_id),
            scope.domain,
            scope.process,
            scope.dataset_id,
            scope.identity_namespace,
            scope.identity_version,
        )

    def _identity_path(self, scope: RecordScope, record_id: UUID) -> Path:
        return safe_resolve(self._scope_dir(scope), "identities", f"{record_id}.json")

    def _version_path(self, scope: RecordScope, record_version_id: UUID) -> Path:
        return safe_resolve(
            self._scope_dir(scope), "versions", f"{record_version_id}.json"
        )

    def _membership_path(self, scope: RecordScope, dataset_revision_id: UUID) -> Path:
        return safe_resolve(
            self._scope_dir(scope), "memberships", f"{dataset_revision_id}.json"
        )

    def _load_all_identities(self, scope: RecordScope) -> list[RecordIdentity]:
        return iter_json_models(
            safe_resolve(self._scope_dir(scope), "identities"),
            RecordIdentity,
        )

    def match(
        self,
        *,
        scope: RecordScope,
        business_key_sha256: SHA256Digest | None,
        content_sha256: SHA256Digest,
        content_fingerprint_version: ContractVersion,
    ) -> RecordMatchResult:
        if business_key_sha256 is None:
            return RecordMatchResult(
                status=RecordMatchStatus.UNVERIFIED,
                reason="Business key fingerprint is unavailable; identity requires review.",
            )

        candidates = [
            identity
            for identity in self._load_all_identities(scope)
            if identity.business_key_sha256 == business_key_sha256
        ]

        if not candidates:
            return RecordMatchResult(status=RecordMatchStatus.NEW_RECORD)

        if len(candidates) > 1 or any(
            item.status == RecordIdentityStatus.AMBIGUOUS for item in candidates
        ):
            return RecordMatchResult(
                status=RecordMatchStatus.AMBIGUOUS,
                reason="Business key matches ambiguous or multiple record identities.",
            )

        identity = candidates[0]
        if identity.status != RecordIdentityStatus.CONFIRMED:
            return RecordMatchResult(
                status=RecordMatchStatus.UNVERIFIED,
                reason="Matched record identity is unverified.",
            )

        versions = self.get_versions(scope=scope, record_id=identity.record_id)
        if not versions:
            return RecordMatchResult(
                status=RecordMatchStatus.UNVERIFIED,
                reason="Confirmed identity has no registered content versions yet.",
            )

        latest = versions[-1]
        if latest.content_fingerprint_version != content_fingerprint_version:
            return RecordMatchResult(
                status=RecordMatchStatus.CONTENT_CHANGED,
                matched_version=latest,
                reason="Content fingerprint version changed; explicit review required.",
            )

        if latest.content_sha256 == content_sha256:
            return RecordMatchResult(
                status=RecordMatchStatus.EXACT_MATCH,
                matched_version=latest,
            )

        return RecordMatchResult(
            status=RecordMatchStatus.CONTENT_CHANGED,
            matched_version=latest,
            reason="Incoming content differs from the latest registered version.",
        )

    def register_identity(
        self,
        *,
        scope: RecordScope,
        identity: RecordIdentity,
    ) -> RecordIdentity:
        if identity.scope != scope:
            raise RegistryScopeError(
                "Record identity scope does not match requested scope."
            )

        path = self._identity_path(scope, identity.record_id)
        if path.is_file():
            existing = load_json_model(path, RecordIdentity)
            if existing == identity:
                return existing
            raise RegistryConflictError(
                "Conflicting metadata for existing record identity."
            )

        if identity.eligible_for_automatic_matching:
            for existing in self._load_all_identities(scope):
                if (
                    existing.eligible_for_automatic_matching
                    and existing.business_key_sha256 == identity.business_key_sha256
                ):
                    raise RegistryConflictError(
                        "Business key fingerprint already belongs to a confirmed identity."
                    )

        atomic_write_text(path, identity.model_dump_json(indent=2))
        return identity

    def register_version(
        self,
        *,
        scope: RecordScope,
        version: RecordVersion,
    ) -> RecordVersion:
        if version.identity.scope != scope:
            raise RegistryScopeError(
                "Record version scope does not match requested scope."
            )

        registered_identity = self.get_identity(
            scope=scope,
            record_id=version.identity.record_id,
        )
        if registered_identity != version.identity:
            raise RegistryConflictError(
                "Record version identity does not match registered identity."
            )

        version_path = self._version_path(scope, version.record_version_id)
        if version_path.is_file():
            existing = load_json_model(version_path, RecordVersion)
            if existing == version:
                return existing
            raise RegistryConflictError(
                "Conflicting metadata for existing record_version_id."
            )

        existing_versions = self.get_versions(
            scope=scope,
            record_id=version.identity.record_id,
        )

        if version.version_number == 1:
            if existing_versions:
                latest = existing_versions[-1]
                if (
                    latest.content_sha256 == version.content_sha256
                    and latest.content_fingerprint_version
                    == version.content_fingerprint_version
                ):
                    return latest
                raise RegistryConflictError(
                    "Initial version already exists with different content."
                )
        else:
            if not existing_versions:
                raise RegistryNotFoundError(
                    "Cannot register updated version without an existing initial version."
                )
            latest = existing_versions[-1]
            if (
                latest.content_sha256 == version.content_sha256
                and latest.content_fingerprint_version
                == version.content_fingerprint_version
            ):
                return latest
            if version.previous_version_id != latest.record_version_id:
                raise RegistryConflictError(
                    "Updated record version must extend the latest registered version."
                )
            try:
                version.validate_previous(latest)
            except ValueError as exc:
                raise RegistryConflictError(str(exc)) from exc

        atomic_write_text(version_path, version.model_dump_json(indent=2))
        return version

    def get_identity(
        self,
        *,
        scope: RecordScope,
        record_id: UUID,
    ) -> RecordIdentity:
        path = self._identity_path(scope, record_id)
        if not path.is_file():
            raise RegistryNotFoundError(
                f"Record identity '{record_id}' not found in scope."
            )
        identity = load_json_model(path, RecordIdentity)
        if identity.scope != scope:
            raise RegistryScopeError("Stored record identity scope mismatch.")
        return identity

    def get_version(
        self,
        *,
        scope: RecordScope,
        record_version_id: UUID,
    ) -> RecordVersion:
        path = self._version_path(scope, record_version_id)
        if not path.is_file():
            raise RegistryNotFoundError(
                f"Record version '{record_version_id}' not found in scope."
            )
        version = load_json_model(path, RecordVersion)
        if version.identity.scope != scope:
            raise RegistryScopeError("Stored record version scope mismatch.")
        return version

    def get_versions(
        self,
        *,
        scope: RecordScope,
        record_id: UUID,
    ) -> tuple[RecordVersion, ...]:
        self.get_identity(scope=scope, record_id=record_id)
        versions_dir = safe_resolve(self._scope_dir(scope), "versions")
        versions = [
            item
            for item in iter_json_models(versions_dir, RecordVersion)
            if item.identity.record_id == record_id
        ]
        versions.sort(key=lambda item: item.version_number)
        return tuple(versions)

    def register_membership(
        self,
        *,
        scope: RecordScope,
        manifest: DatasetMembershipManifest,
    ) -> None:
        if manifest.scope != scope:
            raise RegistryScopeError(
                "Membership manifest scope does not match requested scope."
            )

        if self._dataset_registry is not None:
            artifact = self._dataset_registry.get(
                factory_id=scope.factory_id,
                dataset_revision_id=manifest.dataset_revision_id,
            )
            try:
                manifest.validate_artifact(artifact)
            except ValueError as exc:
                raise RegistryConflictError(str(exc)) from exc

        for membership in manifest.memberships:
            version = self.get_version(
                scope=scope,
                record_version_id=membership.record_version_id,
            )
            try:
                membership.validate_record_version(version)
            except ValueError as exc:
                raise RegistryConflictError(str(exc)) from exc

        path = self._membership_path(scope, manifest.dataset_revision_id)
        if path.is_file():
            existing = load_json_model(path, DatasetMembershipManifest)
            if existing == manifest:
                return
            raise RegistryConflictError(
                "Conflicting membership manifest for existing dataset revision."
            )

        atomic_write_text(path, manifest.model_dump_json(indent=2))

    def get_membership(
        self,
        *,
        scope: RecordScope,
        dataset_revision_id: UUID,
    ) -> DatasetMembershipManifest:
        path = self._membership_path(scope, dataset_revision_id)
        if not path.is_file():
            raise RegistryNotFoundError(
                f"Membership manifest for '{dataset_revision_id}' not found in scope."
            )
        manifest = load_json_model(path, DatasetMembershipManifest)
        if manifest.scope != scope:
            raise RegistryScopeError("Stored membership manifest scope mismatch.")
        return manifest
