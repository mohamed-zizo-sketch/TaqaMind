"""Row-to-record fingerprinting and DatasetMembershipManifest builder (AUD-008)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import DataType
from taqamind.core.ingestion.safe_csv import SafeCsvReader
from taqamind.core.interfaces.dataset import DatasetArtifact, DatasetState
from taqamind.core.interfaces.dataset_registry import (
    DatasetMembershipManifest,
    DatasetRecordMembership,
    RecordLineageRegistry,
)
from taqamind.core.interfaces.lineage import (
    FingerprintField,
    RecordIdentity,
    RecordIdentityStatus,
    RecordMatchStatus,
    RecordScope,
    RecordVersion,
    calculate_record_fingerprint,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage
from taqamind.core.preprocessing.type_conversion import ContractTypeConverter


@dataclass(frozen=True, slots=True)
class MembershipBuildSummary:
    """Audit summary produced while binding processed CSV rows to record lineage."""

    manifest: DatasetMembershipManifest
    new_records: int
    exact_matches: int
    content_changed: int
    ambiguous: int
    unverified: int


def _normalize_fingerprint_value(
    value: Any, data_type: DataType
) -> str | int | float | bool | None:
    if pd.isna(value):
        return None
    if data_type == DataType.INTEGER:
        return int(value)
    if data_type == DataType.FLOAT:
        return round(float(value), 9)
    if data_type == DataType.BOOLEAN:
        return bool(value)
    return str(value).strip()


class RecordMembershipBuilder:
    """Map processed CSV rows to RecordIdentity, RecordVersion, and DatasetMembershipManifest."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        lineage_registry: RecordLineageRegistry,
        *,
        delimiter: str = ",",
        identity_namespace: str = "production_record",
    ) -> None:
        self._storage = storage
        self._lineage = lineage_registry
        self._reader = SafeCsvReader(delimiter=delimiter)
        self._converter = ContractTypeConverter()
        self._identity_namespace = identity_namespace

    def build_and_register(
        self,
        *,
        scope: RecordScope,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        confirm_updates_by: str | None = None,
        update_reason: str | None = None,
    ) -> MembershipBuildSummary:
        """Fingerprint rows, register identities/versions, and persist membership manifest."""
        if artifact.metadata.state == DatasetState.RAW:
            raise ValueError("Membership requires a processed dataset artifact.")
        if artifact.metadata.factory_id != scope.factory_id:
            raise ValueError("Artifact factory_id does not match RecordScope.")
        if not contract.business_keys:
            raise ValueError("Record membership requires DatasetContract.business_keys.")

        with self._storage.open_verified(
            factory_id=scope.factory_id,
            reference=artifact.reference,
        ) as stream:
            df = self._reader.read(stream.read())

        col_map = contract.column_map
        content_cols = [col for col in df.columns if col in col_map]
        for col in content_cols:
            feat = col_map[col]
            if feat.data_type in {
                DataType.INTEGER,
                DataType.FLOAT,
                DataType.BOOLEAN,
                DataType.DATE,
                DataType.DATETIME,
            }:
                conv = self._converter.convert(df[col], data_type=feat.data_type)
                if conv.is_valid:
                    df[col] = conv.values

        memberships: list[DatasetRecordMembership] = []
        counts = {
            "new": 0,
            "exact": 0,
            "changed": 0,
            "ambiguous": 0,
            "unverified": 0,
        }

        for row_idx in range(len(df)):
            row = df.iloc[row_idx]
            key_fields = tuple(
                FingerprintField(
                    name=key_name,
                    value=_normalize_fingerprint_value(
                        row[key_name], col_map[key_name].data_type
                    ),
                )
                for key_name in contract.business_keys
            )
            has_missing_key = any(field.value is None for field in key_fields)
            business_key_sha = (
                None
                if has_missing_key
                else calculate_record_fingerprint(key_fields, namespace="business_key")
            )

            content_fields = tuple(
                FingerprintField(
                    name=col_name,
                    value=_normalize_fingerprint_value(
                        row[col_name], col_map[col_name].data_type
                    ),
                )
                for col_name in content_cols
            )
            content_sha = calculate_record_fingerprint(
                content_fields, namespace="record_content"
            )

            match = self._lineage.match(
                scope=scope,
                business_key_sha256=business_key_sha,
                content_sha256=content_sha,
                content_fingerprint_version="1.0.0",
            )

            if match.status == RecordMatchStatus.NEW_RECORD:
                counts["new"] += 1
                identity = RecordIdentity(
                    scope=scope,
                    business_key_sha256=business_key_sha,
                    status=RecordIdentityStatus.CONFIRMED,
                )
                self._lineage.register_identity(scope=scope, identity=identity)
                version = RecordVersion(
                    identity=identity,
                    content_sha256=content_sha,
                    source_dataset_revision_id=artifact.metadata.revision_id,
                )
                self._lineage.register_version(scope=scope, version=version)

            elif match.status == RecordMatchStatus.EXACT_MATCH:
                counts["exact"] += 1
                assert match.matched_version is not None
                version = match.matched_version
                identity = version.identity

            elif match.status == RecordMatchStatus.CONTENT_CHANGED:
                counts["changed"] += 1
                assert match.matched_version is not None
                prev = match.matched_version
                if confirm_updates_by is None:
                    raise ValueError(
                        f"Row {row_idx} modified existing record '{prev.identity.record_id}' "
                        "without explicit confirmation."
                    )
                identity = prev.identity
                version = RecordVersion(
                    identity=identity,
                    content_sha256=content_sha,
                    source_dataset_revision_id=artifact.metadata.revision_id,
                    version_number=prev.version_number + 1,
                    previous_version_id=prev.record_version_id,
                    revision_reason=update_reason or "Confirmed industrial record correction.",
                    confirmed_by=confirm_updates_by,
                )
                self._lineage.register_version(scope=scope, version=version)

            elif match.status == RecordMatchStatus.AMBIGUOUS:
                counts["ambiguous"] += 1
                raise ValueError(
                    f"Row {row_idx} matched ambiguous business key identity: {match.reason}"
                )
            else:
                counts["unverified"] += 1
                identity = RecordIdentity(
                    scope=scope,
                    business_key_sha256=business_key_sha,
                    status=RecordIdentityStatus.UNVERIFIED,
                )
                self._lineage.register_identity(scope=scope, identity=identity)
                version = RecordVersion(
                    identity=identity,
                    content_sha256=content_sha,
                    source_dataset_revision_id=artifact.metadata.revision_id,
                )
                self._lineage.register_version(scope=scope, version=version)

            memberships.append(
                DatasetRecordMembership(
                    dataset_revision_id=artifact.metadata.revision_id,
                    scope=scope,
                    row_position=row_idx,
                    record_id=identity.record_id,
                    record_version_id=version.record_version_id,
                )
            )

        manifest = DatasetMembershipManifest(
            dataset_revision_id=artifact.metadata.revision_id,
            scope=scope,
            memberships=tuple(memberships),
        )
        self._lineage.register_membership(scope=scope, manifest=manifest)
        return MembershipBuildSummary(
            manifest=manifest,
            new_records=counts["new"],
            exact_matches=counts["exact"],
            content_changed=counts["changed"],
            ambiguous=counts["ambiguous"],
            unverified=counts["unverified"],
        )
