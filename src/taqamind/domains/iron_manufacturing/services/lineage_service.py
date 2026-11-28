"""Record lineage registration, identity matching, and correction handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import pandas as pd

from taqamind.core.contracts import DatasetContract, DatasetPurpose, DataType
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.dataset import DatasetArtifact
from taqamind.core.interfaces.dataset_registry import (
    DatasetMembershipManifest,
    RecordLineageRegistry,
)
from taqamind.core.interfaces.lineage import (
    DatasetRecordMembership,
    FingerprintField,
    RecordIdentity,
    RecordIdentityStatus,
    RecordMatchStatus,
    RecordScope,
    RecordVersion,
    calculate_record_fingerprint,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage


def build_record_scope(
    context: PipelineContext,
    *,
    identity_namespace: str = "production_record",
    identity_version: str = "1.0.0",
) -> RecordScope:
    """Create the deterministic Core-aligned RecordScope for a PipelineContext."""
    return RecordScope(
        factory_id=context.factory_id,
        domain=context.domain,
        process=context.process,
        dataset_id=context.dataset_id,
        identity_namespace=identity_namespace,
        identity_version=identity_version,
    )


def read_artifact_df(
    storage: DatasetArtifactStorage,
    factory_id: UUID,
    artifact: DatasetArtifact,
) -> pd.DataFrame:
    """Read a verified CSV dataset artifact from storage into a DataFrame."""
    with storage.open_verified(factory_id=factory_id, reference=artifact.reference) as stream:
        return pd.read_csv(stream, keep_default_na=True, low_memory=False)


@dataclass(frozen=True)
class LineageResolutionSummary:
    """Audit summary of record lineage matching for one dataset revision."""

    manifest: DatasetMembershipManifest
    new_records: int
    exact_matches: int
    corrected_records: int
    unverified_or_ambiguous: int
    row_match_statuses: tuple[RecordMatchStatus, ...]
    conflicts: tuple[str, ...]


def _normalize_fingerprint_value(
    value: Any, data_type: DataType
) -> str | int | float | bool | None:
    if pd.isna(value):
        return None
    if data_type == DataType.INTEGER:
        return int(round(float(value)))
    if data_type == DataType.FLOAT:
        return round(float(value), 8)
    if data_type == DataType.BOOLEAN:
        return bool(value)
    return str(value).strip()


def _row_fingerprints(
    row: pd.Series,
    contract: DatasetContract,
    *,
    purpose: DatasetPurpose,
) -> tuple[str | None, str]:
    """Compute normalized business-key digest and canonical content digest for one row."""
    bkey_fields: list[FingerprintField] = []
    for key_col in contract.business_keys:
        if key_col not in row.index or pd.isna(row[key_col]) or str(row[key_col]).strip() == "":
            bkey_fields = []
            break
        feat = contract.get_feature(key_col)
        bkey_fields.append(
            FingerprintField(
                name=key_col,
                value=_normalize_fingerprint_value(row[key_col], feat.data_type),
            )
        )

    bkey_sha = (
        calculate_record_fingerprint(
            tuple(bkey_fields),
            namespace=f"{contract.domain}.{contract.process}.business_key",
            version="1.0.0",
        )
        if bkey_fields
        else None
    )

    content_fields: list[FingerprintField] = []
    target_name = contract.target.feature.name
    for col in contract.all_columns:
        if purpose == DatasetPurpose.PREDICTION and col.name == target_name:
            continue
        if col.name not in row.index:
            continue
        content_fields.append(
            FingerprintField(
                name=col.name,
                value=_normalize_fingerprint_value(row[col.name], col.data_type),
            )
        )

    content_sha = calculate_record_fingerprint(
        tuple(content_fields),
        namespace=f"{contract.domain}.{contract.process}.content",
        version="1.0.0",
    )
    return bkey_sha, content_sha


def register_dataset_lineage(
    *,
    context: PipelineContext,
    artifact: DatasetArtifact,
    df: pd.DataFrame,
    contract: DatasetContract,
    lineage_registry: RecordLineageRegistry,
    allow_confirmed_corrections: bool = False,
    correction_confirmed_by: str | None = None,
    correction_reason: str | None = None,
    register_new_records: bool = True,
) -> LineageResolutionSummary:
    """Match rows against registered lineage, handle confirmed corrections, and save membership."""
    scope = build_record_scope(context)
    memberships: list[DatasetRecordMembership] = []
    statuses: list[RecordMatchStatus] = []
    conflicts: list[str] = []

    new_count = 0
    exact_count = 0
    corrected_count = 0
    ambiguous_count = 0

    for row_pos in range(len(df)):
        row = df.iloc[row_pos]
        bkey_sha, content_sha = _row_fingerprints(row, contract, purpose=context.purpose)

        match = lineage_registry.match(
            scope=scope,
            business_key_sha256=bkey_sha,
            content_sha256=content_sha,
            content_fingerprint_version="1.0.0",
        )
        statuses.append(match.status)

        if match.status == RecordMatchStatus.EXACT_MATCH:
            assert match.matched_version is not None
            exact_count += 1
            version = match.matched_version

        elif match.status == RecordMatchStatus.NEW_RECORD:
            new_count += 1
            identity = RecordIdentity(
                scope=scope,
                business_key_sha256=bkey_sha,
                status=(
                    RecordIdentityStatus.CONFIRMED
                    if bkey_sha is not None
                    else RecordIdentityStatus.UNVERIFIED
                ),
            )
            if register_new_records:
                registered_identity = lineage_registry.register_identity(
                    scope=scope,
                    identity=identity,
                )
                version = lineage_registry.register_version(
                    scope=scope,
                    version=RecordVersion(
                        identity=registered_identity,
                        content_sha256=content_sha,
                        source_dataset_revision_id=artifact.metadata.revision_id,
                        version_number=1,
                    ),
                )
            else:
                # For read-only/prediction flows, still assign stable version for this revision
                registered_identity = lineage_registry.register_identity(
                    scope=scope,
                    identity=identity,
                )
                version = lineage_registry.register_version(
                    scope=scope,
                    version=RecordVersion(
                        identity=registered_identity,
                        content_sha256=content_sha,
                        source_dataset_revision_id=artifact.metadata.revision_id,
                        version_number=1,
                    ),
                )

        elif match.status == RecordMatchStatus.CONTENT_CHANGED:
            assert match.matched_version is not None
            prev = match.matched_version
            if context.purpose == DatasetPurpose.PREDICTION:
                # In prediction mode, reuse the stable business identity and latest registered
                # version without mutating historical training content versions!
                exact_count += 1
                version = prev
            elif allow_confirmed_corrections and correction_confirmed_by and correction_reason:
                corrected_count += 1
                new_ver = RecordVersion(
                    identity=prev.identity,
                    content_sha256=content_sha,
                    source_dataset_revision_id=artifact.metadata.revision_id,
                    version_number=prev.version_number + 1,
                    previous_version_id=prev.record_version_id,
                    revision_reason=correction_reason,
                    confirmed_by=correction_confirmed_by,
                )
                version = lineage_registry.register_version(scope=scope, version=new_ver)
            else:
                ambiguous_count += 1
                conflicts.append(
                    f"Row {row_pos}: business key matched existing record "
                    f"{prev.identity.record_id} with changed content; "
                    "explicit correction confirmation is required."
                )
                # Register an unverified placeholder only if not blocking? No — raise cleanly
                # so callers never silently overwrite or double-count corrected records.
                raise ValueError(conflicts[-1])

        else:
            # UNVERIFIED or AMBIGUOUS identity — never treat as confirmed new record!
            ambiguous_count += 1
            identity = RecordIdentity(
                scope=scope,
                business_key_sha256=bkey_sha,
                status=(
                    RecordIdentityStatus.AMBIGUOUS
                    if match.status == RecordMatchStatus.AMBIGUOUS
                    else RecordIdentityStatus.UNVERIFIED
                ),
            )
            registered_identity = lineage_registry.register_identity(scope=scope, identity=identity)
            version = lineage_registry.register_version(
                scope=scope,
                version=RecordVersion(
                    identity=registered_identity,
                    content_sha256=content_sha,
                    source_dataset_revision_id=artifact.metadata.revision_id,
                    version_number=1,
                ),
            )
            conflicts.append(f"Row {row_pos}: record identity status is '{match.status.value}'.")

        memberships.append(
            DatasetRecordMembership(
                dataset_revision_id=artifact.metadata.revision_id,
                scope=scope,
                row_position=row_pos,
                record_id=version.identity.record_id,
                record_version_id=version.record_version_id,
            )
        )

    manifest = DatasetMembershipManifest(
        dataset_revision_id=artifact.metadata.revision_id,
        scope=scope,
        memberships=tuple(memberships),
    )
    manifest.validate_artifact(artifact)
    lineage_registry.register_membership(scope=scope, manifest=manifest)

    return LineageResolutionSummary(
        manifest=manifest,
        new_records=new_count,
        exact_matches=exact_count,
        corrected_records=corrected_count,
        unverified_or_ambiguous=ambiguous_count,
        row_match_statuses=tuple(statuses),
        conflicts=tuple(conflicts),
    )


def lookup_previously_registered_ids(
    *,
    lineage_registry: RecordLineageRegistry,
    context: PipelineContext,
    record_ids: tuple[UUID, ...],
) -> set[UUID]:
    """Return record IDs that already existed prior to the current dataset revision."""
    scope = build_record_scope(context)
    prior: set[UUID] = set()
    for rec_id in record_ids:
        versions = lineage_registry.get_versions(scope=scope, record_id=rec_id)
        if any(v.source_dataset_revision_id != context.dataset_revision_id for v in versions):
            prior.add(rec_id)
    return prior
