"""AUD-008: Row-to-record fingerprinting & DatasetMembershipManifest builder using Core P1.4."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import pandas as pd

from taqamind.core.contracts import DatasetContract
from taqamind.core.interfaces import (
    DatasetMembershipManifest,
    DatasetRecordMembership,
    FingerprintField,
    PipelineContext,
    RecordIdentity,
    RecordIdentityStatus,
    RecordMatchResult,
    RecordMatchStatus,
    RecordScope,
    RecordVersion,
    calculate_record_fingerprint,
)
from taqamind.core.lineage import LocalRecordLineageRegistry


@dataclass(frozen=True, slots=True)
class RowLineageSummary:
    """Resolved row-level record identities, versions, and manifest for a dataset revision."""

    scope: RecordScope
    manifest: DatasetMembershipManifest
    match_results: tuple[RecordMatchResult, ...]
    new_record_ids: tuple[UUID, ...]
    unchanged_record_ids: tuple[UUID, ...]
    corrected_record_ids: tuple[UUID, ...]
    ambiguous_row_indices: tuple[int, ...]
    row_to_record_id: dict[int, UUID]
    row_to_version_id: dict[int, UUID]


class FoodBeverageRowMembershipBuilder:
    """Build DatasetMembershipManifests backed by Shared Core LocalRecordLineageRegistry (P1.4).

    Candidate for migration to ``taqamind.core.lineage``.
    """

    def __init__(self, lineage_registry: LocalRecordLineageRegistry) -> None:
        self._lineage = lineage_registry

    def register_rows(
        self,
        *,
        context: PipelineContext,
        contract: DatasetContract,
        dataset_revision_id: UUID,
        df: pd.DataFrame,
    ) -> RowLineageSummary:
        scope = RecordScope(
            factory_id=context.factory_id,
            domain=context.domain,
            process=context.process,
            dataset_id=context.dataset_id,
            identity_namespace=f"{context.domain}_{context.process}",
        )

        bkeys = list(contract.business_keys)
        key_counts: dict[str, int] = {}
        row_bkey_sha: dict[int, str | None] = {}

        for idx, row in df.iterrows():
            if not bkeys or any(pd.isna(row[k]) for k in bkeys):
                row_bkey_sha[int(idx)] = None
            else:
                fields = tuple(
                    FingerprintField(name=k, value=str(row[k]))
                    for k in sorted(bkeys)
                )
                b_sha = calculate_record_fingerprint(
                    fields,
                    namespace=f"{scope.identity_namespace}_bkey",
                )
                row_bkey_sha[int(idx)] = b_sha
                key_counts[b_sha] = key_counts.get(b_sha, 0) + 1

        content_cols = sorted(c for c in df.columns if c in contract.column_map)
        memberships: list[DatasetRecordMembership] = []
        match_results: list[RecordMatchResult] = []
        new_ids: list[UUID] = []
        unchanged_ids: list[UUID] = []
        corrected_ids: list[UUID] = []
        ambiguous_indices: list[int] = []
        row_to_record_id: dict[int, UUID] = {}
        row_to_version_id: dict[int, UUID] = {}

        for pos, (idx, row) in enumerate(df.iterrows()):
            i = int(idx)
            b_sha = row_bkey_sha[i]

            content_fields = tuple(
                FingerprintField(
                    name=c,
                    value=None
                    if pd.isna(row[c])
                    else (
                        float(row[c])
                        if isinstance(row[c], (int, float))
                        and not isinstance(row[c], bool)
                        else str(row[c])
                    ),
                )
                for c in content_cols
            )
            content_sha = calculate_record_fingerprint(
                content_fields,
                namespace=f"{scope.identity_namespace}_content",
            )

            # Within-file duplicate business key -> AMBIGUOUS
            if b_sha is None or key_counts.get(b_sha, 0) > 1:
                ambiguous_indices.append(i)
                ident = RecordIdentity(
                    scope=scope,
                    business_key_sha256=b_sha,
                    status=RecordIdentityStatus.AMBIGUOUS,
                )
                self._lineage.register_identity(scope=scope, identity=ident)
                ver = RecordVersion(
                    identity=ident,
                    version_number=1,
                    content_sha256=content_sha,
                    source_dataset_revision_id=dataset_revision_id,
                )
                self._lineage.register_version(scope=scope, version=ver)
                match_res = RecordMatchResult(
                    status=RecordMatchStatus.AMBIGUOUS,
                    reason="Missing or duplicate business key within dataset.",
                )
                match_results.append(match_res)
                memberships.append(
                    DatasetRecordMembership(
                        dataset_revision_id=dataset_revision_id,
                        scope=scope,
                        row_position=pos,
                        record_id=ident.record_id,
                        record_version_id=ver.record_version_id,
                    )
                )
                row_to_record_id[i] = ident.record_id
                row_to_version_id[i] = ver.record_version_id
                continue

            match_res = self._lineage.match(
                scope=scope,
                business_key_sha256=b_sha,
                content_sha256=content_sha,
                content_fingerprint_version="1.0.0",
            )
            match_results.append(match_res)

            if match_res.status == RecordMatchStatus.NEW_RECORD:
                ident = RecordIdentity(
                    scope=scope,
                    business_key_sha256=b_sha,
                    status=RecordIdentityStatus.CONFIRMED,
                )
                self._lineage.register_identity(scope=scope, identity=ident)
                ver = RecordVersion(
                    identity=ident,
                    version_number=1,
                    content_sha256=content_sha,
                    source_dataset_revision_id=dataset_revision_id,
                )
                self._lineage.register_version(scope=scope, version=ver)
                new_ids.append(ident.record_id)
            elif match_res.status == RecordMatchStatus.EXACT_MATCH:
                assert match_res.matched_version is not None
                ver = match_res.matched_version
                ident = ver.identity
                unchanged_ids.append(ident.record_id)
            elif match_res.status == RecordMatchStatus.CONTENT_CHANGED:
                assert match_res.matched_version is not None
                prev = match_res.matched_version
                ident = prev.identity
                ver = RecordVersion(
                    identity=ident,
                    version_number=prev.version_number + 1,
                    content_sha256=content_sha,
                    previous_version_id=prev.record_version_id,
                    source_dataset_revision_id=dataset_revision_id,
                    revision_reason="Updated record content detected.",
                    confirmed_by="food_beverage_lineage",
                )
                self._lineage.register_version(scope=scope, version=ver)
                corrected_ids.append(ident.record_id)
            else:
                ambiguous_indices.append(i)
                ident = RecordIdentity(
                    scope=scope,
                    business_key_sha256=b_sha,
                    status=RecordIdentityStatus.AMBIGUOUS,
                )
                self._lineage.register_identity(scope=scope, identity=ident)
                ver = RecordVersion(
                    identity=ident,
                    version_number=1,
                    content_sha256=content_sha,
                    source_dataset_revision_id=dataset_revision_id,
                )
                self._lineage.register_version(scope=scope, version=ver)

            memberships.append(
                DatasetRecordMembership(
                    dataset_revision_id=dataset_revision_id,
                    scope=scope,
                    row_position=pos,
                    record_id=ident.record_id,
                    record_version_id=ver.record_version_id,
                )
            )
            row_to_record_id[i] = ident.record_id
            row_to_version_id[i] = ver.record_version_id

        manifest = DatasetMembershipManifest(
            dataset_revision_id=dataset_revision_id,
            scope=scope,
            memberships=tuple(memberships),
        )
        try:
            self._lineage.register_membership(scope=scope, manifest=manifest)
        except Exception:
            # Idempotent re-registration of identical manifest
            pass

        return RowLineageSummary(
            scope=scope,
            manifest=manifest,
            match_results=tuple(match_results),
            new_record_ids=tuple(new_ids),
            unchanged_record_ids=tuple(unchanged_ids),
            corrected_record_ids=tuple(corrected_ids),
            ambiguous_row_indices=tuple(ambiguous_indices),
            row_to_record_id=row_to_record_id,
            row_to_version_id=row_to_version_id,
        )
