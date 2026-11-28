"""Row-to-record fingerprinting, preflight validation, and lineage builders."""

from __future__ import annotations

from dataclasses import dataclass

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import (
    DatasetPurpose,
    DataType,
    FeatureOrigin,
    MissingValuePolicy,
)
from taqamind.core.ingestion.artifact_csv import read_verified_csv
from taqamind.core.ingestion.resource_policy import (
    CsvResourcePolicy,
    resolve_internal_csv_policy,
)
from taqamind.core.ingestion.safe_csv import SafeCsvReader
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.dataset import DatasetArtifact, DatasetState
from taqamind.core.interfaces.dataset_registry import (
    DatasetMembershipManifest,
    DatasetRecordMembership,
    RecordLineageRegistry,
)
from taqamind.core.interfaces.lineage import (
    FingerprintField,
    PredictionIdentityStatus,
    PredictionObservation,
    PredictionObservationBatch,
    RecordIdentity,
    RecordIdentityStatus,
    RecordMatchResult,
    RecordMatchStatus,
    RecordScope,
    RecordVersion,
    calculate_record_fingerprint,
)
from taqamind.core.interfaces.storage import DatasetArtifactStorage
from taqamind.core.lineage.canonicalization import (
    FINGERPRINT_VERSION_V1,
    FINGERPRINT_VERSION_V2,
    CanonicalizedColumn,
    canonicalize_column,
    hash_canonical_v2_row,
)
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


@dataclass(frozen=True, slots=True)
class _PreparedHistoricalRow:
    row_position: int
    biz_sha_v2: str | None
    content_sha: str
    match: RecordMatchResult


class RecordMembershipBuilder:
    """Map processed CSV rows to RecordIdentity, RecordVersion, or PredictionObservationBatch."""

    def __init__(
        self,
        storage: DatasetArtifactStorage,
        lineage_registry: RecordLineageRegistry,
        *,
        delimiter: str = ",",
        identity_namespace: str = "production_record",
        identity_version: str = "1.0.0",
        resource_policy: CsvResourcePolicy | None = None,
    ) -> None:
        self._storage = storage
        self._lineage = lineage_registry
        self._resource_policy = resolve_internal_csv_policy(resource_policy)
        self._reader = SafeCsvReader(delimiter=delimiter, resource_policy=self._resource_policy)
        self._converter = ContractTypeConverter()
        self._identity_namespace = identity_namespace
        self._identity_version = identity_version

    def _validate_scope_and_contract(
        self,
        *,
        scope: RecordScope,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        context: PipelineContext | None,
    ) -> None:
        if artifact.metadata.state == DatasetState.RAW:
            raise ValueError("Membership requires a processed dataset artifact.")
        if artifact.reference.factory_id != scope.factory_id:
            raise ValueError("Artifact reference factory_id does not match RecordScope.")
        if artifact.metadata.factory_id != scope.factory_id:
            raise ValueError("Artifact factory_id does not match RecordScope.")
        if artifact.metadata.domain != scope.domain or contract.domain != scope.domain:
            raise ValueError("Domain mismatch across RecordScope, artifact, and contract.")
        if artifact.metadata.process != scope.process or contract.process != scope.process:
            raise ValueError("Process mismatch across RecordScope, artifact, and contract.")
        if (
            artifact.metadata.dataset_id != scope.dataset_id
            or contract.dataset_id != scope.dataset_id
        ):
            raise ValueError("Dataset ID mismatch across RecordScope, artifact, and contract.")
        if artifact.metadata.contract_version != contract.version:
            raise ValueError("Artifact contract_version does not match DatasetContract.")
        if scope.identity_namespace != self._identity_namespace:
            raise ValueError("RecordScope identity_namespace does not match builder policy.")
        if scope.identity_version != self._identity_version:
            raise ValueError("RecordScope identity_version does not match builder policy.")
        if not contract.business_keys:
            raise ValueError("Record membership requires DatasetContract.business_keys.")
        if context is not None:
            scope.validate_context(context)
            artifact.metadata.validate_scope(context)
            context.validate_dataset_contract(contract)

    def _read_and_validate_frame(
        self,
        *,
        scope: RecordScope,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        purpose: DatasetPurpose,
    ):
        df = read_verified_csv(
            storage=self._storage,
            factory_id=scope.factory_id,
            reference=artifact.reference,
            reader=self._reader,
            policy=self._resource_policy,
        )
        if len(df) != artifact.metadata.row_count:
            raise ValueError("Actual CSV row count does not match artifact metadata row_count.")
        declared_cols = {c.name for c in artifact.metadata.columns}
        actual_cols = {str(c) for c in df.columns}
        if declared_cols != actual_cols:
            raise ValueError("CSV columns do not match artifact metadata column schema.")
        missing_keys = [k for k in contract.business_keys if k not in df.columns]
        if missing_keys:
            raise ValueError(f"Dataset is missing required business key columns: {missing_keys}.")
        col_map = contract.column_map
        for req_col in contract.required_columns(purpose):
            feat = col_map.get(req_col)
            if (
                feat is not None
                and feat.origin == FeatureOrigin.DERIVED
                and artifact.metadata.state != DatasetState.ENGINEERED
            ):
                continue
            if req_col not in df.columns:
                raise ValueError(f"Dataset is missing required contract column '{req_col}'.")
        return df

    def build_and_register(
        self,
        *,
        scope: RecordScope,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        context: PipelineContext | None = None,
        purpose: DatasetPurpose | None = None,
        confirm_updates_by: str | None = None,
        update_reason: str | None = None,
        content_fingerprint_version: str = FINGERPRINT_VERSION_V2,
    ) -> MembershipBuildSummary:
        """Preflight-validate, fingerprint, and register historical record lineage."""
        effective_purpose = context.purpose if context is not None else purpose
        if effective_purpose == DatasetPurpose.PREDICTION:
            raise ValueError(
                "HISTORICAL_REGISTRATION_WITH_PREDICTION_PURPOSE: "
                "Prediction datasets must use resolve_prediction_observations."
            )
        if content_fingerprint_version not in {FINGERPRINT_VERSION_V1, FINGERPRINT_VERSION_V2}:
            raise ValueError(
                f"UNSUPPORTED_FINGERPRINT_VERSION: '{content_fingerprint_version}'."
            )

        self._validate_scope_and_contract(
            scope=scope,
            artifact=artifact,
            contract=contract,
            context=context,
        )
        df = self._read_and_validate_frame(
            scope=scope,
            artifact=artifact,
            contract=contract,
            purpose=effective_purpose or DatasetPurpose.TRAINING,
        )

        col_map = contract.column_map
        sorted_keys = tuple(sorted(contract.business_keys))
        key_has_float = any(col_map[k].data_type == DataType.FLOAT for k in contract.business_keys)

        # Authoritative historical content excludes recomputable DERIVED features.
        content_cols = sorted(
            [
                str(col)
                for col in df.columns
                if str(col) in col_map and col_map[str(col)].origin != FeatureOrigin.DERIVED
            ]
        )
        if not content_cols:
            content_cols = list(sorted_keys)
        schema_prov = [[c, col_map[c].data_type.value] for c in content_cols]

        cols_to_canon = sorted(set(contract.business_keys) | set(content_cols))
        canon: dict[str, CanonicalizedColumn] = {
            c: canonicalize_column(df[c], col_map[c], converter=self._converter)
            for c in cols_to_canon
        }

        # Preflight all rows and perform read-only matching BEFORE any registry writes.
        prepared: list[_PreparedHistoricalRow] = []
        seen_batch_keys: set[str] = set()

        for row_idx in range(len(df)):
            missing_req_keys = [
                k
                for k in contract.business_keys
                if canon[k].values[row_idx] is None
                and (
                    col_map[k].missing_policy == MissingValuePolicy.REJECT
                    or not col_map[k].nullable
                )
            ]
            if missing_req_keys:
                raise ValueError(
                    f"Row {row_idx} has missing required business key(s): {missing_req_keys}."
                )

            has_missing_key = any(canon[k].values[row_idx] is None for k in contract.business_keys)
            if has_missing_key:
                biz_sha_v2 = None
                biz_sha_v1 = None
            else:
                biz_sha_v2 = hash_canonical_v2_row(
                    namespace="business_key",
                    field_pairs=[(k, canon[k].v2_tokens[row_idx]) for k in sorted_keys],
                    version=FINGERPRINT_VERSION_V2,
                )
                if biz_sha_v2 in seen_batch_keys:
                    raise ValueError(
                        "DUPLICATE_BUSINESS_KEY: "
                        f"Duplicate canonical business key at row {row_idx}."
                    )
                seen_batch_keys.add(biz_sha_v2)
                biz_sha_v1 = (
                    None
                    if key_has_float
                    else calculate_record_fingerprint(
                        tuple(
                            FingerprintField(name=k, value=canon[k].legacy_values[row_idx])
                            for k in contract.business_keys
                        ),
                        namespace="business_key",
                        version=FINGERPRINT_VERSION_V1,
                    )
                )

            if content_fingerprint_version == FINGERPRINT_VERSION_V2:
                content_sha = hash_canonical_v2_row(
                    namespace="record_content",
                    field_pairs=[(c, canon[c].v2_tokens[row_idx]) for c in content_cols],
                    schema_provenance=schema_prov,
                    version=FINGERPRINT_VERSION_V2,
                )
            else:
                content_sha = calculate_record_fingerprint(
                    tuple(
                        FingerprintField(name=c, value=canon[c].legacy_values[row_idx])
                        for c in content_cols
                    ),
                    namespace="record_content",
                    version=FINGERPRINT_VERSION_V1,
                )

            match = self._lineage.match(
                scope=scope,
                business_key_sha256=biz_sha_v2,
                content_sha256=content_sha,
                content_fingerprint_version=content_fingerprint_version,
            )

            # Safe legacy v1 compatibility check when v2 digest yields NEW_RECORD.
            if match.status == RecordMatchStatus.NEW_RECORD and biz_sha_v2 is not None:
                if biz_sha_v1 is not None and biz_sha_v1 != biz_sha_v2:
                    content_sha_v1 = calculate_record_fingerprint(
                        tuple(
                            FingerprintField(name=c, value=canon[c].legacy_values[row_idx])
                            for c in content_cols
                        ),
                        namespace="record_content",
                        version=FINGERPRINT_VERSION_V1,
                    )
                    legacy_match = self._lineage.match(
                        scope=scope,
                        business_key_sha256=biz_sha_v1,
                        content_sha256=content_sha_v1,
                        content_fingerprint_version=FINGERPRINT_VERSION_V1,
                    )
                    if legacy_match.status != RecordMatchStatus.NEW_RECORD:
                        if content_fingerprint_version != FINGERPRINT_VERSION_V1:
                            raise ValueError(
                                "LEGACY_MIGRATION_REQUIRED: Existing v1 record identity found; "
                                "explicit migration is required before v2 registration."
                            )
                        match = legacy_match

            if match.status == RecordMatchStatus.MIGRATION_REQUIRED:
                raise ValueError(
                    f"LEGACY_MIGRATION_REQUIRED: Row {row_idx} requires explicit "
                    "fingerprint migration."
                )
            if match.status == RecordMatchStatus.AMBIGUOUS:
                raise ValueError(
                    f"AMBIGUOUS_IDENTITY: Row {row_idx} matched ambiguous business key: "
                    f"{match.reason}"
                )
            if match.status == RecordMatchStatus.CONTENT_CHANGED:
                if match.reason and "older historical version" in match.reason:
                    raise ValueError(
                        f"HISTORICAL_REPLAY_REVIEW_REQUIRED: Row {row_idx} matches an older "
                        "historical version rather than the latest version."
                    )
                if confirm_updates_by is None:
                    raise ValueError(
                        f"Row {row_idx} modified existing record "
                        f"'{match.matched_version.identity.record_id}' "
                        "without explicit confirmation."
                    )

            prepared.append(
                _PreparedHistoricalRow(
                    row_position=row_idx,
                    biz_sha_v2=biz_sha_v2,
                    content_sha=content_sha,
                    match=match,
                )
            )

        # Registration pass: all rows validated and preflight-checked.
        memberships: list[DatasetRecordMembership] = []
        new_identities: list[RecordIdentity] = []
        new_versions: list[RecordVersion] = []
        counts = {
            "new": 0,
            "exact": 0,
            "changed": 0,
            "ambiguous": 0,
            "unverified": 0,
        }

        for item in prepared:
            match = item.match
            if match.status == RecordMatchStatus.NEW_RECORD:
                counts["new"] += 1
                identity = RecordIdentity(
                    scope=scope,
                    business_key_sha256=item.biz_sha_v2,
                    status=RecordIdentityStatus.CONFIRMED,
                )
                version = RecordVersion(
                    identity=identity,
                    content_sha256=item.content_sha,
                    content_fingerprint_version=content_fingerprint_version,
                    source_dataset_revision_id=artifact.metadata.revision_id,
                )
                new_identities.append(identity)
                new_versions.append(version)
            elif match.status == RecordMatchStatus.EXACT_MATCH:
                counts["exact"] += 1
                assert match.matched_version is not None
                version = match.matched_version
                identity = version.identity
            elif match.status == RecordMatchStatus.CONTENT_CHANGED:
                counts["changed"] += 1
                assert match.matched_version is not None
                prev = match.matched_version
                identity = prev.identity
                version = RecordVersion(
                    identity=identity,
                    content_sha256=item.content_sha,
                    content_fingerprint_version=content_fingerprint_version,
                    source_dataset_revision_id=artifact.metadata.revision_id,
                    version_number=prev.version_number + 1,
                    previous_version_id=prev.record_version_id,
                    revision_reason=update_reason or "Confirmed industrial record correction.",
                    confirmed_by=confirm_updates_by,
                )
                new_versions.append(version)
            else:
                counts["unverified"] += 1
                identity = RecordIdentity(
                    scope=scope,
                    business_key_sha256=item.biz_sha_v2,
                    status=RecordIdentityStatus.UNVERIFIED,
                )
                version = RecordVersion(
                    identity=identity,
                    content_sha256=item.content_sha,
                    content_fingerprint_version=content_fingerprint_version,
                    source_dataset_revision_id=artifact.metadata.revision_id,
                )
                new_identities.append(identity)
                new_versions.append(version)

            memberships.append(
                DatasetRecordMembership(
                    dataset_revision_id=artifact.metadata.revision_id,
                    scope=scope,
                    row_position=item.row_position,
                    record_id=identity.record_id,
                    record_version_id=version.record_version_id,
                )
            )

        manifest = DatasetMembershipManifest(
            dataset_revision_id=artifact.metadata.revision_id,
            scope=scope,
            memberships=tuple(memberships),
        )
        if hasattr(self._lineage, "register_batch"):
            self._lineage.register_batch(
                scope=scope,
                identities=tuple(new_identities),
                versions=tuple(new_versions),
                manifest=manifest,
            )
        else:
            for identity in new_identities:
                self._lineage.register_identity(scope=scope, identity=identity)
            for version in new_versions:
                self._lineage.register_version(scope=scope, version=version)
            self._lineage.register_membership(scope=scope, manifest=manifest)
        return MembershipBuildSummary(
            manifest=manifest,
            new_records=counts["new"],
            exact_matches=counts["exact"],
            content_changed=counts["changed"],
            ambiguous=counts["ambiguous"],
            unverified=counts["unverified"],
        )

    def resolve_prediction_observations(
        self,
        *,
        scope: RecordScope,
        artifact: DatasetArtifact,
        contract: DatasetContract,
        context: PipelineContext,
    ) -> PredictionObservationBatch:
        """Read-only prediction observation resolution without mutating historical lineage."""
        if context.purpose != DatasetPurpose.PREDICTION:
            raise ValueError(
                "Prediction observation resolution requires DatasetPurpose.PREDICTION."
            )
        self._validate_scope_and_contract(
            scope=scope,
            artifact=artifact,
            contract=contract,
            context=context,
        )
        df = self._read_and_validate_frame(
            scope=scope,
            artifact=artifact,
            contract=contract,
            purpose=DatasetPurpose.PREDICTION,
        )

        target_name = contract.target.feature.name
        if target_name in df.columns:
            raise ValueError("TARGET_PRESENT_IN_PREDICTION: Prediction input must exclude target.")

        col_map = contract.column_map
        for col in df.columns:
            col_s = str(col)
            feat = col_map.get(col_s)
            if feat is not None and not contract.is_available_at_prediction(feat):
                raise ValueError(
                    f"UNAVAILABLE_FEATURE_IN_PREDICTION: Column '{col_s}' is unavailable."
                )

        pred_cols = sorted(
            [
                str(col)
                for col in df.columns
                if str(col) in col_map
                and str(col) != target_name
                and contract.is_available_at_prediction(col_map[str(col)])
            ]
        )
        schema_prov = [[c, col_map[c].data_type.value] for c in pred_cols]
        cols_to_canon = sorted(set(contract.business_keys) | set(pred_cols))
        canon: dict[str, CanonicalizedColumn] = {
            c: canonicalize_column(df[c], col_map[c], converter=self._converter)
            for c in cols_to_canon
        }

        sorted_keys = tuple(sorted(contract.business_keys))
        key_has_float = any(col_map[k].data_type == DataType.FLOAT for k in contract.business_keys)
        seen_keys: set[str] = set()
        observations: list[PredictionObservation] = []
        dummy_sha = "0" * 64

        for row_idx in range(len(df)):
            pred_input_sha = hash_canonical_v2_row(
                namespace="prediction_input",
                field_pairs=[(c, canon[c].v2_tokens[row_idx]) for c in pred_cols],
                schema_provenance=schema_prov,
                version=FINGERPRINT_VERSION_V2,
            )

            has_missing_key = any(canon[k].values[row_idx] is None for k in contract.business_keys)
            if has_missing_key:
                observations.append(
                    PredictionObservation(
                        scope=scope,
                        prediction_dataset_revision_id=artifact.metadata.revision_id,
                        row_position=row_idx,
                        business_key_sha256=None,
                        prediction_input_sha256=pred_input_sha,
                        resolution_status=PredictionIdentityStatus.UNVERIFIED,
                        known_record_id=None,
                        reason="Business key missing; prediction identity unverified.",
                    )
                )
                continue

            biz_sha_v2 = hash_canonical_v2_row(
                namespace="business_key",
                field_pairs=[(k, canon[k].v2_tokens[row_idx]) for k in sorted_keys],
                version=FINGERPRINT_VERSION_V2,
            )
            if biz_sha_v2 in seen_keys:
                raise ValueError(
                    f"DUPLICATE_BUSINESS_KEY: Duplicate canonical business key at row {row_idx}."
                )
            seen_keys.add(biz_sha_v2)

            match = self._lineage.match(
                scope=scope,
                business_key_sha256=biz_sha_v2,
                content_sha256=dummy_sha,
                content_fingerprint_version=FINGERPRINT_VERSION_V2,
            )

            if match.status == RecordMatchStatus.NEW_RECORD and not key_has_float:
                biz_sha_v1 = calculate_record_fingerprint(
                    tuple(
                        FingerprintField(name=k, value=canon[k].legacy_values[row_idx])
                        for k in contract.business_keys
                    ),
                    namespace="business_key",
                    version=FINGERPRINT_VERSION_V1,
                )
                if biz_sha_v1 != biz_sha_v2:
                    legacy_match = self._lineage.match(
                        scope=scope,
                        business_key_sha256=biz_sha_v1,
                        content_sha256=dummy_sha,
                        content_fingerprint_version=FINGERPRINT_VERSION_V1,
                    )
                    if legacy_match.status != RecordMatchStatus.NEW_RECORD:
                        match = legacy_match

            if match.status in {
                RecordMatchStatus.EXACT_MATCH,
                RecordMatchStatus.CONTENT_CHANGED,
                RecordMatchStatus.MIGRATION_REQUIRED,
            }:
                assert match.matched_version is not None
                observations.append(
                    PredictionObservation(
                        scope=scope,
                        prediction_dataset_revision_id=artifact.metadata.revision_id,
                        row_position=row_idx,
                        business_key_sha256=biz_sha_v2,
                        prediction_input_sha256=pred_input_sha,
                        resolution_status=PredictionIdentityStatus.KNOWN_HISTORICAL,
                        known_record_id=match.matched_version.identity.record_id,
                    )
                )
            elif match.status == RecordMatchStatus.AMBIGUOUS:
                observations.append(
                    PredictionObservation(
                        scope=scope,
                        prediction_dataset_revision_id=artifact.metadata.revision_id,
                        row_position=row_idx,
                        business_key_sha256=biz_sha_v2,
                        prediction_input_sha256=pred_input_sha,
                        resolution_status=PredictionIdentityStatus.AMBIGUOUS,
                        known_record_id=None,
                        reason=match.reason or "Ambiguous historical business key.",
                    )
                )
            elif match.status == RecordMatchStatus.UNVERIFIED:
                observations.append(
                    PredictionObservation(
                        scope=scope,
                        prediction_dataset_revision_id=artifact.metadata.revision_id,
                        row_position=row_idx,
                        business_key_sha256=biz_sha_v2,
                        prediction_input_sha256=pred_input_sha,
                        resolution_status=PredictionIdentityStatus.UNVERIFIED,
                        known_record_id=None,
                        reason=match.reason or "Historical business key is unverified.",
                    )
                )
            else:
                observations.append(
                    PredictionObservation(
                        scope=scope,
                        prediction_dataset_revision_id=artifact.metadata.revision_id,
                        row_position=row_idx,
                        business_key_sha256=biz_sha_v2,
                        prediction_input_sha256=pred_input_sha,
                        resolution_status=PredictionIdentityStatus.NEW_OBSERVATION_CANDIDATE,
                        known_record_id=None,
                    )
                )

        batch = PredictionObservationBatch(
            scope=scope,
            prediction_dataset_revision_id=artifact.metadata.revision_id,
            observations=tuple(observations),
        )
        if hasattr(self._lineage, "register_prediction_observations"):
            return self._lineage.register_prediction_observations(scope=scope, batch=batch)
        return batch
