"""Industrial record identity and version lineage contracts."""

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from taqamind.core.contracts.feature import (
    FeatureName,
    NonEmptyString,
)
from taqamind.core.contracts.target import ContractVersion
from taqamind.core.interfaces.artifacts import (
    SHA256Digest,
    calculate_bytes_sha256,
)
from taqamind.core.interfaces.context import PipelineContext

FingerprintValue = str | int | float | bool | None


class FingerprintField(BaseModel):
    """One canonical field used to calculate a fingerprint."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    name: FeatureName

    value: FingerprintValue


def calculate_record_fingerprint(
    fields: tuple[FingerprintField, ...],
    *,
    namespace: str,
    version: str = "1.0.0",
) -> str:
    """Hash deterministic, already-normalized record fields."""

    if not fields:
        raise ValueError(
            "Fingerprint requires at least one field."
        )

    if not namespace.strip():
        raise ValueError(
            "Fingerprint namespace cannot be empty."
        )

    names = [field.name for field in fields]

    if len(names) != len(set(names)):
        raise ValueError(
            "Fingerprint field names must be unique."
        )

    payload = {
        "namespace": namespace,
        "version": version,
        "fields": {
            field.name: field.value
            for field in fields
        },
    }

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    return calculate_bytes_sha256(canonical)


class RecordScope(BaseModel):
    """Identity namespace for registered industrial records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    factory_id: UUID

    domain: FeatureName

    process: FeatureName

    dataset_id: FeatureName

    identity_namespace: FeatureName

    identity_version: ContractVersion = "1.0.0"

    def validate_context(
        self,
        context: PipelineContext,
    ) -> None:
        fields = (
            "factory_id",
            "domain",
            "process",
            "dataset_id",
        )

        for field in fields:
            if getattr(self, field) != getattr(context, field):
                raise ValueError(
                    f"Record scope mismatch for '{field}'."
                )


class RecordIdentityStatus(StrEnum):
    """Reliability of an industrial record identity."""

    CONFIRMED = "confirmed"

    UNVERIFIED = "unverified"

    AMBIGUOUS = "ambiguous"


class RecordIdentity(BaseModel):
    """Stable identity of one industrial business record."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    record_id: UUID = Field(default_factory=uuid4)

    scope: RecordScope

    business_key_sha256: SHA256Digest | None = None

    status: RecordIdentityStatus

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @property
    def eligible_for_automatic_matching(self) -> bool:
        return (
            self.status == RecordIdentityStatus.CONFIRMED
            and self.business_key_sha256 is not None
        )

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.created_at.utcoffset() is None:
            raise ValueError(
                "Record timestamp must be timezone-aware."
            )

        if (
            self.status == RecordIdentityStatus.CONFIRMED
            and self.business_key_sha256 is None
        ):
            raise ValueError(
                "Confirmed record requires a business key fingerprint."
            )

        return self


class RecordVersion(BaseModel):
    """Immutable content version of a registered record."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    record_version_id: UUID = Field(
        default_factory=uuid4,
    )

    identity: RecordIdentity

    content_sha256: SHA256Digest

    content_fingerprint_version: ContractVersion = "1.0.0"

    source_dataset_revision_id: UUID

    version_number: int = Field(
        default=1,
        ge=1,
    )

    previous_version_id: UUID | None = None

    revision_reason: NonEmptyString | None = None

    confirmed_by: NonEmptyString | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @model_validator(mode="after")
    def validate_version(self) -> Self:
        if self.created_at.utcoffset() is None:
            raise ValueError(
                "Record version timestamp must be timezone-aware."
            )

        if self.created_at < self.identity.created_at:
            raise ValueError(
                "Record version cannot precede record identity."
            )

        if self.version_number == 1:
            if self.previous_version_id is not None:
                raise ValueError(
                    "First record version cannot have a parent."
                )

        else:
            if self.previous_version_id is None:
                raise ValueError(
                    "Updated record version requires a parent."
                )

            if not self.identity.eligible_for_automatic_matching:
                raise ValueError(
                    "Record updates require a confirmed identity."
                )

            if (
                self.revision_reason is None
                or self.confirmed_by is None
            ):
                raise ValueError(
                    "Record updates require a reason and confirmation."
                )

        if self.previous_version_id == self.record_version_id:
            raise ValueError(
                "Record version cannot reference itself."
            )

        return self

    def validate_previous(
        self,
        previous: "RecordVersion",
    ) -> None:
        """Validate a proposed update against its previous version."""

        if self.previous_version_id != previous.record_version_id:
            raise ValueError(
                "Previous record version mismatch."
            )

        if self.identity != previous.identity:
            raise ValueError(
                "Record identity cannot change between versions."
            )

        if self.version_number != previous.version_number + 1:
            raise ValueError(
                "Record versions must advance sequentially."
            )

        if self.created_at < previous.created_at:
            raise ValueError(
                "Record version cannot precede its parent."
            )

        if (
            self.content_fingerprint_version
            != previous.content_fingerprint_version
        ):
            raise ValueError(
                "Fingerprint version changes require explicit migration."
            )

        if self.content_sha256 == previous.content_sha256:
            raise ValueError(
                "Unchanged content must reuse the existing version."
            )


class RecordMatchStatus(StrEnum):
    """Outcome of matching an incoming industrial record."""

    NEW_RECORD = "new_record"

    EXACT_MATCH = "exact_match"

    CONTENT_CHANGED = "content_changed"

    AMBIGUOUS = "ambiguous"

    UNVERIFIED = "unverified"


class RecordMatchResult(BaseModel):
    """Decision metadata for an incoming record."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: RecordMatchStatus

    matched_version: RecordVersion | None = None

    reason: NonEmptyString | None = None

    @property
    def requires_review(self) -> bool:
        return self.status in {
            RecordMatchStatus.CONTENT_CHANGED,
            RecordMatchStatus.AMBIGUOUS,
            RecordMatchStatus.UNVERIFIED,
        }

    @model_validator(mode="after")
    def validate_match(self) -> Self:
        matched_statuses = {
            RecordMatchStatus.EXACT_MATCH,
            RecordMatchStatus.CONTENT_CHANGED,
        }

        if self.status in matched_statuses:
            if self.matched_version is None:
                raise ValueError(
                    "Matched record requires an existing version."
                )

            if not (
                self.matched_version.identity
                .eligible_for_automatic_matching
            ):
                raise ValueError(
                    "Matched record requires a confirmed identity."
                )

        elif self.matched_version is not None:
            raise ValueError(
                "Unmatched record cannot expose a matched version."
            )

        if self.requires_review and self.reason is None:
            raise ValueError(
                "Record review requires a reason."
            )

        return self


class DatasetRecordMembership(BaseModel):
    """Link a physical dataset row to a registered record version."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    dataset_revision_id: UUID

    scope: RecordScope

    row_position: int = Field(ge=0)

    record_id: UUID

    record_version_id: UUID

    def validate_record_version(
        self,
        version: RecordVersion,
    ) -> None:
        if self.scope != version.identity.scope:
            raise ValueError(
                "Membership record scope mismatch."
            )

        if self.record_id != version.identity.record_id:
            raise ValueError(
                "Membership record identity mismatch."
            )

        if self.record_version_id != version.record_version_id:
            raise ValueError(
                "Membership record version mismatch."
            )