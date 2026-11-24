"""Training, split and protected evaluation registry contracts."""

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, Self, runtime_checkable
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from taqamind.core.contracts.enums import DatasetPurpose
from taqamind.core.contracts.feature import FeatureName, NonEmptyString
from taqamind.core.contracts.target import ContractVersion
from taqamind.core.interfaces.artifacts import (
    SHA256Digest,
    calculate_bytes_sha256,
)
from taqamind.core.interfaces.candidates import TrainingRunResult
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.dataset_registry import (
    DatasetMembershipManifest,
)
from taqamind.core.interfaces.evaluation import (
    AcceptancePolicy,
    FinalEvaluationResult,
    FrozenModelSelection,
)
from taqamind.core.interfaces.leakage import (
    LeakageValidationResult,
    TrainingAuthorization,
)
from taqamind.core.interfaces.lineage import RecordScope
from taqamind.core.interfaces.splitting import SplitManifest
from taqamind.core.interfaces.training import TrainingConfiguration


def _hash_payload(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    return calculate_bytes_sha256(canonical)


class TrainingMode(StrEnum):
    """Explicit training operation types."""

    INITIAL = "initial"
    RETRAINING = "retraining"


class TrainingIntent(BaseModel):
    """Explicit request to create or update a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context: PipelineContext

    mode: TrainingMode

    requested_by: NonEmptyString

    reason: NonEmptyString

    previous_model_version_id: UUID | None = None

    @model_validator(mode="after")
    def validate_intent(self) -> Self:
        if self.context.purpose != DatasetPurpose.TRAINING:
            raise ValueError(
                "Training intent requires a training context."
            )

        if (
            self.mode == TrainingMode.INITIAL
            and self.previous_model_version_id is not None
        ):
            raise ValueError(
                "Initial training cannot reference a previous model."
            )

        if (
            self.mode == TrainingMode.RETRAINING
            and self.previous_model_version_id is None
        ):
            raise ValueError(
                "Retraining requires a previous model version."
            )

        return self


def calculate_training_configuration_fingerprint(
    configuration: TrainingConfiguration,
    policy: AcceptancePolicy,
) -> str:
    """Hash effective training settings rather than generated IDs."""

    payload = configuration.model_dump(
        mode="json",
        exclude={
            "configuration_id": True,
            "justification": True,
            "candidates": {
                "__all__": {"candidate_id": True},
            },
        },
    )

    for candidate in payload["candidates"]:
        candidate["parameters"] = sorted(
            candidate["parameters"],
            key=lambda parameter: parameter["name"],
        )

    payload["acceptance_policy"] = {
        "minimum_improvement_pct": policy.minimum_improvement_pct,
        "maximum_test_mae": policy.maximum_test_mae,
    }

    return _hash_payload(payload)


def calculate_split_fingerprint(
    manifest: SplitManifest,
    membership: DatasetMembershipManifest,
) -> str:
    """Hash exact split assignments and immutable record versions."""

    if membership.dataset_revision_id != manifest.dataset_revision_id:
        raise ValueError(
            "Membership and split dataset revisions must match."
        )

    membership.scope.validate_context(manifest.context)

    if membership.has_repeated_record_ids:
        raise ValueError(
            "Training membership contains repeated record identities."
        )

    versions = {
        item.record_id: item.record_version_id
        for item in membership.memberships
    }

    manifest.validate_record_coverage(tuple(versions))

    def encode_records(record_ids):
        return sorted(
            (str(record_id), str(versions[record_id]))
            for record_id in record_ids
        )

    payload = {
        "configuration": manifest.configuration.model_dump(
            mode="json",
            exclude={"justification"},
        ),
        "train": encode_records(manifest.train.record_ids),
        "validation": encode_records(manifest.validation.record_ids),
        "test": encode_records(manifest.test.record_ids),
        "excluded": encode_records(manifest.excluded_record_ids),
        "exclusion_reason": manifest.exclusion_reason,
    }

    return _hash_payload(payload)


class TrainingReuseKey(BaseModel):
    """Content-based identity of an equivalent training request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    factory_id: UUID

    domain: FeatureName

    process: FeatureName

    task_id: FeatureName

    dataset_id: FeatureName

    contract_version: ContractVersion

    task_version: ContractVersion

    target_version: ContractVersion

    mode: TrainingMode

    previous_model_version_id: UUID | None = None

    dataset_content_sha256: SHA256Digest

    contract_sha256: SHA256Digest

    split_sha256: SHA256Digest

    training_configuration_sha256: SHA256Digest

    implementation_sha256: SHA256Digest

    @property
    def fingerprint(self) -> str:
        return _hash_payload(self.model_dump(mode="json"))

    @classmethod
    def from_intent(
        cls,
        *,
        intent: TrainingIntent,
        dataset_content_sha256: SHA256Digest,
        contract_sha256: SHA256Digest,
        split_sha256: SHA256Digest,
        training_configuration_sha256: SHA256Digest,
        implementation_sha256: SHA256Digest,
    ) -> Self:
        context = intent.context

        return cls(
            factory_id=context.factory_id,
            domain=context.domain,
            process=context.process,
            task_id=context.task_id,
            dataset_id=context.dataset_id,
            contract_version=context.contract_version,
            task_version=context.task_version,
            target_version=context.target_version,
            mode=intent.mode,
            previous_model_version_id=intent.previous_model_version_id,
            dataset_content_sha256=dataset_content_sha256,
            contract_sha256=contract_sha256,
            split_sha256=split_sha256,
            training_configuration_sha256=training_configuration_sha256,
            implementation_sha256=implementation_sha256,
        )

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if (
            self.mode == TrainingMode.INITIAL
            and self.previous_model_version_id is not None
        ):
            raise ValueError("Initial training cannot have a previous model.")

        if (
            self.mode == TrainingMode.RETRAINING
            and self.previous_model_version_id is None
        ):
            raise ValueError("Retraining requires a previous model.")

        return self


class RecordUsageRole(StrEnum):
    """How a registered record was exposed to a model workflow."""

    TRAIN = "train"
    VALIDATION = "validation"
    PROTECTED_TEST = "protected_test"


class RecordUsage(BaseModel):
    """Immutable record exposure event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    usage_id: UUID = Field(default_factory=uuid4)

    scope: RecordScope

    record_id: UUID

    record_version_id: UUID

    execution_id: UUID

    split_id: UUID

    role: RecordUsageRole

    exposed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    @model_validator(mode="after")
    def validate_timestamp(self) -> Self:
        if self.exposed_at.utcoffset() is None:
            raise ValueError(
                "Record usage timestamp must be timezone-aware."
            )

        return self


class EvaluationReservation(BaseModel):
    """Reservation created before accessing protected test data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reservation_id: UUID = Field(default_factory=uuid4)

    scope: RecordScope

    execution_id: UUID

    split_id: UUID

    selection_id: UUID

    test_record_ids: tuple[UUID, ...] = Field(min_length=1)

    reserved_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    @classmethod
    def from_selection(
        cls,
        *,
        scope: RecordScope,
        selection: FrozenModelSelection,
        manifest: SplitManifest,
    ) -> Self:
        scope.validate_context(manifest.context)

        development = selection.training_run.development_split

        if development.split_id != manifest.split_id:
            raise ValueError("Evaluation split mismatch.")

        if development.context != manifest.context:
            raise ValueError("Evaluation context mismatch.")

        if development.dataset_revision_id != manifest.dataset_revision_id:
            raise ValueError("Evaluation dataset revision mismatch.")

        if (
            development.train != manifest.train
            or development.validation != manifest.validation
        ):
            raise ValueError("Evaluation development partition mismatch.")

        return cls(
            scope=scope,
            execution_id=manifest.context.execution_id,
            split_id=manifest.split_id,
            selection_id=selection.selection_id,
            test_record_ids=manifest.test.record_ids,
        )

    @model_validator(mode="after")
    def validate_reservation(self) -> Self:
        if self.reserved_at.utcoffset() is None:
            raise ValueError("Reservation timestamp must be timezone-aware.")

        if len(self.test_record_ids) != len(set(self.test_record_ids)):
            raise ValueError("Reserved test record IDs must be unique.")

        return self


def find_previously_exposed_test_records(
    *,
    manifest: SplitManifest,
    scope: RecordScope,
    history: tuple[RecordUsage, ...],
) -> tuple[UUID, ...]:
    """Find protected test identities appearing in prior ML history."""

    scope.validate_context(manifest.context)

    test_ids = set(manifest.test.record_ids)

    exposed_ids = set()

    for usage in history:
        if usage.scope != scope:
            raise ValueError(
                "Record usage history contains an incompatible scope."
            )

        if usage.record_id in test_ids:
            exposed_ids.add(usage.record_id)

    return tuple(sorted(exposed_ids, key=str))


@runtime_checkable
class SplitRegistry(Protocol):
    """Persist immutable split manifests and leakage assessments."""

    def register_split(
        self,
        *,
        factory_id: UUID,
        manifest: SplitManifest,
    ) -> SplitManifest:
        """Register or reuse an identical immutable split."""
        ...

    def get_split(
        self,
        *,
        factory_id: UUID,
        split_id: UUID,
    ) -> SplitManifest:
        ...

    def register_leakage_validation(
        self,
        *,
        factory_id: UUID,
        validation: LeakageValidationResult,
    ) -> None:
        ...

    def issue_training_authorization(
        self,
        *,
        factory_id: UUID,
        validation_id: UUID,
    ) -> TrainingAuthorization:
        """Issue authorization from verified registered metadata."""
        ...

    def verify_training_authorization(
        self,
        *,
        factory_id: UUID,
        authorization: TrainingAuthorization,
    ) -> None:
        """Verify authorization and recheck applicable leakage rules."""
        ...


@runtime_checkable
class RecordUsageRegistry(Protocol):
    """Track historical record exposure."""

    def record_usage(
        self,
        *,
        scope: RecordScope,
        usages: tuple[RecordUsage, ...],
    ) -> None:
        """Persist exposure before releasing data to ML consumers."""
        ...

    def get_history(
        self,
        *,
        scope: RecordScope,
        record_ids: tuple[UUID, ...],
    ) -> tuple[RecordUsage, ...]:
        ...


@runtime_checkable
class TrainingRegistry(Protocol):
    """Track training executions and equivalent completed results."""

    def register_intent(
        self,
        *,
        factory_id: UUID,
        intent: TrainingIntent,
    ) -> None:
        """Validate explicit training intent against model lifecycle."""
        ...

    def find_reusable_result(
        self,
        *,
        key: TrainingReuseKey,
    ) -> TrainingRunResult | None:
        """Return only verified, reusable completed training results."""
        ...

    def register_result(
        self,
        *,
        factory_id: UUID,
        key: TrainingReuseKey,
        result: TrainingRunResult,
    ) -> None:
        """Validate execution scope, authorization and reuse metadata."""
        ...

    def get_result(
        self,
        *,
        factory_id: UUID,
        training_run_id: UUID,
    ) -> TrainingRunResult:
        ...


@runtime_checkable
class EvaluationRegistry(Protocol):
    """Control protected test exposure and preserve final results."""

    def register_selection(
        self,
        *,
        factory_id: UUID,
        selection: FrozenModelSelection,
    ) -> None:
        ...

    def reserve_test_access(
        self,
        *,
        factory_id: UUID,
        reservation: EvaluationReservation,
    ) -> EvaluationReservation:
        """Atomically reserve an independently eligible test partition.

        Verify the registered split, selection, policy and record
        history before allowing access. Prevent competing reservations
        from consuming overlapping protected records.
        """
        ...

    def mark_test_exposed(
        self,
        *,
        factory_id: UUID,
        reservation_id: UUID,
    ) -> None:
        """Persist test exposure before releasing any protected data."""
        ...

    def register_evaluation(
        self,
        *,
        factory_id: UUID,
        reservation_id: UUID,
        evaluation: FinalEvaluationResult,
    ) -> None:
        """Persist one final result bound to its registered reservation."""
        ...

    def get_evaluation(
        self,
        *,
        factory_id: UUID,
        evaluation_id: UUID,
    ) -> FinalEvaluationResult:
        ...