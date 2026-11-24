"""Serialization helpers for immutable pipeline contracts."""

from typing import Any

from pydantic import BaseModel


def serialize_contract(
    contract: BaseModel,
) -> dict[str, Any]:
    """Serialize a contract without computed fields."""

    return contract.model_dump(
        mode="json",
        round_trip=True,
    )


def deserialize_contract[T: BaseModel](
    model_type: type[T],
    payload: dict[str, Any],
) -> T:
    """Restore and validate a serialized contract."""

    return model_type.model_validate(payload)