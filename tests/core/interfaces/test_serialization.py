"""Round-trip serialization tests for shared contracts."""

import json

import pytest

from taqamind.core.interfaces import (
    deserialize_contract,
    serialize_contract,
)
from tests.core.interfaces.test_data_access import (
    make_development_access,
    make_test_access,
)
from tests.core.interfaces.test_evaluation import make_evaluation
from tests.core.interfaces.test_leakage import make_validation
from tests.core.interfaces.test_model_registry import make_validated
from tests.core.interfaces.test_prediction import make_prediction_batch
from tests.core.interfaces.test_training_registry import make_reuse_key


@pytest.mark.parametrize(
    "factory",
    [
        make_validation,
        make_evaluation,
        make_development_access,
        make_test_access,
        make_validated,
        make_prediction_batch,
        make_reuse_key,
    ],
)
def test_contract_json_round_trip(factory):
    original = factory()

    payload = serialize_contract(original)

    encoded = json.dumps(
        payload,
        allow_nan=False,
    )

    decoded = json.loads(encoded)

    restored = deserialize_contract(
        type(original),
        decoded,
    )

    assert restored == original


def test_nested_computed_fields_are_excluded():
    original = make_validation()

    payload = serialize_contract(original)

    assert "status" not in payload["report"]

    assert "summary" not in payload["report"]

    assert "has_blocking_issues" not in payload["report"]

    nested_report = payload["readiness"]["result"]["report"]

    assert "summary" not in nested_report

    restored = deserialize_contract(
        type(original),
        payload,
    )

    assert restored == original

    assert restored.eligible_for_training