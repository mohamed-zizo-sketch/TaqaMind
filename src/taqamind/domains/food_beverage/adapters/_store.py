"""JSON persistence helpers for Food & Beverage P1.6-P1.8 local registries.

Recursively strips Pydantic @computed_field attributes (e.g. ValidationReport.status,
ValidationReport.summary) before serialization so strict extra='forbid' models reload cleanly.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def strip_computed_fields(obj: Any) -> Any:
    """Recursively serialize a BaseModel while excluding all @computed_field properties."""

    if isinstance(obj, BaseModel):
        computed = set(type(obj).model_computed_fields.keys())
        data: dict[str, Any] = {}
        for name in type(obj).model_fields:
            if name in computed:
                continue
            data[name] = strip_computed_fields(getattr(obj, name))
        return data
    if isinstance(obj, tuple):
        return [strip_computed_fields(x) for x in obj]
    if isinstance(obj, list):
        return [strip_computed_fields(x) for x in obj]
    if isinstance(obj, dict):
        return {k: strip_computed_fields(v) for k, v in obj.items()}
    return obj


def dump_model_json(model: BaseModel) -> str:
    """Serialize a BaseModel to deterministic JSON without computed fields."""

    cleaned = strip_computed_fields(model)
    return json.dumps(cleaned, indent=2, default=str)


def atomic_write_model(path: Path, model: BaseModel) -> None:
    """Atomically write a Pydantic model JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = dump_model_json(model)
    fd, tmp_raw = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    tmp_path = Path(tmp_raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def load_model_json[TModel: BaseModel](path: Path, model_cls: type[TModel]) -> TModel:
    """Load and strictly validate a Pydantic model from JSON."""

    return model_cls.model_validate_json(path.read_text(encoding="utf-8"))
