"""Internal atomic JSON and path helpers for temporary textile adapters.

Replacement note: Mirrors Core `taqamind.core.registry._json_store` without importing
private Core modules (`_json_store`), so adapters remain isolated and replaceable.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python


def safe_resolve(root: Path, *parts: str) -> Path:
    """Resolve a path under root and reject path traversal."""
    resolved_root = root.resolve()
    target = resolved_root.joinpath(*parts).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Resolved adapter path escapes root directory.") from exc
    return target


def atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically using a temporary file and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    tmp_path = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _strip_computed_fields(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {
            field_name: _strip_computed_fields(getattr(value, field_name))
            for field_name in type(value).model_fields
        }
    if isinstance(value, (list, tuple)):
        return [_strip_computed_fields(item) for item in value]
    if isinstance(value, dict):
        return {k: _strip_computed_fields(v) for k, v in value.items()}
    return value


def dump_json_model(model: BaseModel) -> str:
    """Serialize a Pydantic model excluding @computed_field attributes (strict round-trip)."""
    stripped = _strip_computed_fields(model)
    return json.dumps(to_jsonable_python(stripped), indent=2)


def load_json_model[TModel: BaseModel](path: Path, model_type: type[TModel]) -> TModel:
    """Load and validate a Pydantic model from a JSON file."""
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


def iter_json_models[TModel: BaseModel](
    directory: Path, model_type: type[TModel]
) -> list[TModel]:
    """Load all JSON files directly inside directory sorted by filename."""
    if not directory.is_dir():
        return []
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix == ".json" and not path.name.startswith(".tmp-")
    )
    return [load_json_model(path, model_type) for path in paths]
