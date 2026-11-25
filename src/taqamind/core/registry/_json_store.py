"""Internal atomic JSON persistence helpers for local registries."""

import os
import tempfile
from pathlib import Path

from pydantic import BaseModel


def safe_resolve(root: Path, *parts: str) -> Path:
    """Resolve a relative path inside root and reject path traversal."""
    resolved_root = root.resolve()
    target = resolved_root.joinpath(*parts).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Resolved registry path escapes root directory.") from exc
    return target


def atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically via temporary file and os.replace."""
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


def load_json_model[TModel: BaseModel](path: Path, model_type: type[TModel]) -> TModel:
    """Load and validate a Pydantic model from a JSON file."""
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


def iter_json_models[TModel: BaseModel](
    directory: Path, model_type: type[TModel]
) -> list[TModel]:
    """Load all JSON model files directly under directory in sorted filename order."""
    if not directory.is_dir():
        return []
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix == ".json" and not path.name.startswith(".tmp-")
    )
    return [load_json_model(path, model_type) for path in paths]
