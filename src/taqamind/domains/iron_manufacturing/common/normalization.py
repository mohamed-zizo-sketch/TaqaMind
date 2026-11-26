"""Deterministic CSV reading, header normalization, and spreadsheet export safety."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

import pandas as pd

from taqamind.domains.iron_manufacturing.config import MAX_UPLOAD_SIZE_BYTES

_ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def normalize_header(raw: str) -> str:
    """Normalize English/Arabic column headers deterministically for alias matching."""
    text = str(raw).strip().casefold()
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = re.sub(r"[()\[\]{}/\\:;,\-]+", " ", text)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sanitize_spreadsheet_cell(value: Any) -> Any:
    """Prevent CSV formula injection on exported string values."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def safe_export_csv(df: pd.DataFrame, destination: Path) -> Path:
    """Write a DataFrame to CSV safely with UTF-8 encoding and formula neutralization."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe_df = df.copy()
    for col in safe_df.select_dtypes(include=["object", "string"]).columns:
        safe_df[col] = safe_df[col].map(sanitize_spreadsheet_cell)
    safe_df.to_csv(destination, index=False, encoding="utf-8")
    return destination


def inspect_raw_csv_headers(raw_bytes: bytes) -> tuple[list[str], str, str]:
    """Decode raw CSV bytes and inspect header row for malformed/duplicate/blank names."""
    if not raw_bytes:
        raise ValueError("CSV content is empty.")
    if len(raw_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError(
            f"CSV payload size ({len(raw_bytes)} bytes) exceeds limit "
            f"({MAX_UPLOAD_SIZE_BYTES} bytes)."
        )

    head = raw_bytes[:4096]
    if b"\x00" in head or head.lstrip().lower().startswith((b"<html", b"<!doctype")):
        raise ValueError("Invalid CSV payload: binary null bytes or HTML markup detected.")

    last_err: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1256", "latin-1"):
        try:
            text_head = head.decode(encoding, errors="strict")
            first_line = text_head.splitlines()[0] if text_head.splitlines() else ""
            counts = {delim: first_line.count(delim) for delim in (",", ";", "\t")}
            delimiter = max(counts, key=lambda d: counts[d]) if max(counts.values()) > 0 else ","
            full_text = raw_bytes.decode(encoding, errors="strict")
            reader = csv.reader(io.StringIO(full_text), delimiter=delimiter)
            headers = next(reader, None)
            if headers is None:
                raise ValueError("CSV file has no header row.")
            return headers, encoding, delimiter
        except ValueError:
            raise
        except Exception as exc:
            last_err = exc

    raise ValueError(f"Unable to decode CSV payload safely: {last_err}")


def safe_read_csv_bytes(raw_bytes: bytes) -> tuple[pd.DataFrame, str, str, list[str]]:
    """Parse raw CSV bytes safely while checking for malformed/duplicate/blank headers."""
    headers, encoding, delimiter = inspect_raw_csv_headers(raw_bytes)
    stripped = [h.strip() for h in headers]
    if any(not h for h in stripped):
        raise ValueError("Malformed CSV header: blank column header detected.")
    if len(stripped) != len(set(stripped)):
        seen: set[str] = set()
        dups: list[str] = []
        for h in stripped:
            if h in seen and h not in dups:
                dups.append(h)
            seen.add(h)
        raise ValueError(f"Malformed CSV header: duplicate column headers detected: {dups}.")

    df = pd.read_csv(
        io.BytesIO(raw_bytes),
        sep=delimiter,
        encoding=encoding,
        low_memory=False,
    )
    df.columns = [str(col).strip() for col in df.columns]
    return df, encoding, delimiter, stripped


def safe_read_csv(path: Path) -> tuple[pd.DataFrame, str, str, list[str]]:
    """Read a CSV file from disk using size, payload, and header validations."""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    size = path.stat().st_size
    if size > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError(
            f"CSV file size ({size} bytes) exceeds limit ({MAX_UPLOAD_SIZE_BYTES} bytes)."
        )
    return safe_read_csv_bytes(path.read_bytes())
