"""Lossless lexical CSV parsing before contract-driven type conversion."""

import csv
from io import StringIO

import pandas as pd


class SafeCsvError(ValueError):
    """A structural CSV failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SafeCsvReader:
    """Parse explicit CSV dialects without guessing types or missing tokens."""

    VERSION = "1.0.0"

    def __init__(
        self, *, delimiter: str = ",", max_rows: int | None = None
    ) -> None:
        if delimiter not in {",", ";", "\t", "|"}:
            raise ValueError("Unsupported CSV delimiter.")
        if max_rows is not None and max_rows <= 0:
            raise ValueError("max_rows must be positive when provided.")
        self._delimiter = delimiter
        self._max_rows = max_rows

    def read(self, raw_bytes: bytes) -> pd.DataFrame:
        """Return strings, treating only empty fields as missing values."""
        try:
            text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise SafeCsvError("CSV_DECODE_ERROR", "CSV must be valid UTF-8.") from exc
        if not text.strip():
            raise SafeCsvError("CSV_EMPTY_FILE", "Uploaded CSV file is empty.")
        if "\x00" in text:
            raise SafeCsvError("CSV_PARSE_ERROR", "CSV contains a NUL character.")

        reader = csv.reader(StringIO(text, newline=""), delimiter=self._delimiter, strict=True)
        try:
            headers = [header.strip() for header in next(reader)]
            if not headers or any(not header for header in headers):
                raise SafeCsvError("CSV_BLANK_HEADER", "CSV header contains blank column names.")
            if len(headers) != len(set(headers)):
                raise SafeCsvError(
                    "CSV_DUPLICATE_HEADER", "CSV header contains duplicate column names."
                )
            rows: list[list[str]] = []
            for row in reader:
                if len(row) != len(headers):
                    raise SafeCsvError(
                        "CSV_ROW_WIDTH_MISMATCH",
                        f"CSV record ending at line {reader.line_num} has {len(row)} "
                        f"fields; expected {len(headers)}.",
                    )
                if self._max_rows is not None and len(rows) >= self._max_rows:
                    raise SafeCsvError("CSV_ROW_LIMIT_EXCEEDED", "CSV row limit exceeded.")
                rows.append(row)
        except csv.Error as exc:
            raise SafeCsvError(
                "CSV_PARSE_ERROR", f"Invalid CSV near line {reader.line_num}."
            ) from exc

        frame = pd.DataFrame(rows, columns=headers, dtype="string")
        return frame.mask(frame.eq(""), pd.NA)
