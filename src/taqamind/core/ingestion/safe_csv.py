"""Lossless lexical CSV parsing before contract-driven type conversion."""

import csv
from copy import copy
from io import StringIO
from typing import BinaryIO

import pandas as pd

from taqamind.core.ingestion.resource_policy import CsvResourcePolicy

READ_CHUNK_SIZE = 1024 * 1024


class SafeCsvError(ValueError):
    """A structural CSV failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SafeCsvReader:
    """Parse explicit CSV dialects with configurable resource limits."""

    VERSION = "1.1.0"

    def __init__(
        self,
        *,
        delimiter: str = ",",
        max_rows: int | None = None,
        max_bytes: int | None = None,
        max_columns: int | None = None,
        max_cells: int | None = None,
        resource_policy: CsvResourcePolicy | None = None,
    ) -> None:
        if delimiter not in {",", ";", "\t", "|"}:
            raise ValueError("Unsupported CSV delimiter.")

        explicit_limits = {
            "max_rows": max_rows,
            "max_bytes": max_bytes,
            "max_columns": max_columns,
            "max_cells": max_cells,
        }

        if resource_policy is not None:
            if any(value is not None for value in explicit_limits.values()):
                raise ValueError("Use resource_policy or explicit limits, not both.")

            max_rows = resource_policy.max_rows
            max_bytes = resource_policy.max_bytes
            max_columns = resource_policy.max_columns
            max_cells = resource_policy.max_cells

        limits = {
            "max_rows": max_rows,
            "max_bytes": max_bytes,
            "max_columns": max_columns,
            "max_cells": max_cells,
        }

        for name, value in limits.items():
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer when provided.")

        self._delimiter = delimiter
        self._max_rows = max_rows
        self._max_bytes = max_bytes
        self._max_columns = max_columns
        self._max_cells = max_cells

    def bounded_by(self, policy: CsvResourcePolicy) -> "SafeCsvReader":
        """Return an independent reader with limits no looser than policy."""
        reader = copy(self)
        for name in ("max_bytes", "max_rows", "max_columns", "max_cells"):
            current = getattr(self, f"_{name}")
            ceiling = getattr(policy, name)
            setattr(reader, f"_{name}", ceiling if current is None else min(current, ceiling))
        return reader

    def read(self, raw_bytes: bytes) -> pd.DataFrame:
        """Parse existing bytes without implicit type coercion."""

        if self._max_bytes is not None and len(raw_bytes) > self._max_bytes:
            raise SafeCsvError(
                "CSV_BYTE_LIMIT_EXCEEDED",
                "CSV byte limit exceeded.",
            )

        return self._parse(raw_bytes)

    def read_stream(self, stream: BinaryIO) -> pd.DataFrame:
        """Read a binary stream within a mandatory byte limit."""

        if self._max_bytes is None:
            raise ValueError("read_stream requires a configured max_bytes limit.")

        buffer = bytearray()

        while True:
            remaining = self._max_bytes - len(buffer)

            read_size = min(
                READ_CHUNK_SIZE,
                remaining + 1,
            )

            chunk = stream.read(read_size)

            if not chunk:
                break

            if len(chunk) > remaining:
                raise SafeCsvError(
                    "CSV_BYTE_LIMIT_EXCEEDED",
                    "CSV byte limit exceeded.",
                )

            buffer.extend(chunk)

        return self.read(bytes(buffer))

    def _parse(self, raw_bytes: bytes) -> pd.DataFrame:
        """Parse lexical CSV values and validate structural limits."""

        try:
            text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise SafeCsvError(
                "CSV_DECODE_ERROR",
                "CSV must be valid UTF-8.",
            ) from exc

        if not text.strip():
            raise SafeCsvError(
                "CSV_EMPTY_FILE",
                "Uploaded CSV file is empty.",
            )

        if "\x00" in text:
            raise SafeCsvError(
                "CSV_PARSE_ERROR",
                "CSV contains a NUL character.",
            )

        reader = csv.reader(
            StringIO(text, newline=""),
            delimiter=self._delimiter,
            strict=True,
        )

        try:
            headers = [header.strip() for header in next(reader)]

            column_count = len(headers)

            if self._max_columns is not None and column_count > self._max_columns:
                raise SafeCsvError(
                    "CSV_COLUMN_LIMIT_EXCEEDED",
                    "CSV column count limit exceeded.",
                )

            if not headers or any(not header for header in headers):
                raise SafeCsvError(
                    "CSV_BLANK_HEADER",
                    "CSV header contains blank column names.",
                )

            if column_count != len(set(headers)):
                raise SafeCsvError(
                    "CSV_DUPLICATE_HEADER",
                    "CSV header contains duplicate column names.",
                )

            rows: list[list[str]] = []

            for row in reader:
                if len(row) != column_count:
                    raise SafeCsvError(
                        "CSV_ROW_WIDTH_MISMATCH",
                        f"CSV record ending at line {reader.line_num} "
                        f"has {len(row)} fields; "
                        f"expected {column_count}.",
                    )

                next_row_count = len(rows) + 1

                if self._max_rows is not None and next_row_count > self._max_rows:
                    raise SafeCsvError(
                        "CSV_ROW_LIMIT_EXCEEDED",
                        "CSV row limit exceeded.",
                    )

                if self._max_cells is not None and next_row_count * column_count > self._max_cells:
                    raise SafeCsvError(
                        "CSV_CELL_LIMIT_EXCEEDED",
                        "CSV cell count limit exceeded.",
                    )

                rows.append(row)

        except csv.Error as exc:
            raise SafeCsvError(
                "CSV_PARSE_ERROR",
                f"Invalid CSV near line {reader.line_num}.",
            ) from exc

        frame = pd.DataFrame(
            rows,
            columns=headers,
            dtype="string",
        )

        return frame.mask(frame.eq(""), pd.NA)
