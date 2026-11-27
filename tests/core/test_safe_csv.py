"""Regression tests for lossless lexical CSV parsing."""

from io import BytesIO

import pandas as pd
import pytest

from taqamind.core.ingestion.resource_policy import CsvResourcePolicy
from taqamind.core.ingestion.safe_csv import SafeCsvError, SafeCsvReader


def test_preserves_tokens_and_missing_values():
    frame = SafeCsvReader().read(
        b"id,value,label\n000123,9007199254740993,NA\n00123,1.2300,NULL\n0,,NaN\n"
    )
    assert frame["id"].tolist() == ["000123", "00123", "0"]
    assert frame["value"].iloc[:2].tolist() == ["9007199254740993", "1.2300"]
    assert pd.isna(frame["value"].iloc[2])
    assert frame["label"].tolist() == ["NA", "NULL", "NaN"]


def test_quoted_fields_and_whitespace():
    frame = SafeCsvReader().read(b'id,note\r\n001,"a,b\r\nc"\r\n002,"  text  "\r\n')
    assert frame["note"].tolist() == ["a,b\r\nc", "  text  "]


def test_bom_and_trimmed_headers():
    frame = SafeCsvReader().read(b"\xef\xbb\xbf id , value\n001,2\n")
    assert frame.columns.tolist() == ["id", "value"]
    assert frame.iloc[0].tolist() == ["001", "2"]


@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_explicit_delimiters(delimiter):
    frame = SafeCsvReader(delimiter=delimiter).read(
        f"id{delimiter}value\n001{delimiter}2\n".encode()
    )
    assert frame.iloc[0].tolist() == ["001", "2"]


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (b"", "CSV_EMPTY_FILE"),
        (b" \n", "CSV_EMPTY_FILE"),
        (b"\xff", "CSV_DECODE_ERROR"),
        (b"id,id\n1,2\n", "CSV_DUPLICATE_HEADER"),
        (b"id, id \n1,2\n", "CSV_DUPLICATE_HEADER"),
        (b"id, \n1,2\n", "CSV_BLANK_HEADER"),
        (b"id,value\n1,2,3\n", "CSV_ROW_WIDTH_MISMATCH"),
        (b"id,value\n1\n", "CSV_ROW_WIDTH_MISMATCH"),
        (b"id,value\n\n", "CSV_ROW_WIDTH_MISMATCH"),
        (b'id,value\n1,"unfinished\n', "CSV_PARSE_ERROR"),
        (b'id,value\n1,"a"b\n', "CSV_PARSE_ERROR"),
        (b"id,value\n1,\x00\n", "CSV_PARSE_ERROR"),
    ],
)
def test_invalid_csv(payload, code):
    with pytest.raises(SafeCsvError) as error:
        SafeCsvReader().read(payload)
    assert error.value.code == code


def test_row_limit_and_header_only():
    reader = SafeCsvReader(max_rows=1)
    assert len(reader.read(b"id,value\n1,2\n")) == 1
    assert reader.read(b"id,value\n").empty
    with pytest.raises(SafeCsvError) as error:
        reader.read(b"id,value\n1,2\n3,4\n")
    assert error.value.code == "CSV_ROW_LIMIT_EXCEEDED"


def test_invalid_configuration():
    with pytest.raises(ValueError):
        SafeCsvReader(delimiter="::")
    with pytest.raises(ValueError):
        SafeCsvReader(max_rows=0)


def test_round_trip():
    reader = SafeCsvReader()
    original = reader.read(b'id,value,note\n0001,9007199254740993,"a,b"\n0002,,NULL\n')
    restored = reader.read(original.to_csv(index=False).encode())
    pd.testing.assert_frame_equal(original, restored)


def test_byte_limit_exact_boundary():
    payload = b"id,value\n1,2\n"

    reader = SafeCsvReader(max_bytes=len(payload))

    assert len(reader.read(payload)) == 1


def test_byte_limit_exceeded():
    payload = b"id,value\n1,2\n"

    reader = SafeCsvReader(max_bytes=len(payload) - 1)

    with pytest.raises(SafeCsvError) as error:
        reader.read(payload)

    assert error.value.code == "CSV_BYTE_LIMIT_EXCEEDED"


def test_column_limit_exact_boundary():
    reader = SafeCsvReader(max_columns=2)

    frame = reader.read(b"id,value\n1,2\n")

    assert len(frame.columns) == 2


def test_column_limit_exceeded():
    reader = SafeCsvReader(max_columns=2)

    with pytest.raises(SafeCsvError) as error:
        reader.read(b"a,b,c\n1,2,3\n")

    assert error.value.code == "CSV_COLUMN_LIMIT_EXCEEDED"


def test_cell_limit_exact_boundary():
    reader = SafeCsvReader(max_cells=4)

    frame = reader.read(b"a,b\n1,2\n3,4\n")

    assert frame.shape == (2, 2)


def test_cell_limit_exceeded():
    reader = SafeCsvReader(max_cells=3)

    with pytest.raises(SafeCsvError) as error:
        reader.read(b"a,b\n1,2\n3,4\n")

    assert error.value.code == "CSV_CELL_LIMIT_EXCEEDED"


def test_reader_accepts_resource_policy():
    policy = CsvResourcePolicy(
        max_bytes=100,
        max_rows=1,
        max_columns=2,
        max_cells=2,
    )

    reader = SafeCsvReader(resource_policy=policy)

    frame = reader.read(b"a,b\n1,2\n")

    assert frame.shape == (1, 2)

    with pytest.raises(SafeCsvError) as error:
        reader.read(b"a,b\n1,2\n3,4\n")

    assert error.value.code == "CSV_ROW_LIMIT_EXCEEDED"


def test_conflicting_configuration_is_rejected():
    with pytest.raises(ValueError):
        SafeCsvReader(
            resource_policy=CsvResourcePolicy(),
            max_rows=10,
        )


def test_stream_exact_byte_limit():
    payload = b"id,value\n0001,NA\n"

    reader = SafeCsvReader(max_bytes=len(payload))

    with BytesIO(payload) as stream:
        result = reader.read_stream(stream)

        assert not stream.closed

    assert result.iloc[0].tolist() == ["0001", "NA"]


def test_stream_reads_only_one_extra_byte():
    payload = b"id,value\n1,2\n" + b"x" * 10_000
    limit = 100

    reader = SafeCsvReader(max_bytes=limit)

    with BytesIO(payload) as stream:
        with pytest.raises(SafeCsvError) as error:
            reader.read_stream(stream)

        assert stream.tell() == limit + 1

    assert error.value.code == "CSV_BYTE_LIMIT_EXCEEDED"


def test_stream_requires_byte_limit():
    reader = SafeCsvReader()

    with pytest.raises(ValueError, match="max_bytes"):
        reader.read_stream(BytesIO(b"a,b\n1,2\n"))


def test_stream_supports_partial_reads():
    class PartialReadStream(BytesIO):
        def read(self, size=-1):
            assert size > 0
            return super().read(min(size, 3))

    payload = b"\xef\xbb\xbfid,value\n0001,123\n"

    reader = SafeCsvReader(max_bytes=len(payload))

    with PartialReadStream(payload) as stream:
        frame = reader.read_stream(stream)

    pd.testing.assert_frame_equal(
        frame,
        reader.read(payload),
    )


def test_stream_never_requests_unbounded_reads():
    class BoundedOnlyStream(BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            return super().read(size)

    payload = b"a,b\n1,2\n"

    reader = SafeCsvReader(max_bytes=100)

    with BoundedOnlyStream(payload) as stream:
        frame = reader.read_stream(stream)

    assert frame.shape == (1, 2)
