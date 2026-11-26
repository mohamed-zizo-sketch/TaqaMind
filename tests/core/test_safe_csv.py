"""Regression tests for lossless lexical CSV parsing."""

import pandas as pd
import pytest

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
