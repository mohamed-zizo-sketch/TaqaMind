"""Shared lexical missing-value normalization for text columns."""

import pandas as pd

from taqamind.core.contracts.enums import DataType


def normalize_text_missing(series: pd.Series, *, data_type: DataType) -> pd.Series:
    """Treat blank text as missing only for string and category features."""
    if data_type not in {DataType.STRING, DataType.CATEGORY}:
        return series
    blank = series.map(lambda value: isinstance(value, str) and not value.strip())
    return series.mask(blank, pd.NA)
