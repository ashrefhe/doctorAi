import pandas as pd
import numpy as np
import io
import chardet


def load_dataset(uploaded_file) -> pd.DataFrame:
    """
    Load dataset from uploaded file (CSV or Excel).
    Handles unusual separators, multiple encodings, and malformed files.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        raw = uploaded_file.read()
        # Auto-detect encoding
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"

        # Try detected encoding first, then common fallbacks
        for enc in [encoding, "utf-8", "latin-1", "cp1252"]:
            try:
                content = raw.decode(enc)
                break
            except (UnicodeDecodeError, TypeError):
                continue
        else:
            raise ValueError(
                "Could not decode the CSV file. "
                "Please save it as UTF-8 and try again."
            )

        # Auto-detect separator
        sample = content[:4096]
        sep = ","
        for candidate in [",", ";", "\t", "|"]:
            count = sample.count(candidate)
            if count > sample.count(sep):
                sep = candidate

        try:
            df = pd.read_csv(io.StringIO(content), sep=sep)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {e}")

    elif filename.endswith((".xls", ".xlsx")):
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            raise ValueError(f"Failed to parse Excel file: {e}")

    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")

    if df.empty:
        raise ValueError("The uploaded file is empty or could not be parsed correctly.")

    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Return basic stats about the dataset."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "nunique": df.nunique().to_dict(),
        "describe": df.describe(include="all").to_dict(),
        "info": info_str,
    }


def format_metric(value: float, decimals: int = 4) -> str:
    """Format a metric for display."""
    return f"{value:.{decimals}f}"
