"""
DataFrame cleaning, column typing, and descriptive summarization.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from vision_agent.utils.logging_utils import get_logger

logger = get_logger("dataframe_processor")


class DataFrameProcessor:
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans column names, removes empty rows/cols, and parses numeric/date types.
        """
        if df.empty:
            return df

        df = df.dropna(how="all").dropna(how="all", axis=1)

        # Clean column names
        clean_cols = [str(c).strip().replace("\n", " ") for c in df.columns]
        df.columns = clean_cols

        # Attempt type conversions
        for col in df.columns:
            # Strip whitespace in string columns
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()

            # Attempt numeric conversion
            try:
                cleaned_numeric = (
                    df[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("$", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.replace("€", "", regex=False)
                    .str.replace("£", "", regex=False)
                )
                numeric_series = pd.to_numeric(cleaned_numeric, errors="coerce")
                if numeric_series.notna().sum() >= len(df) * 0.5:
                    df[col] = numeric_series
            except Exception:
                pass

        return df

    @staticmethod
    def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize DataFrame columns into numeric, categorical, datetime."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]

        return {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols,
        }

    @staticmethod
    def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """Generates statistical and structural summary of a DataFrame."""
        if df.empty:
            return {"rows": 0, "columns": 0, "summary": "Empty DataFrame"}

        types = DataFrameProcessor.get_column_types(df)
        num_summary = {}
        for col in types["numeric"]:
            s = df[col].dropna()
            if not s.empty:
                num_summary[col] = {
                    "total": float(s.sum()),
                    "mean": float(s.mean()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "count": int(s.count()),
                }

        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "column_types": types,
            "numeric_stats": num_summary,
            "preview_markdown": df.head(10).to_markdown(index=False),
        }
