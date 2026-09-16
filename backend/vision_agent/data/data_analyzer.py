"""
Analytical engine for computing totals, averages, column comparisons, and trends.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import pandas as pd

from .dataframe_processor import DataFrameProcessor
from vision_agent.utils.logging_utils import get_logger

logger = get_logger("data_analyzer")


class DataAnalyzer:
    @staticmethod
    def analyze_table(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs comprehensive tabular analysis on DataFrame and produces structured insights.
        """
        clean_df = DataFrameProcessor.clean_dataframe(df)
        summary = DataFrameProcessor.summarize_dataframe(clean_df)
        insights: List[str] = []

        num_stats = summary.get("numeric_stats", {})
        for col, stats in num_stats.items():
            insights.append(
                f"Column '{col}': Total = {stats['total']:,.2f}, Average = {stats['mean']:,.2f}, "
                f"Min = {stats['min']:,.2f}, Max = {stats['max']:,.2f}"
            )

            # Check for top and bottom rows
            try:
                max_idx = clean_df[col].idxmax()
                min_idx = clean_df[col].idxmin()
                if pd.notna(max_idx) and pd.notna(min_idx):
                    first_col = clean_df.columns[0]
                    if first_col != col:
                        insights.append(
                            f"  -> Highest {col} is for '{clean_df.loc[max_idx, first_col]}' ({clean_df.loc[max_idx, col]:,.2f})"
                        )
                        insights.append(
                            f"  -> Lowest {col} is for '{clean_df.loc[min_idx, first_col]}' ({clean_df.loc[min_idx, col]:,.2f})"
                        )
            except Exception:
                pass

        # Check for simple trend if rows > 3 and numeric col exists
        if len(clean_df) >= 3 and num_stats:
            first_num_col = list(num_stats.keys())[0]
            series = clean_df[first_num_col].dropna()
            if len(series) >= 2:
                start_val = series.iloc[0]
                end_val = series.iloc[-1]
                diff = end_val - start_val
                pct_change = (diff / start_val * 100) if start_val != 0 else 0
                direction = "increased" if diff > 0 else "decreased" if diff < 0 else "remained steady"
                insights.append(
                    f"Trend for '{first_num_col}': {direction} from {start_val:,.2f} to {end_val:,.2f} ({pct_change:+.2f}%)"
                )

        return {
            "summary": summary,
            "insights": insights,
            "cleaned_dataframe": clean_df,
        }
