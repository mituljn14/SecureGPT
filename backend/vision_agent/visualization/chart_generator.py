"""
Safe, template-driven chart generation using Matplotlib and Seaborn.
SECURITY RULE: Does NOT evaluate or execute arbitrary LLM code strings.
Uses curated, parameter-validated Python plotting functions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from vision_agent.config.settings import settings
from vision_agent.utils.file_utils import safe_output_path
from vision_agent.utils.logging_utils import get_logger

logger = get_logger("chart_generator")


class ChartGenerator:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.VISUALIZATIONS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Apply modern, elegant styling
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "figure.titlesize": 13,
                "figure.dpi": 150,
            }
        )

    def _get_save_path(self, base_name: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"{base_name}_{timestamp}"
        return safe_output_path(self.output_dir, safe_name, ".png")

    def plot_bar(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: Optional[str] = None,
        horizontal: bool = False,
    ) -> Path:
        """Generates and saves a bar chart."""
        plt.figure(figsize=(9, 5))
        chart_title = title or f"{y_col} by {x_col}"

        if horizontal:
            sns.barplot(data=df, y=x_col, x=y_col, hue=x_col, palette="Blues_d", legend=False)
            plt.xlabel(y_col)
            plt.ylabel(x_col)
        else:
            sns.barplot(data=df, x=x_col, y=y_col, hue=x_col, palette="viridis", legend=False)
            plt.xticks(rotation=30, ha="right")
            plt.xlabel(x_col)
            plt.ylabel(y_col)

        plt.title(chart_title, fontweight="bold", pad=12)
        plt.tight_layout()

        save_path = self._get_save_path("bar_chart")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Generated bar chart: {save_path}")
        return save_path

    def plot_line(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: Optional[str] = None,
    ) -> Path:
        """Generates and saves a line chart."""
        plt.figure(figsize=(9, 5))
        chart_title = title or f"{y_col} Trend over {x_col}"

        sns.lineplot(data=df, x=x_col, y=y_col, marker="o", linewidth=2.5, color="#1f77b4")
        plt.xticks(rotation=30, ha="right")
        plt.title(chart_title, fontweight="bold", pad=12)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()

        save_path = self._get_save_path("line_chart")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Generated line chart: {save_path}")
        return save_path

    def plot_pie(
        self,
        df: pd.DataFrame,
        labels_col: str,
        values_col: str,
        title: Optional[str] = None,
    ) -> Path:
        """Generates and saves a pie chart."""
        plt.figure(figsize=(7, 7))
        chart_title = title or f"Distribution of {values_col} by {labels_col}"

        # Top 8 slices to avoid clutter
        plot_df = df.nlargest(8, values_col) if len(df) > 8 else df
        colors = sns.color_palette("pastel")[0 : len(plot_df)]

        plt.pie(
            plot_df[values_col],
            labels=plot_df[labels_col],
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
        )
        plt.title(chart_title, fontweight="bold", pad=12)
        plt.tight_layout()

        save_path = self._get_save_path("pie_chart")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Generated pie chart: {save_path}")
        return save_path

    def auto_visualize(self, df: pd.DataFrame, title_hint: Optional[str] = None) -> Optional[Path]:
        """
        Inspects DataFrame structure and automatically generates the most meaningful chart.
        """
        if df.empty or len(df.columns) < 2:
            return None

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        all_cols = df.columns.tolist()
        cat_cols = [c for c in all_cols if c not in numeric_cols]

        if not numeric_cols:
            return None

        # Scenario 1: One Categorical + One Numeric -> Bar chart
        if cat_cols and numeric_cols:
            cat = cat_cols[0]
            num = numeric_cols[0]
            if len(df) <= 15:
                return self.plot_bar(df, x_col=cat, y_col=num, title=title_hint)
            else:
                # Top 10 horizontal bar
                top_df = df.nlargest(10, num)
                return self.plot_bar(top_df, x_col=cat, y_col=num, title=title_hint, horizontal=True)

        # Scenario 2: Two Numeric columns -> Line chart
        if len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            return self.plot_line(df, x_col=x_col, y_col=y_col, title=title_hint)

        # Scenario 3: Single numeric column with index -> Line chart
        if len(numeric_cols) == 1:
            df_plot = df.reset_index()
            return self.plot_line(df_plot, x_col="index", y_col=numeric_cols[0], title=title_hint)

        return None
