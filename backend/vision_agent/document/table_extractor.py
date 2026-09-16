"""
Table extraction from PDFs, images, and data files into Pandas DataFrames.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import io
import re
import pandas as pd
import pdfplumber

from vision_agent.core.models import TableData
from vision_agent.utils.logging_utils import get_logger

logger = get_logger("table_extractor")


class TableExtractor:
    @staticmethod
    def extract_tables_from_pdf(pdf_path: Path) -> List[TableData]:
        """
        Extract native tables from PDF pages using pdfplumber.
        """
        extracted_tables: List[TableData] = []
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1
                    raw_tables = page.extract_tables()
                    for tbl_idx, raw_table in enumerate(raw_tables):
                        if not raw_table or len(raw_table) < 2:
                            continue

                        # Clean nulls and whitespace
                        cleaned_matrix = [
                            [str(cell).strip() if cell is not None else "" for cell in row]
                            for row in raw_table
                        ]

                        # Attempt to construct DataFrame
                        df = TableExtractor.matrix_to_dataframe(cleaned_matrix)
                        extracted_tables.append(
                            TableData(
                                page_number=page_num,
                                table_index=tbl_idx + 1,
                                df=df,
                                raw_matrix=cleaned_matrix,
                                caption=f"Table {tbl_idx + 1} on Page {page_num}",
                            )
                        )
        except Exception as exc:
            logger.warning(f"Native PDF table extraction encountered an issue: {exc}")

        return extracted_tables

    @staticmethod
    def extract_from_data_file(file_path: Path) -> List[TableData]:
        """Loads tables from CSV, TSV, or Excel files."""
        ext = file_path.suffix.lower()
        extracted: List[TableData] = []
        try:
            if ext in (".csv", ".tsv"):
                delimiter = "\t" if ext == ".tsv" else ","
                df = pd.read_csv(file_path, sep=delimiter)
                extracted.append(
                    TableData(
                        page_number=1,
                        table_index=1,
                        df=df,
                        raw_matrix=[df.columns.tolist()] + df.astype(str).values.tolist(),
                        caption=file_path.name,
                    )
                )
            elif ext in (".xlsx", ".xls"):
                excel = pd.read_excel(file_path, sheet_name=None)
                for idx, (sheet_name, df) in enumerate(excel.items()):
                    extracted.append(
                        TableData(
                            page_number=idx + 1,
                            table_index=idx + 1,
                            df=df,
                            raw_matrix=[df.columns.tolist()] + df.astype(str).values.tolist(),
                            caption=f"{file_path.name} - {sheet_name}",
                        )
                    )
        except Exception as exc:
            logger.error(f"Failed to read data file {file_path} as table: {exc}")
        return extracted

    @staticmethod
    def parse_markdown_table(text: str, page_number: int = 1) -> Optional[TableData]:
        """Parses a markdown table string into a DataFrame TableData."""
        lines = [line.strip() for line in text.splitlines() if "|" in line]
        if len(lines) < 3:
            return None

        # Filter out markdown separator lines like |---|---|
        data_rows = []
        for line in lines:
            if re.match(r"^\|?\s*[-:]+\s*(\|\s*[-:]+\s*)+\|?$", line):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            data_rows.append(cells)

        if len(data_rows) >= 2:
            df = TableExtractor.matrix_to_dataframe(data_rows)
            return TableData(
                page_number=page_number,
                table_index=1,
                df=df,
                raw_matrix=data_rows,
                caption="Extracted Table",
            )
        return None

    @staticmethod
    def matrix_to_dataframe(matrix: List[List[str]]) -> Optional[pd.DataFrame]:
        """Convert a 2D list of strings into a clean pandas DataFrame."""
        if not matrix or len(matrix) < 2:
            return None
        try:
            header = matrix[0]
            # Ensure unique non-empty header names
            clean_headers = []
            seen = {}
            for idx, col in enumerate(header):
                name = col.strip() or f"Column_{idx+1}"
                count = seen.get(name, 0)
                seen[name] = count + 1
                if count > 0:
                    name = f"{name}_{count}"
                clean_headers.append(name)

            data = matrix[1:]
            # Normalize row lengths
            max_len = len(clean_headers)
            normalized_data = [row + [""] * (max_len - len(row)) if len(row) < max_len else row[:max_len] for row in data]

            df = pd.DataFrame(normalized_data, columns=clean_headers)

            # Auto-convert numeric columns
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col].str.replace(",", "").str.replace("$", ""))
                except Exception:
                    pass

            return df
        except Exception as exc:
            logger.debug(f"Matrix to DataFrame conversion error: {exc}")
            return None
