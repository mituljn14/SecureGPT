"""
Interactive terminal wizard for the Local Vision Agent.
Prompts the user for operations, provides direct terminal querying against Qdrant,
native Windows file browsing / path entry, and executes the full multimodal pipeline.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional, List

from vision_agent.config.settings import settings
from vision_agent.core.pipeline import VisionPipeline
from vision_agent.cli.commands import CLIHandler
from vision_agent.utils.file_utils import detect_file_type
from vision_agent.utils.logging_utils import get_logger

logger = get_logger("interactive")


def open_file_browser(title: str = "Select Document or Image") -> Optional[str]:
    """
    Opens native OS file picker dialog to let user easily browse for files.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        filetypes = [
            ("Supported Files", "*.pdf;*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tiff;*.csv;*.xlsx;*.txt;*.md"),
            ("PDF Documents (*.pdf)", "*.pdf"),
            ("Images (*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tiff)", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tiff"),
            ("Data Tables (*.csv;*.xlsx;*.xls)", "*.csv;*.xlsx;*.xls"),
            ("All Files (*.*)", "*.*"),
        ]

        file_path = filedialog.askopenfilename(
            title=title,
            initialdir=str(settings.INPUTS_DIR if settings.INPUTS_DIR.exists() else settings.PROJECT_ROOT),
            filetypes=filetypes,
        )
        root.destroy()
        return file_path if file_path else None
    except Exception as exc:
        print(f"Note: Could not open graphical file browser ({exc}).")
        return None


def list_existing_input_files() -> List[Path]:
    """Finds all existing files in inputs/ and outputs/temporary/."""
    candidates = []
    search_dirs = [settings.INPUTS_DIR, settings.TEMP_DIR]
    for d in search_dirs:
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    candidates.append(f)
    return sorted(candidates, key=lambda x: x.name)


def prompt_for_file(operation_name: str) -> Optional[str]:
    """
    Prompts user for file selection via:
    1. Graphical File Browser (Open Dialog)
    2. Typing / Pasting File Path
    3. Picking from existing input files
    """
    print()
    print(f"--- Select File for: {operation_name} ---")
    print("  [1] Browse file with Windows File Explorer (Open File Dialog)")
    print("  [2] Type or paste file path directly")
    print("  [3] Choose from available workspace / inputs files")
    print("  [0] Cancel / Back to Main Menu")
    print()

    choice = input("Enter choice (0-3) [default: 1]: ").strip() or "1"

    if choice == "0":
        return None

    if choice == "1":
        print("Opening Windows File Browser... Please select your file.")
        selected = open_file_browser(title=f"Select file for {operation_name}")
        if selected:
            print(f"Selected: {selected}")
            return selected
        else:
            print("No file selected in browser. Falling back to manual entry.")
            return prompt_for_file(operation_name)

    elif choice == "2":
        path_str = input("Enter or paste file path: ").strip().strip('"').strip("'")
        if not path_str:
            print("No path entered.")
            return prompt_for_file(operation_name)
        p = Path(path_str)
        if not p.exists():
            print(f"Warning: File '{path_str}' does not exist.")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry == "y":
                return prompt_for_file(operation_name)
            return None
        return str(p.resolve())

    elif choice == "3":
        files = list_existing_input_files()
        if not files:
            print("No existing files found in inputs/ or temporary outputs.")
            return prompt_for_file(operation_name)

        print("\nAvailable workspace files:")
        for idx, f in enumerate(files, 1):
            try:
                rel = f.relative_to(settings.PROJECT_ROOT)
                print(f"  [{idx}] {rel} ({detect_file_type(f)})")
            except Exception:
                print(f"  [{idx}] {f.name} ({detect_file_type(f)})")

        sel = input(f"\nSelect file number (1-{len(files)}): ").strip()
        try:
            sel_idx = int(sel) - 1
            if 0 <= sel_idx < len(files):
                selected = str(files[sel_idx].resolve())
                print(f"Selected: {selected}")
                return selected
            else:
                print("Invalid selection number.")
        except ValueError:
            print("Invalid input.")
        return prompt_for_file(operation_name)

    else:
        print("Invalid choice.")
        return prompt_for_file(operation_name)


def run_interactive_wizard():
    """
    Main interactive loop for terminal users.
    """
    handler = CLIHandler()

    while True:
        print()
        print("=" * 68)
        print("           LOCAL VISION AGENT - INTERACTIVE TERMINAL           ")
        print("=" * 68)
        print("Select an option or type your question directly:")
        print("  [1] Ask Question / Query Knowledge Base directly (Qdrant & AI)")
        print("  [2] Ask a Question about a specific Document / Image / CAD")
        print("  [3] Summarize Document / Scanned PDF / Image / CAD")
        print("  [4] Data Visualization & Chart Generation (Pandas / Seaborn)")
        print("  [5] CAD Blueprint & Technical Diagram Visualization / Analysis")
        print("  [6] Extract Tables & Text into Structured CSV")
        print("  [7] Deep Multimodal Analysis (Tables, Visuals & OCR)")
        print("  [8] System, Vision Model & Qdrant Vector DB Inspection")
        print("  [9] Launch Graphical Desktop UI Window")
        print("  [10] Exit")
        print("=" * 68)

        user_input = input("Enter choice (1-10) or type your query: ").strip()

        if not user_input:
            continue

        if user_input in ("10", "exit", "quit", "q"):
            print("\nExiting Vision Agent. Goodbye!")
            break

        elif user_input == "9":
            print("\nLaunching Graphical Desktop UI...")
            from vision_agent.ui.desktop_app import launch_desktop_ui
            launch_desktop_ui()

        # Option 1: Direct Ask / Query Knowledge Base
        elif user_input == "1":
            print()
            query = input("Enter your question / search query: ").strip()
            if query:
                print(f"\nSearching Qdrant knowledge base for: '{query}'...")
                handler.handle_direct_query(query)
            else:
                print("Query cannot be empty.")

        # Option 2: Ask about a specific document
        elif user_input == "2":
            file_path = prompt_for_file("Question Answering over File")
            if file_path:
                question = input("\nEnter your question about this document/image: ").strip()
                if question:
                    print(f"\nProcessing and answering question...")
                    handler.handle_ask(file_path, question)
                else:
                    print("Question cannot be empty.")

        # Option 3: Summarize
        elif user_input == "3":
            file_path = prompt_for_file("Summarize Document")
            if file_path:
                print(f"\nProcessing summary for: {file_path}...")
                handler.handle_summarize(file_path)

        # Option 4: Data Visualization
        elif user_input == "4":
            file_path = prompt_for_file("Data Visualization")
            if file_path:
                print(f"\nGenerating visualization for: {file_path}...")
                handler.handle_visualize(file_path)

        # Option 5: CAD & Diagram Analysis
        elif user_input == "5":
            file_path = prompt_for_file("CAD & Diagram Analysis")
            if file_path:
                print(f"\nAnalyzing CAD / Technical Diagram: {file_path}...")
                handler.handle_analyze(file_path)

        # Option 6: Extract Tables & Text
        elif user_input == "6":
            file_path = prompt_for_file("Extract Tables & Text")
            if file_path:
                print(f"\nExtracting structured data from: {file_path}...")
                handler.handle_extract(file_path)

        # Option 7: Deep Multimodal Analysis
        elif user_input == "7":
            file_path = prompt_for_file("Deep Multimodal Analysis")
            if file_path:
                print(f"\nRunning deep analysis on: {file_path}...")
                handler.handle_analyze(file_path)

        # Option 8: System Inspection
        elif user_input == "8":
            print("\nInspecting system environment, models, and Qdrant...")
            handler.handle_inspect()

        # If user directly typed a question sentence instead of menu number
        elif len(user_input.split()) >= 2:
            print(f"\nDirect query detected: '{user_input}'")
            print("Searching Qdrant knowledge base and generating answer...")
            handler.handle_direct_query(user_input)

        else:
            print("\nInvalid choice. Please enter a number between 1 and 9 or type a question.")

        input("\nPress Enter to return to the main menu...")
