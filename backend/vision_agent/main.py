"""
Production Local Vision Agent CLI Entry Point.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Ensure package modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_agent.config.settings import settings
from vision_agent.utils.logging_utils import setup_tee_logger, get_logger
from vision_agent.cli.commands import CLIHandler

# Initialize Tee Logging to dual-output to terminal & logs/
setup_tee_logger("vision_agent_session")
from vision_agent.cli.interactive import run_interactive_wizard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Local Vision Agent - Multimodal PDF, Image, OCR, Table, and Chart Analysis with Qdrant Retrieval",
        epilog="Examples:\n"
        "  python main.py                              # Interactive Menu Wizard\n"
        "  python main.py summarize data/report.pdf\n"
        "  python main.py analyze inputs/diagram.png\n"
        "  python main.py ask data/financials.pdf 'What were the total Q1 revenues?'\n"
        "  python main.py extract data/sales.xlsx\n"
        "  python main.py visualize data/metrics.csv\n"
        "  python main.py inspect\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive terminal wizard")
    parser.add_argument("--ui", "--gui", action="store_true", help="Launch native desktop graphical interface")
    subparsers = parser.add_subparsers(dest="command", help="Operation to perform")

    # gui
    subparsers.add_parser("gui", help="Launch native desktop GUI window")

    # query (direct ask without file)
    p_qry = subparsers.add_parser("query", help="Ask a question directly to Qdrant vector store and AI without file")
    p_qry.add_argument("question", help="Question to search across knowledge base")

    # summarize
    p_sum = subparsers.add_parser("summarize", help="Generate a comprehensive summary of a document or image")
    p_sum.add_argument("file", help="Path to PDF, image, or document file")

    # analyze
    p_ana = subparsers.add_parser("analyze", help="Deep multimodal inspection (tables, charts, visuals, OCR)")
    p_ana.add_argument("file", help="Path to PDF, image, or document file")

    # ask
    p_ask = subparsers.add_parser("ask", help="Ask a question about a document or visual with grounded retrieval")
    p_ask.add_argument("file", help="Path to PDF, image, or document file")
    p_ask.add_argument("question", help="Question to ask")

    # extract
    p_ext = subparsers.add_parser("extract", help="Extract raw structured tables and text")
    p_ext.add_argument("file", help="Path to PDF, image, or spreadsheet file")

    # visualize
    p_vis = subparsers.add_parser("visualize", help="Extract data and generate analytical charts")
    p_vis.add_argument("file", help="Path to PDF, image, or tabular file")

    # inspect
    p_ins = subparsers.add_parser("inspect", help="Inspect system, Ollama models, and Qdrant collections")
    p_ins.add_argument("file", nargs="?", default=None, help="Optional file to inspect")

    return parser


def main():
    if len(sys.argv) == 1:
        # Launch Interactive Wizard directly
        try:
            run_interactive_wizard()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
        return

    parser = build_parser()
    args = parser.parse_args()

    if args.ui or args.command == "gui":
        from vision_agent.ui.desktop_app import launch_desktop_ui
        launch_desktop_ui()
        return

    if args.interactive:
        try:
            run_interactive_wizard()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
        return

    handler = CLIHandler()

    try:
        if args.command == "query":
            handler.handle_direct_query(args.question)
        elif args.command == "summarize":
            handler.handle_summarize(args.file)
        elif args.command == "analyze":
            handler.handle_analyze(args.file)
        elif args.command == "ask":
            handler.handle_ask(args.file, args.question)
        elif args.command == "extract":
            handler.handle_extract(args.file)
        elif args.command == "visualize":
            handler.handle_visualize(args.file)
        elif args.command == "inspect":
            handler.handle_inspect(args.file)
        else:
            parser.print_help()

    except FileNotFoundError as exc:
        print(f"\n[ERROR]: File not found: {exc}", file=sys.stderr)
        sys.exit(1)
    except ConnectionError as exc:
        print(f"\n[ERROR]: Connection Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.exception("An unexpected error occurred during execution.")
        print(f"\n[ERROR]: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
