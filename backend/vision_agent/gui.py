"""
Launch the Local Vision Agent Native Desktop UI.
Run:
    python gui.py
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision_agent.ui.desktop_app import launch_desktop_ui

if __name__ == "__main__":
    launch_desktop_ui()
