"""
test_sandbox_executor.py

Tests for Step 5 (sandbox_executor.py).

Requires:
  - Docker running
  - Image built: docker build -t coding-agent-sandbox -f Dockerfile .

Run:
    python test_sandbox_executor.py
"""

import os
import tempfile

from sandbox_executor import run_in_sandbox


def test_basic_success():
    code = "print('RESULT: 42')"
    result = run_in_sandbox(code, file_mapping={}, timeout_seconds=15)
    assert result["exit_code"] == 0, f"Unexpected exit_code: {result}"
    assert "RESULT: 42" in result["stdout"], f"stdout was: {result['stdout']!r}"
    assert result["timed_out"] is False
    print("PASS: basic_success")
    print(f"  execution_time_seconds={result['execution_time_seconds']}")


def test_captures_stderr_and_nonzero_exit():
    code = "raise ValueError('deliberate failure')"
    result = run_in_sandbox(code, file_mapping={}, timeout_seconds=15)
    assert result["exit_code"] != 0, f"Expected nonzero exit, got: {result}"
    assert "ValueError" in result["stderr"], f"stderr was: {result['stderr']!r}"
    print("PASS: captures_stderr_and_nonzero_exit")


def test_timeout_is_enforced():
    code = "import time\nwhile True:\n    time.sleep(1)\n"
    result = run_in_sandbox(code, file_mapping={}, timeout_seconds=5)
    assert result["timed_out"] is True, f"Expected timeout, got: {result}"
    print("PASS: timeout_is_enforced")


def test_network_is_blocked():
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('8.8.8.8', 53), timeout=3)\n"
        "    print('RESULT: network reachable (BAD)')\n"
        "except Exception as e:\n"
        "    print('RESULT: network blocked as expected:', e)\n"
    )
    result = run_in_sandbox(code, file_mapping={}, timeout_seconds=15)
    assert "network blocked" in result["stdout"], f"stdout was: {result['stdout']!r}"
    print("PASS: network_is_blocked")


def test_only_mapped_file_is_visible():
    # create a real temp file on disk to act as the "source_path" file
    tmp_dir = tempfile.mkdtemp()
    real_file = os.path.join(tmp_dir, "sample_data.csv")
    with open(real_file, "w", encoding="utf-8") as f:
        f.write("a,b\n1,2\n3,4\n")

    code = (
        "import pandas as pd\n"
        "df = pd.read_csv('data/sample_data.csv')\n"
        "print('RESULT:', df['a'].sum())\n"
    )
    file_mapping = {"data/sample_data.csv": real_file}
    result = run_in_sandbox(code, file_mapping=file_mapping, timeout_seconds=15)
    assert result["exit_code"] == 0, f"Unexpected failure: {result}"
    assert "RESULT: 4" in result["stdout"], f"stdout was: {result['stdout']!r}"
    print("PASS: only_mapped_file_is_visible")


if __name__ == "__main__":
    test_basic_success()
    test_captures_stderr_and_nonzero_exit()
    test_timeout_is_enforced()
    test_network_is_blocked()
    test_only_mapped_file_is_visible()
    print("\nAll Step 5 tests passed.")
