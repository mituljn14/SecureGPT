"""
verifier.py
Step 6: Verifier

Takes raw output from the Docker sandbox (Step 5) and determines whether
execution actually succeeded and produced a valid, usable result.

Convention (must match how Step 4 prompts Qwen to write code):
The generated Python code must print its final answer as JSON on a single
line prefixed with the marker below:

    ___RESULT___{"answer": ..., "details": {...}}

Anything else printed before/after is treated as normal stdout/log noise.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


RESULT_MARKER = "___RESULT___"

KNOWN_ERROR_PATTERNS = [
    (r"KeyError", "The code referenced a column/key that does not exist in the data."),
    (r"FileNotFoundError", "The code could not find an input file."),
    (r"ZeroDivisionError", "The code divided by zero (likely empty/filtered dataset)."),
    (r"MemoryError", "The code ran out of memory."),
    (r"ModuleNotFoundError|ImportError", "The code imported a package not available in the sandbox."),
    (r"SyntaxError", "The generated code has a syntax error."),
    (r"TimeoutExpired|timed out", "Execution exceeded the sandbox timeout."),
]


@dataclass
class VerificationResult:
    status: str                    # "pass" | "fail"
    reason: str
    result: Optional[Any] = None   # parsed JSON payload from ___RESULT___
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "status": self.status,
            "reason": self.reason,
            "result": self.result,
            "warnings": self.warnings,
        }


class Verifier:
    def __init__(self, max_execution_time: float = 30.0):
        self.max_execution_time = max_execution_time

    def verify(self, sandbox_output: dict) -> VerificationResult:
        """
        sandbox_output must contain: stdout, stderr, exit_code, execution_time
        """
        stdout = sandbox_output.get("stdout", "") or ""
        stderr = sandbox_output.get("stderr", "") or ""
        exit_code = sandbox_output.get("exit_code", 1)
        exec_time = sandbox_output.get("execution_time", 0.0)

        warnings = []

        if exit_code != 0:
            reason = self._classify_error(stderr)
            return VerificationResult(status="fail", reason=reason, warnings=warnings)

        if stderr.strip():
            warnings.append("stderr was not empty even though exit_code was 0.")

        if exec_time > self.max_execution_time:
            warnings.append(
                f"Execution took {exec_time:.2f}s, exceeding expected max of {self.max_execution_time}s."
            )

        parsed = self._extract_result(stdout)
        if parsed is None:
            return VerificationResult(
                status="fail",
                reason=f"No '{RESULT_MARKER}' JSON payload found in stdout. Code ran but produced no parseable result.",
                warnings=warnings,
            )

        content_issue = self._validate_content(parsed)
        if content_issue:
            return VerificationResult(status="fail", reason=content_issue, result=parsed, warnings=warnings)

        return VerificationResult(status="pass", reason="Execution succeeded and result validated.", result=parsed, warnings=warnings)

    def _classify_error(self, stderr: str) -> str:
        for pattern, explanation in KNOWN_ERROR_PATTERNS:
            if re.search(pattern, stderr):
                return explanation
        if stderr.strip():
            lines = [l for l in stderr.strip().splitlines() if l.strip()]
            return f"Execution failed: {lines[-1] if lines else 'unknown error'}"
        return "Execution failed with non-zero exit code and no stderr output."

    def _extract_result(self, stdout: str) -> Optional[Any]:
        # Search for ___RESULT___ line or pattern anywhere in stdout
        for line in stdout.splitlines():
            line = line.strip()
            if RESULT_MARKER in line:
                idx = line.find(RESULT_MARKER)
                payload = line[idx + len(RESULT_MARKER):].strip()
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    continue

        # Regex fallback to match ___RESULT___ followed by JSON object
        match = re.search(r"___RESULT___\s*(\{.*?\})", stdout, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        return None

    def _validate_content(self, parsed: Any) -> Optional[str]:
        if parsed is None:
            return "Result payload was null."
        if isinstance(parsed, dict):
            if "answer" not in parsed:
                return "Result JSON is missing required 'answer' field."
            answer = parsed["answer"]
            if answer is None or answer == "":
                return "Result 'answer' field is empty/null."
            if isinstance(answer, float) and answer != answer:  # NaN check
                return "Result 'answer' field is NaN."
        return None


if __name__ == "__main__":
    v = Verifier(max_execution_time=30.0)

    good = {
        "stdout": f'some log line\n{RESULT_MARKER}{{"answer": 4523.7, "details": {{"rows_used": 108}}}}\n',
        "stderr": "", "exit_code": 0, "execution_time": 1.2,
    }
    r1 = v.verify(good)
    print("CASE 1 (expected pass):", r1.to_dict())
    assert r1.status == "pass"

    bad = {
        "stdout": "", "stderr": "Traceback (most recent call last):\nKeyError: 'Sales_2025'",
        "exit_code": 1, "execution_time": 0.3,
    }
    r2 = v.verify(bad)
    print("CASE 2 (expected fail):", r2.to_dict())
    assert r2.status == "fail"

    no_marker = {
        "stdout": "computed average = 4523.7\n", "stderr": "", "exit_code": 0, "execution_time": 0.5,
    }
    r3 = v.verify(no_marker)
    print("CASE 3 (expected fail, no marker):", r3.to_dict())
    assert r3.status == "fail"

    print("\nAll Step 6 tests passed.")