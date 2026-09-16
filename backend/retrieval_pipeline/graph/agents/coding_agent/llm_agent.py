"""
llm_agent.py

STEP 4: Code Generation via Ollama (Qwen2.5-Coder 3B / configurable).

Responsibilities:
- Send user query + optional file context to the LLM.
- Support both file-based data analysis and standalone calculation tasks.
- Enforce Docker sandbox paths (/workspace/<filename>).
- Provide explicit file-reading instructions per file_type (Excel, CSV, PDF, DOCX).
- Extract Python code cleanly even if model returns markdown fences or comments.
- Automatically ensure required imports (json, pandas), default=str in json.dumps, and result marker.
- Validate Python syntax via AST.
- Support retry with previous error context.
"""

import ast
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


# ============================================================
# CONFIG
# ============================================================

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:3b")

# Timeout in seconds for local inference
TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT", "180"))

# Prevent unnecessarily long responses
MAX_OUTPUT_TOKENS = 512

RESULT_MARKER = "___RESULT___"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = r"""You are an expert Python code generator for an automated data-analysis coding agent.
Your generated code executes inside an isolated Docker sandbox.

OUTPUT CONTRACT:
1. Output ONLY executable Python code. Do NOT output explanations or markdown.
2. The code MUST end by printing the final result as a JSON string prefixed by '___RESULT___':
   print("___RESULT___" + json.dumps({
       "answer": <final_answer_value>,
       "details": <supporting_dict_or_stats>
   }, default=str))
3. The result print statement MUST be the final statement executed.
4. Always import json when printing the result.
5. All numeric answers should be cast to float() or int() (or use default=str in json.dumps).

DOCKER FILE ACCESS:
- The input file is located inside the Docker container working directory at:
  /workspace/<filename>
- Always read the exact file provided from /workspace/<filename>.
- Do NOT change the filename or file extension (e.g. do not change .pdf to .xlsx).
- Use the appropriate library:
  * Excel (.xlsx, .xls) -> import pandas as pd; df = pd.read_excel('/workspace/<filename>')
  * CSV (.csv) -> import pandas as pd; df = pd.read_csv('/workspace/<filename>')
  * PDF (.pdf) -> import pymupdf as fitz; doc = fitz.open('/workspace/<filename>')
  * DOCX (.docx) -> import docx; doc = docx.Document('/workspace/<filename>')

SECURITY:
- Do NOT import network libraries (urllib, requests, socket).
- Do NOT use shell execution (subprocess, os.system).
"""


# ============================================================
# COMPACT FILE CONTEXT
# ============================================================

def compact_file_context(file_context: Optional[dict]) -> Optional[dict]:
    """
    Reduce large file-reader output before sending it to the LLM
    to keep prompt token count small.
    """
    if not isinstance(file_context, dict):
        return file_context

    context = dict(file_context)

    # Truncate sheets preview
    sheets = context.get("sheets")
    if isinstance(sheets, dict):
        compact_sheets = {}
        for sheet_name, sheet_data in sheets.items():
            if not isinstance(sheet_data, dict):
                compact_sheets[sheet_name] = sheet_data
                continue
            sheet = dict(sheet_data)
            if isinstance(sheet.get("preview"), list):
                sheet["preview"] = sheet["preview"][:3]
            compact_sheets[sheet_name] = sheet
        context["sheets"] = compact_sheets

    # Truncate top-level preview
    if isinstance(context.get("preview"), list):
        context["preview"] = context["preview"][:3]

    return context


# ============================================================
# GET DOCKER SANDBOX PATH
# ============================================================

def get_sandbox_path(file_context: Optional[dict]) -> str:
    """
    Convert host source_path into container sandbox path:
    C:/Users/.../Budget_Tracker_Q3_2025.xlsx -> /workspace/Budget_Tracker_Q3_2025.xlsx
    """
    if not isinstance(file_context, dict):
        return ""

    source_path = str(file_context.get("source_path", "")).strip()
    if not source_path:
        return ""

    normalized = source_path.replace("\\", "/")
    filename = normalized.split("/")[-1]
    if not filename:
        return ""

    return f"/workspace/{filename}"


# ============================================================
# BUILD USER PROMPT
# ============================================================

def build_user_prompt(
    user_query: str,
    file_context: Optional[dict] = None
) -> str:
    """
    Construct the prompt sent to the LLM. Supports both file-backed
    data queries and standalone calculation tasks.
    """
    prompt_parts = [
        f"USER REQUEST:\n{user_query}\n"
    ]

    has_file = False
    if isinstance(file_context, dict) and file_context.get("source_path"):
        sandbox_path = get_sandbox_path(file_context)
        file_type = str(file_context.get("file_type", "")).lower()
        if sandbox_path:
            has_file = True
            filename = Path(sandbox_path).name
            compact_context = compact_file_context(file_context)

            if file_type in ("xlsx", "xls"):
                reader_hint = f"import pandas as pd\ndf = pd.read_excel('{sandbox_path}')"
            elif file_type == "csv":
                reader_hint = f"import pandas as pd\ndf = pd.read_csv('{sandbox_path}')"
            elif file_type == "pdf":
                reader_hint = f"import pymupdf as fitz\ndoc = fitz.open('{sandbox_path}')"
            elif file_type == "docx":
                reader_hint = f"import docx\ndoc = docx.Document('{sandbox_path}')"
            else:
                reader_hint = f"# Open '{sandbox_path}'"

            prompt_parts.append(
                f"INPUT FILE CONTEXT:\n"
                f"- Filename: {filename}\n"
                f"- File Type: {file_type}\n"
                f"- Container Path: {sandbox_path}\n"
                f"Read this file using:\n{reader_hint}\n\n"
                f"SCHEMA & PREVIEW:\n"
                f"{json.dumps(compact_context, indent=2, default=str)}\n"
            )

    if not has_file:
        prompt_parts.append(
            "NOTE: This is a direct Python calculation/coding task without any external file.\n"
        )

    prompt_parts.append(
        "Generate ONLY raw executable Python code ending with:\n"
        'print("___RESULT___" + json.dumps({"answer": <computed_value>, "details": {...}}, default=str))\n'
    )

    return "\n".join(prompt_parts)


# ============================================================
# EXTRACT PYTHON CODE
# ============================================================

def _extract_code(raw_text: str) -> str:
    """
    Extract pure Python code if the model returns markdown code blocks,
    explanations, or comments.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # Match ```python ... ```
    match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Match generic ``` ... ```
    match = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    # Remove fence lines if partially present
    lines = text.splitlines()
    cleaned_lines = []
    in_fenced_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fenced_block = not in_fenced_block
            continue
        cleaned_lines.append(line)

    candidate = "\n".join(cleaned_lines).strip()

    # Find where the Python code starts
    code_lines = []
    found_code_start = False
    for line in candidate.splitlines():
        if not found_code_start:
            if re.match(r"^\s*(import\s+|from\s+|def\s+|class\s+|#|[a-zA-Z_]\w*\s*=)", line):
                found_code_start = True
                code_lines.append(line)
        else:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    return candidate


# ============================================================
# REPAIR CODE & ENSURE IMPORTS & RESULT MARKER
# ============================================================

def _clean_and_repair_code(code: str) -> str:
    """
    Post-processing safeguard:
    - Ensure necessary imports (import json, import pandas as pd, etc.) are present.
    - Add default=str to json.dumps if not already present.
    - If the LLM omitted ___RESULT___, wrap the last print or variable assignment.
    """
    if not code.strip():
        return code

    lines = code.splitlines()

    # Prepend import json if json is referenced or ___RESULT___ is outputted
    has_json = any(re.search(r"^\s*import\s+json\b", l) for l in lines)
    if (RESULT_MARKER in code or "json." in code) and not has_json:
        code = "import json\n" + code
        lines = code.splitlines()

    # Prepend import pandas as pd if pd is used without import
    has_pandas = any(re.search(r"^\s*import\s+pandas\b", l) for l in lines)
    if any(re.search(r"\bpd\.", l) for l in lines) and not has_pandas:
        code = "import pandas as pd\n" + code
        lines = code.splitlines()

    # Ensure json.dumps has default=str to prevent numpy float/int serialization errors
    if "json.dumps(" in code and "default=" not in code:
        code = re.sub(r'json\.dumps\(([^,\)]+)\)', r'json.dumps(\1, default=str)', code)
        code = re.sub(r'json\.dumps\((.+?)\)', lambda m: m.group(0) if "default=" in m.group(0) else f"json.dumps({m.group(1)}, default=str)", code)

    # If result marker already present, return
    if RESULT_MARKER in code:
        return code

    try:
        parsed = ast.parse(code)
    except SyntaxError:
        return code

    last_stmt = parsed.body[-1] if parsed.body else None
    if isinstance(last_stmt, ast.Expr) and isinstance(last_stmt.value, ast.Call):
        func = last_stmt.value.func
        if isinstance(func, ast.Name) and func.id == "print":
            code_without_last = "\n".join(lines[:last_stmt.lineno - 1])
            args = last_stmt.value.args
            if args:
                last_arg_repr = ast.unparse(args[-1]) if hasattr(ast, "unparse") else "None"
                import_header = "" if any("import json" in l for l in lines) else "import json\n"
                patched_code = (
                    f"{import_header}{code_without_last}\n"
                    f'print("___RESULT___" + json.dumps({{"answer": {last_arg_repr}, "details": {{}}}}, default=str))\n'
                )
                try:
                    ast.parse(patched_code)
                    return patched_code.strip()
                except SyntaxError:
                    pass

    if isinstance(last_stmt, ast.Assign) and last_stmt.targets:
        target = last_stmt.targets[0]
        if isinstance(target, ast.Name):
            var_name = target.id
            import_header = "" if any("import json" in l for l in lines) else "import json\n"
            patched_code = (
                f"{import_header}{code}\n"
                f'print("___RESULT___" + json.dumps({{"answer": {var_name}, "details": {{}}}}, default=str))\n'
            )
            try:
                ast.parse(patched_code)
                return patched_code.strip()
            except SyntaxError:
                pass

    return code


# ============================================================
# SYNTAX VALIDATION
# ============================================================

def validate_python_syntax(code: str) -> dict:
    """
    Validate generated Python code using AST parsing.
    """
    if not code.strip():
        return {
            "valid": False,
            "error": {
                "message": "Model returned empty code.",
                "line": None,
                "column": None,
                "code_line": ""
            }
        }

    try:
        ast.parse(code)
        return {
            "valid": True,
            "error": None
        }
    except SyntaxError as e:
        lines = code.splitlines()
        problematic_line = ""
        if e.lineno and e.lineno <= len(lines):
            problematic_line = lines[e.lineno - 1]

        return {
            "valid": False,
            "error": {
                "line": e.lineno,
                "column": e.offset,
                "message": e.msg,
                "code_line": problematic_line
            }
        }


# ============================================================
# GENERATE CODE
# ============================================================

def generate_code(
    user_query: str,
    file_context: Optional[dict] = None,
    model: str = MODEL_NAME,
    error_context: Optional[str] = None
) -> dict:
    """
    Generate Python code using Qwen2.5-Coder via Ollama.
    """
    user_prompt = build_user_prompt(user_query, file_context)

    # Handle retry with previous error feedback
    if error_context:
        sandbox_path = get_sandbox_path(file_context) if file_context else ""
        retry_msg = (
            "\n\n"
            "============================================================\n"
            "PREVIOUS ATTEMPT FAILED WITH ERROR:\n"
            "============================================================\n"
            f"{error_context}\n\n"
            "Fix the Python code. Ensure:\n"
            "1. Output ONLY valid executable Python code.\n"
            "2. Always import json and end with: "
            'print("___RESULT___" + json.dumps({"answer": <value>, "details": {...}}, default=str))\n'
        )
        if sandbox_path:
            retry_msg += f"3. Read the exact input file strictly from: {sandbox_path}\n"
        user_prompt += retry_msg

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": MAX_OUTPUT_TOKENS
        }
    }

    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            error_body = ""
        raise ConnectionError(f"Ollama HTTP error {e.code}: {error_body}")
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Could not connect to Ollama at {OLLAMA_HOST}.\n"
            f"Make sure Ollama is running (`ollama serve`).\n"
            f"Original error: {e}"
        )
    except TimeoutError as e:
        raise TimeoutError(
            f"Ollama did not respond within {TIMEOUT_SECONDS} seconds.\n"
            f"Original error: {e}"
        )

    raw_text = body.get("message", {}).get("content", "")
    if not raw_text.strip():
        raise RuntimeError("Ollama returned an empty response.")

    code = _extract_code(raw_text)
    code = _clean_and_repair_code(code)
    syntax_result = validate_python_syntax(code)

    return {
        "raw_response": raw_text,
        "code": code,
        "model": model,
        "syntax_valid": syntax_result["valid"],
        "syntax_error": syntax_result["error"]
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter query for LLM code generation (or press Enter for default): ").strip()
        if not query:
            query = "Calculate the average of 10, 20, and 30"
    print(f"Testing generate_code with query: {query!r}\n")
    res = generate_code(query, file_context=None)
    print("----- EXTRACTED CODE -----")
    print(res["code"])
    print("--------------------------")
    print("Syntax Valid:", res["syntax_valid"])
