"""
Start a RunPod GPU pod serving a large judge model for RAGAS evaluation.

WHY THIS EXISTS
---------------
RAGAS grades answers by asking an LLM to emit structured JSON. It drives that
through instructor in `Mode.JSON` (ragas/llms/base.py), which pastes the
Pydantic JSON Schema into the prompt instead of using the tool-calling API. A
3B model handed a schema frequently echoes the schema back rather than
instantiating it -- observed here as:

    validation error for StatementGeneratorOutput
    statements  Field required
    input_value={'description': 'Structured output for statement generation.',
                 'properties': {...}, 'type': 'object'}

...on 3 of 5 scored rows, failing all four retries. Judging needs a bigger
model than serving does, and a bigger model needs a GPU.

WHAT THIS DOES NOT TOUCH
------------------------
Nothing in the pipeline. This starts a pod, serves a model over an
OpenAI-compatible API, and prints the three environment variables that point
the *judge* at it. `retrieval_pipeline` keeps running on local Ollama exactly
as before; if you never set those variables, the eval behaves as it does now.

USAGE
-----
    python -m evaluation.runpod_judge start
    python -m evaluation.runpod_judge stop <pod_id>

Follows the same conventions as pipeline/runpod_embed.py: RUNPOD_API_KEY from
the environment, GPU type from RUNPOD_GPU.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

# vLLM exposes an OpenAI-compatible /v1 endpoint, which is exactly what
# _build_ragas_llm() expects -- no adapter needed on either side.
IMAGE = os.getenv("RAGAS_JUDGE_IMAGE", "vllm/vllm-openai:latest")
GPU_TYPE = os.getenv("RUNPOD_GPU", "NVIDIA GeForce RTX 4090")
JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
POD_NAME = os.getenv("RAGAS_JUDGE_POD_NAME", "securegpt-ragas-judge")
HTTP_PORT = 8000


def _load_api_key():
    key = os.getenv("RUNPOD_API_KEY")
    if key:
        return key
    # Same fallback runpod_embed.py uses: read it out of backend/.env.
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("RUNPOD_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def _runpod():
    try:
        import runpod
    except ImportError:
        raise SystemExit(
            "The runpod SDK is not installed in this venv.\n"
            "  pip install runpod"
        )
    key = _load_api_key()
    if not key:
        raise SystemExit("RUNPOD_API_KEY not found in the environment or backend/.env")
    runpod.api_key = key
    return runpod


def start():
    runpod = _runpod()

    print(f"Creating GPU pod ({GPU_TYPE}) serving {JUDGE_MODEL} ...", flush=True)
    pod = runpod.create_pod(
        name=POD_NAME,
        image_name=IMAGE,
        gpu_type_id=GPU_TYPE,
        cloud_type="ALL",
        gpu_count=1,
        volume_in_gb=80,
        container_disk_in_gb=80,
        ports=f"{HTTP_PORT}/http,22/tcp",
        support_public_ip=True,
        start_ssh=True,
        docker_args=(
            f"--model {JUDGE_MODEL} --port {HTTP_PORT} "
            f"--max-model-len 8192 --gpu-memory-utilization 0.92"
        ),
    )
    pod_id = pod["id"]
    print(f"pod id: {pod_id}")

    endpoint = None
    print("Waiting for the HTTP port ...", flush=True)
    while True:
        info = runpod.get_pod(pod_id)
        rt = info.get("runtime")
        if info.get("desiredStatus") == "RUNNING" and rt and rt.get("ports"):
            # RunPod's HTTP proxy reports isIpPublic=False for /http ports
            # while still routing them, so waiting on that flag hangs forever.
            # The proxy hostname is derived from the pod id; readiness is
            # decided by asking the endpoint itself.
            if any(p["privatePort"] == HTTP_PORT for p in rt["ports"]):
                endpoint = f"https://{pod_id}-{HTTP_PORT}.proxy.runpod.net/v1"
                break
        time.sleep(6)

    print()
    print("Pod is up. vLLM still has to download and load the weights -- that")
    print("can take several minutes before the endpoint answers.")
    print()
    print("Point the judge at it (PowerShell):")
    print(f'  $env:RAGAS_JUDGE_API_BASE="{endpoint}"')
    print(f'  $env:RAGAS_JUDGE_MODEL="{JUDGE_MODEL}"')
    print('  $env:RAGAS_JUDGE_API_KEY="EMPTY"')
    print()
    print("Then run the eval as usual:")
    print("  .\\venv\\Scripts\\python.exe -u -m evaluation.ragas_eval --only enterprise")
    print()
    print(f"STOP THE POD WHEN DONE -- it bills by the second:")
    print(f"  python -m evaluation.runpod_judge stop {pod_id}")
    return pod_id


def stop(pod_id):
    runpod = _runpod()
    runpod.terminate_pod(pod_id)
    print(f"terminated {pod_id}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "start"
    if cmd == "start":
        start()
    elif cmd == "stop":
        if len(sys.argv) < 3:
            raise SystemExit("usage: runpod_judge stop <pod_id>")
        stop(sys.argv[2])
    else:
        raise SystemExit(f"unknown command: {cmd}")
