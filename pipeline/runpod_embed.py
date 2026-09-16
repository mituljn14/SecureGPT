#!/usr/bin/env python3
"""
Launch + drive the BGE-M3 embedding run on a RunPod GPU pod.

    python pipeline/runpod_embed.py                 # create pod, upload, run, sync
    python pipeline/runpod_embed.py --sync-only --pod-id X --ip Y --port Z

Flow:
  1. create a GPU pod (defaults to RTX 4090 -- ample for BGE-M3 fp16)
  2. scp embed_input.jsonl.gz (61MB) + embed_pod.py
  3. pip install FlagEmbedding into the image's EXISTING python, which
     already has a CUDA build of torch. Installing into a fresh venv would
     pull a CPU-only torch and silently run ~50x slower.
  4. launch detached under nohup so a dropped SSH does not kill the run
  5. poll every 60s, pulling completed shards down to embeddings/

Shards are written atomically on the pod, so a sync can never grab a
half-written file.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
for env in (ROOT / ".env", ROOT.parent / "SecProj" / "SecureGPT" / "backend" / ".env"):
    if env.exists():
        load_dotenv(env)
        break

import runpod  # noqa: E402

runpod.api_key = os.getenv("RUNPOD_API_KEY")
if not runpod.api_key:
    raise SystemExit("RUNPOD_API_KEY not found")

GPU_TYPE = os.getenv("RUNPOD_GPU", "NVIDIA GeForce RTX 4090")
IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
POD_NAME = "SecureGPT-BGEM3-Embed"
SSH_KEY = os.path.expanduser("~/.ssh/id_rsa")

INPUT_GZ = ROOT / "chunks" / "embed_input.jsonl.gz"
POD_SCRIPT = ROOT / "pipeline" / "embed_pod.py"
LOCAL_SHARDS = ROOT / "embeddings"


def ssh_cmd(ip, port):
    return ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20",
            "-i", SSH_KEY, "-p", str(port), f"root@{ip}"]


def run_ssh(ip, port, command, check=True, timeout=None):
    return subprocess.run(ssh_cmd(ip, port) + [command], check=check, timeout=timeout)


def wait_for_pod(pod_id):
    print(f"Waiting for pod {pod_id} ...", flush=True)
    while True:
        pod = runpod.get_pod(pod_id)
        rt = pod.get("runtime")
        if pod.get("desiredStatus") == "RUNNING" and rt and rt.get("ports"):
            for p in rt["ports"]:
                if p["privatePort"] == 22:
                    return p["ip"], p["publicPort"]
        time.sleep(5)


def sync_shards(ip, port):
    LOCAL_SHARDS.mkdir(exist_ok=True)
    r = subprocess.run(ssh_cmd(ip, port) + ["ls /workspace/shards/ 2>/dev/null || true"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    remote = [f for f in (r.stdout or "").split() if f.endswith(".npz")]
    have = {p.name for p in LOCAL_SHARDS.glob("*.npz")}
    todo = [f for f in remote if f not in have]
    for f in todo:
        subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY,
                        "-P", str(port), f"root@{ip}:/workspace/shards/{f}",
                        str(LOCAL_SHARDS / f)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1200)
    return len(have) + len(todo), len(todo)


def is_done(ip, port):
    r = subprocess.run(ssh_cmd(ip, port) + ["test -f /workspace/EMBED_DONE && echo YES || echo NO"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    return "YES" in (r.stdout or "")


def tail_log(ip, port, n=6):
    """Only the shard progress lines -- the raw log is full of tqdm bars whose
    bytes are undecodable under the Windows default codepage and which would
    drown the console anyway."""
    try:
        r = subprocess.run(
            ssh_cmd(ip, port) + [f"grep -a 'chunks in' /workspace/embed.log 2>/dev/null | tail -n {n} || true"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync-only", action="store_true")
    ap.add_argument("--pod-id"); ap.add_argument("--ip"); ap.add_argument("--port")
    ap.add_argument("--keep-alive", action="store_true", help="don't terminate pod at the end")
    ap.add_argument("--shard-size", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=96)
    args = ap.parse_args()

    if args.sync_only:
        pod_id, ip, port = args.pod_id, args.ip, args.port
    else:
        print(f"Creating GPU pod ({GPU_TYPE}) ...", flush=True)
        pod = runpod.create_pod(
            name=POD_NAME, image_name=IMAGE, gpu_type_id=GPU_TYPE,
            cloud_type="ALL", gpu_count=1,
            volume_in_gb=80, container_disk_in_gb=80,
            ports="22/tcp", support_public_ip=True, start_ssh=True,
        )
        pod_id = pod["id"]
        print(f"pod_id={pod_id}", flush=True)
        ip, port = wait_for_pod(pod_id)
        print(f"SSH: root@{ip} -p {port}", flush=True)
        time.sleep(20)

        print("Uploading input + script ...", flush=True)
        for local, remote in ((INPUT_GZ, "/workspace/embed_input.jsonl.gz"),
                              (POD_SCRIPT, "/workspace/embed_pod.py")):
            subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY,
                            "-P", str(port), str(local), f"root@{ip}:{remote}"],
                           check=True, timeout=3600)

        print("Installing FlagEmbedding (into the CUDA-enabled system python) ...", flush=True)
        run_ssh(ip, port,
                "pip install -q --no-cache-dir 'FlagEmbedding>=1.2.10' && "
                "python -c \"import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())\"",
                timeout=2400)

        print("Launching embedding (detached) ...", flush=True)
        run_ssh(ip, port,
                f"cd /workspace && nohup python embed_pod.py --shard-size {args.shard_size} "
                f"--batch-size {args.batch_size} > embed.log 2>&1 < /dev/null & sleep 2; echo LAUNCHED",
                check=False, timeout=120)

    print(f"\nSyncing shards to {LOCAL_SHARDS} every 60s ...\n", flush=True)
    try:
        while True:
            total, new = sync_shards(ip, port)
            log = tail_log(ip, port, 3)
            print(f"[{time.strftime('%H:%M:%S')}] shards local={total} (+{new})", flush=True)
            if log:
                print("   " + log.replace("\n", "\n   "), flush=True)
            if is_done(ip, port):
                print("\nRemote embedding finished. Final sync ...", flush=True)
                sync_shards(ip, port)
                break
            time.sleep(60)
    finally:
        if not args.keep_alive and pod_id:
            print(f"Terminating pod {pod_id} ...", flush=True)
            try:
                runpod.terminate_pod(pod_id)
                print("Pod terminated (billing stopped).", flush=True)
            except Exception as e:
                print(f"! terminate failed: {e} -- do it in the RunPod console.", flush=True)


if __name__ == "__main__":
    main()
