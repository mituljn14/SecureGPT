#!/usr/bin/env python3
"""
Runs the markdown conversion stage (pipeline/convert_all.py) on a RunPod
CPU pod instead of locally -- same code, same converters, just more cores
with none of the local thread-oversubscription problem (more room per
worker so pymupdf4llm's internal threads don't fight each other as badly).

Flow:
  1. Zip corpus/ (the in-scope, cleaned files) and pipeline/ locally.
  2. Create a 32 vCPU / 64GB CPU-only RunPod pod (same base image your
     existing runpod_pipeline.py already uses successfully for SSH).
  3. scp both zips up, set up a venv + deps, unzip, launch convert_all.py
     detached (nohup) so a dropped SSH connection doesn't kill the run.
  4. Poll every 45s: zip the pod's markdown/ + docs/conversion_log.jsonl,
     scp them down, extract into our local markdown/ and docs/ -- so you
     see files landing locally the same way as the local run did.
  5. On completion (DONE marker file) or Ctrl-C, do one final sync and
     terminate the pod (stop paying for it).

Reuses RUNPOD_API_KEY from ../SecProj/SecureGPT/backend/.env (same
account you already used for the Chandra run) unless RUNPOD_API_KEY is
already set in this project's own environment/.env.
"""
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
SECUREGPT_ENV = ROOT.parent / "SecProj" / "SecureGPT" / "backend" / ".env"

# Prefer a local .env if present, else fall back to the SecureGPT one.
if (ROOT / ".env").exists():
    load_dotenv(ROOT / ".env")
elif SECUREGPT_ENV.exists():
    load_dotenv(SECUREGPT_ENV)
    print(f"(loaded RUNPOD_API_KEY from {SECUREGPT_ENV})")

import runpod  # noqa: E402  (import after dotenv load so api key env var is set)

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    raise SystemExit("RUNPOD_API_KEY not found (checked project .env and SecureGPT's .env)")
runpod.api_key = RUNPOD_API_KEY

IMAGE_NAME = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"  # proven to expose SSH cleanly
POD_NAME = "SecureGPT-MD-Conversion-CPU"
CPU_INSTANCE_ID = "cpu3c-32-64"  # 3GHz Compute-Optimized, 32 vCPU / 64GB (verified against RunPod API)
REMOTE_WORKERS = 24  # leave headroom under 32 vCPU rather than 100% saturate

SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_rsa")


def sh(cmd, **kw):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kw)


def zip_dir(src: Path, dest_zip: Path, exclude_dirs=()):
    print(f"Zipping {src} -> {dest_zip} ...")
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in src.rglob("*"):
            if p.is_dir():
                continue
            if any(part in exclude_dirs for part in p.parts):
                continue
            zf.write(p, p.relative_to(src.parent))
    print(f"  {dest_zip.stat().st_size / 1e6:.1f} MB")


def wait_for_pod(pod_id):
    print(f"Waiting for pod {pod_id} to be running with ports exposed...")
    while True:
        pod = runpod.get_pod(pod_id)
        if pod.get("desiredStatus") == "RUNNING" and pod.get("runtime") and pod["runtime"].get("ports"):
            return pod
        time.sleep(5)


def main():
    corpus_zip = ROOT / "corpus_upload.zip"
    pipeline_zip = ROOT / "pipeline_upload.zip"

    zip_dir(ROOT / "corpus", corpus_zip)
    zip_dir(ROOT / "pipeline", pipeline_zip, exclude_dirs=("__pycache__",))

    print(f"\nCreating CPU pod ({CPU_INSTANCE_ID}, {POD_NAME}) ...")
    pod = runpod.create_pod(
        name=POD_NAME,
        image_name=IMAGE_NAME,
        gpu_type_id=None,           # CPU-only pod
        cloud_type="ALL",
        instance_id=CPU_INSTANCE_ID,
        volume_in_gb=60,
        container_disk_in_gb=60,
        ports="22/tcp",
        support_public_ip=True,
        start_ssh=True,
    )
    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")

    try:
        pod_info = wait_for_pod(pod_id)
        ssh_ip = ssh_port = None
        for p in pod_info["runtime"]["ports"]:
            if p["privatePort"] == 22:
                ssh_ip, ssh_port = p["ip"], p["publicPort"]
                break
        if not ssh_ip:
            raise RuntimeError("Could not get SSH IP/port for pod")

        print(f"SSH: ssh root@{ssh_ip} -p {ssh_port}")
        print("Waiting 20s for SSH daemon to come up...")
        time.sleep(20)

        ssh_base = ["ssh", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY_PATH, "-p", str(ssh_port), f"root@{ssh_ip}"]
        scp_up = lambda local, remote: sh(["scp", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY_PATH,
                                            "-P", str(ssh_port), str(local), f"root@{ssh_ip}:{remote}"])
        scp_down = lambda remote, local: subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY_PATH, "-P", str(ssh_port),
             f"root@{ssh_ip}:{remote}", str(local)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print("Uploading corpus + pipeline code ...")
        scp_up(corpus_zip, "/workspace/corpus_upload.zip")
        scp_up(pipeline_zip, "/workspace/pipeline_upload.zip")

        setup_cmds = [
            "cd /workspace",
            "unzip -qo corpus_upload.zip",
            "unzip -qo pipeline_upload.zip",
            "python3.10 -m venv venv",
            "venv/bin/pip install -q --upgrade pip",
            "venv/bin/pip install -q pymupdf4llm python-docx python-pptx openpyxl pandas tiktoken tqdm ftfy",
            f"nohup bash -c 'venv/bin/python pipeline/convert_all.py --workers {REMOTE_WORKERS}; "
            f"touch /workspace/pipeline_done' > convert.log 2>&1 < /dev/null & sleep 1",
        ]
        print("Setting up remote env and launching conversion (detached) ...")
        sh(ssh_base + [" && ".join(setup_cmds)])

        print("\nRunning on pod. Syncing markdown/ + log back every 45s (Ctrl-C to stop syncing; pod keeps running)...")
        while True:
            time.sleep(45)
            subprocess.run(ssh_base + ["cd /workspace && zip -rq sync_out.zip markdown docs/conversion_log.jsonl"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            scp_down("/workspace/sync_out.zip", ROOT / "sync_out.zip")
            if (ROOT / "sync_out.zip").exists():
                try:
                    with zipfile.ZipFile(ROOT / "sync_out.zip") as zf:
                        zf.extractall(ROOT)
                    (ROOT / "sync_out.zip").unlink()
                    n = sum(1 for _ in (ROOT / "markdown").rglob("*.md"))
                    print(f"  synced. {n} markdown files locally so far.")
                except Exception as e:
                    print(f"  sync parse failed (will retry next cycle): {e}")

            done = subprocess.run(ssh_base + ["test -f /workspace/pipeline_done && echo YES || echo NO"],
                                   capture_output=True, text=True)
            if "YES" in done.stdout:
                print("Remote conversion finished. Final sync done above.")
                break

    finally:
        print(f"Terminating pod {pod_id} ...")
        try:
            runpod.terminate_pod(pod_id)
            print("Pod terminated (billing stopped).")
        except Exception as e:
            print(f"! Could not auto-terminate pod {pod_id}: {e}. Terminate it manually in the RunPod console.")


if __name__ == "__main__":
    main()
