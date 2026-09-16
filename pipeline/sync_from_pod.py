#!/usr/bin/env python3
"""
Standalone sync loop for the already-running RunPod conversion job
(pod was set up manually after runpod_convert.py's automated setup hit a
missing `unzip` on the base image -- this just does the polling/download
half of that script, pointed at the pod that's already converting).

Zips markdown/ + docs/conversion_log.jsonl on the pod, pulls it down,
extracts locally every 45s, until /workspace/pipeline_done appears --
then does one final sync and terminates the pod.
"""
import os
import subprocess
import time
import zipfile
from pathlib import Path

import runpod
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT.parent / "SecProj" / "SecureGPT" / "backend" / ".env")
runpod.api_key = os.getenv("RUNPOD_API_KEY")

POD_ID = "pwoouu0xqd72my"
SSH_IP = "213.173.105.83"
SSH_PORT = "16349"
SSH_KEY = os.path.expanduser("~/.ssh/id_rsa")

ssh_base = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
            "-i", SSH_KEY, "-p", SSH_PORT, f"root@{SSH_IP}"]


def sync_once():
    subprocess.run(ssh_base + ["cd /workspace && zip -rq sync_out.zip markdown docs/conversion_log.jsonl"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    r = subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY, "-P", SSH_PORT,
                         f"root@{SSH_IP}:/workspace/sync_out.zip", str(ROOT / "sync_out.zip")],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    if r.returncode != 0 or not (ROOT / "sync_out.zip").exists():
        return None
    try:
        with zipfile.ZipFile(ROOT / "sync_out.zip") as zf:
            zf.extractall(ROOT)
        (ROOT / "sync_out.zip").unlink()
    except Exception as e:
        print(f"  ! sync extract failed: {e}")
        return None
    n = sum(1 for _ in (ROOT / "markdown").rglob("*.md"))
    return n


def is_done():
    r = subprocess.run(ssh_base + ["test -f /workspace/pipeline_done && echo YES || echo NO"],
                        capture_output=True, text=True, timeout=30)
    return "YES" in r.stdout


def main():
    print(f"Syncing from pod {POD_ID} ({SSH_IP}:{SSH_PORT}) every 45s ...")
    while True:
        n = sync_once()
        if n is not None:
            print(f"  synced. {n} markdown files locally so far.")
        else:
            print("  sync failed this cycle, will retry.")

        if is_done():
            print("Remote conversion finished! Doing one more final sync...")
            sync_once()
            break

        time.sleep(45)

    print(f"Terminating pod {POD_ID} ...")
    try:
        runpod.terminate_pod(POD_ID)
        print("Pod terminated (billing stopped).")
    except Exception as e:
        print(f"! Could not auto-terminate: {e}. Terminate manually in the RunPod console.")


if __name__ == "__main__":
    main()
