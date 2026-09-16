"""
network_classifier.py

Single source of truth for "did this trajectory step touch the network,
and was that call local-only (offline-safe) or could it reach the public
internet." Backs the frontend's Network Monitor / Sovereignty Check view.

WHY THIS EXISTS (found live): the frontend used to decide "local" by
substring-matching the free-text network_target string for words like
"local" or "ollama". The coding agent's own trajectory event hardcoded
network_target="Docker Sandbox / Ollama" regardless of what LLM backend
actually generated the code -- so even after switching LLM_DEPLOYMENT_TYPE
to a real cloud provider for a demo, that ONE entry would keep reporting
"local" forever, because the label itself never changed. For a system
whose whole point is proving what is and isn't touching the network, a
stale/hardcoded label defeats the purpose.

The fix: every log_event() call now passes the REAL endpoint being hit (or
None if no network call was made at all), and this module classifies it by
actually parsing the target -- never by guessing from wording. Two shapes
are accepted:
  - A real URL ("http://localhost:11434/v1") -- classified by hostname.
  - An explicit, deliberately-authored non-network label from
    _KNOWN_LOCAL_LABELS below (e.g. "Docker Sandbox (network: none)" for
    a step that structurally cannot reach the network, like a container
    run with `--network none`) -- anything NOT in that exact set is
    treated as unclassifiable-as-local, so a new/unexpected label is
    surfaced as "needs verification" rather than silently assumed safe.
"""

from typing import Optional
from urllib.parse import urlparse

_LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

# Deliberately small and explicit. Only add a label here when it is
# genuinely, structurally impossible for that step to reach the network
# (e.g. a subprocess run with --network none) -- never as a shortcut for
# "this is probably fine".
_KNOWN_LOCAL_LABELS = {
    "Docker Sandbox (network: none)",
    "Local CPU",
    "local (instant)",
    # Supervisor fast paths: pure string / file-extension matching, with no
    # call of any kind. Listed explicitly rather than pattern-matched on the
    # word "local", which is the substring bug this module exists to avoid.
    "local (regex)",
    "local (file type)",
}


def _is_private_ip(host: str) -> bool:
    if host.startswith("127.") or host.startswith("10.") or host.startswith("192.168."):
        return True
    parts = host.split(".")
    if len(parts) == 4 and parts[0] == "172" and parts[1].isdigit():
        return 16 <= int(parts[1]) <= 31
    return False


def is_local_target(target: Optional[str]) -> bool:
    """True = offline-safe (loopback/local, or no network call at all).
    False = reaches (or could reach) an external network -- flagged in the
    UI as EXTERNAL DETECTED regardless of what the surrounding text says."""
    if not target:
        return True  # nothing logged -- pure local computation, e.g. context assembly

    if "://" in target:
        host = (urlparse(target).hostname or "").lower()
        return host in _LOCAL_HOSTNAMES or _is_private_ip(host)

    # Not a URL -- only an exact, explicitly-authored label counts as
    # local. A free-text description that happens to mention "local" or a
    # tool name is NOT enough (that substring-matching is exactly the bug
    # this module replaces).
    return target in _KNOWN_LOCAL_LABELS
