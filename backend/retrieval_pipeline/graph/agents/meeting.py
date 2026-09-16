"""
Meeting agent node -- audio to minutes.

Wraps meeting_agent.workflow.meeting_graph so the supervisor can route an
uploaded recording through STT -> diarization -> RAG -> summary ->
decisions -> action items -> DOCX -> transcript indexing.

The meeting agent's own RAG step retrieves from securegpt_v2 with the
caller's role, so the minutes can only be enriched with documents that
caller is cleared to read.
"""
import os
import uuid

from retrieval_pipeline.graph.telemetry import log_event
from retrieval_pipeline.llm.provider import get_network_target

# Everything ffmpeg can decode for Whisper. Deliberately broad, because the
# extension a recording arrives with is not under the user's control: a
# re-downloaded "x.mp3" becomes "x.mp3 (1).mpeg" on Windows, messengers export
# .opus, and meeting tools (Zoom/Teams/Meet) hand out .mp4/.webm containers
# whose audio track transcribes fine.
#
# Missing an extension does not fail loudly -- the file just misses the audio
# fast-path, drifts to the enterprise agent, and gets "I don't have any
# information about the attached file". So err towards accepting.
AUDIO_EXTS = {
    # audio formats
    ".wav", ".mp3", ".m4a", ".m4b", ".flac", ".ogg", ".oga", ".opus",
    ".aac", ".wma", ".aiff", ".aif", ".amr", ".weba",
    # MPEG audio aliases -- what an .mp3 often becomes once a browser or
    # messenger rewrites the filename (audio/mpeg -> .mpeg).
    ".mpeg", ".mpga",
    # video containers commonly used for meeting recordings; ffmpeg pulls the
    # audio track out, which is all Whisper needs.
    ".mp4", ".webm", ".mkv", ".mov", ".avi",
}


def pick_audio_file(uploaded_files):
    """First uploaded file whose extension ffmpeg/Whisper can decode.

    Matches on the LAST extension, so "recording.mp3 (1).mpeg" is recognised
    by its real trailing suffix rather than being missed.
    """
    for f in uploaded_files or []:
        if os.path.splitext(str(f))[1].lower() in AUDIO_EXTS:
            return str(f)
    return None


def meeting_agent_node(state):
    print("=" * 80)
    print("Routed to: MEETING AGENT")
    print("=" * 80)

    audio = pick_audio_file(state.get("uploaded_files"))
    if not audio:
        state["answer"] = (
            "The meeting agent needs an audio recording. Attach a .wav, "
            ".mp3, .m4a or .flac file and ask again."
        )
        log_event(state, action="MEETING Agent",
                  details="No audio attached -- nothing to transcribe.",
                  network_target=None)
        return state

    try:
        from meeting_agent.workflow import meeting_graph
    except Exception as exc:
        state["answer"] = (
            f"The meeting agent is not available on this machine ({exc}). "
            "Install openai-whisper to enable it."
        )
        log_event(state, action="MEETING Agent",
                  details=f"Unavailable: {type(exc).__name__}: {exc}",
                  network_target=None)
        return state

    job_id = uuid.uuid4().hex[:12]
    try:
        final = meeting_graph.invoke({
            "audio_path": audio,
            "job_id": job_id,
            "user_role": state["role"],
            "user_department": state.get("department"),
            "meeting_title": os.path.splitext(os.path.basename(audio))[0],
        })
    except Exception as exc:
        state["answer"] = f"Meeting processing failed: {exc}"
        log_event(state, action="MEETING Agent",
                  details=f"Pipeline error: {type(exc).__name__}: {exc}",
                  network_target=get_network_target())
        return state

    summary = final.get("meeting_summary") or final.get("summary") or ""
    decisions = final.get("decisions") or []
    actions = final.get("action_items") or []
    minutes = final.get("minutes_path")

    parts = [f"**Meeting minutes — {os.path.basename(audio)}**"]
    if summary:
        parts.append("\n**Executive summary**\n" + str(summary))
    if decisions:
        parts.append("\n**Decisions**\n" + "\n".join(
            f"- {d.get('decision', d) if isinstance(d, dict) else d}" for d in decisions[:10]))
    if actions:
        lines = []
        for a in actions[:10]:
            if isinstance(a, dict):
                lines.append(f"- {a.get('task', a)} "
                             f"(owner: {a.get('owner', 'unassigned')}, "
                             f"due: {a.get('deadline', 'n/a')})")
            else:
                lines.append(f"- {a}")
        parts.append("\n**Action items**\n" + "\n".join(lines))
    if minutes:
        parts.append(f"\n**Minutes document**: {os.path.basename(str(minutes))}")

    state["answer"] = "\n".join(parts)
    state["generated_files"] = [str(minutes)] if minutes else []
    state["prompt"] = f"Meeting minutes for {os.path.basename(audio)}"

    log_event(
        state,
        action="MEETING Agent",
        details=(f"Transcribed {os.path.basename(audio)} "
                 f"({len(final.get('raw_transcript') or '')} chars): "
                 f"{len(decisions)} decision(s), {len(actions)} action item(s), "
                 f"minutes={'yes' if minutes else 'no'}."),
        network_target=get_network_target(),
    )
    return state
