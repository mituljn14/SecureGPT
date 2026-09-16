"""
Meeting Audio Agent — LangGraph Workflow
-----------------------------------------
Full pipeline:

  whisper_stt
      │
  speaker_diarization
      │
  rag_context          ← pulls verified enterprise data from Qdrant
      │
  summarize
      │
  decisions
      │
  action_items
      │
  build_minutes        ← writes DOCX to disk
      │
  index_transcript     ← chunks & indexes transcript into Qdrant (meeting_transcripts)
      │
     END
"""
from langgraph.graph import StateGraph, END

from meeting_agent.state import MeetingState
from meeting_agent.stt import whisper_stt_node
from meeting_agent.diarization import speaker_diarization_node
from meeting_agent.agents.meeting_agent import (
    rag_context_node,
    summarize_node,
    decisions_node,
    action_items_node,
)
from meeting_agent.minutes_builder import build_minutes_node
from meeting_agent.transcript_indexer import index_transcript_node

# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

builder = StateGraph(MeetingState)

builder.add_node("whisper_stt", whisper_stt_node)
builder.add_node("speaker_diarization", speaker_diarization_node)
builder.add_node("rag_context", rag_context_node)
builder.add_node("summarize", summarize_node)
builder.add_node("decisions", decisions_node)
builder.add_node("action_items", action_items_node)
builder.add_node("build_minutes", build_minutes_node)
builder.add_node("index_transcript", index_transcript_node)

# Entry
builder.set_entry_point("whisper_stt")

# Sequential edges — each node hands off to the next
builder.add_edge("whisper_stt", "speaker_diarization")
builder.add_edge("speaker_diarization", "rag_context")
builder.add_edge("rag_context", "summarize")
builder.add_edge("summarize", "decisions")
builder.add_edge("decisions", "action_items")
builder.add_edge("action_items", "build_minutes")
builder.add_edge("build_minutes", "index_transcript")
builder.add_edge("index_transcript", END)

# Compile
meeting_graph = builder.compile()
