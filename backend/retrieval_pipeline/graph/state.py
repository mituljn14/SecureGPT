from typing import TypedDict, List, Optional

from retrieval_pipeline.retrieval.schemas import RetrievedChunk


class GraphState(TypedDict):

    query: str
    role: str
    department: Optional[str]

    # Retrieval variables
    query_complexity: str
    # Raw dense cosine of the single best match (0..1) -- a real similarity,
    # set by fast_retrieval_node. NOT the RRF fusion score.
    retrieval_confidence: float
    # CRAG's LLM self-graded relevance (0..1). Kept separate from
    # retrieval_confidence on purpose: they are different signals on
    # different scales, and conflating them is what made the old cascade
    # impossible to reason about.
    crag_score: float
    rewritten_queries: List[str]
    retrieved_chunks: List[RetrievedChunk]
    # Which cascade stage produced the chunks currently in retrieved_chunks
    # ("fast" | "crag" | "multi_agent"), and the stage-by-stage record of
    # what each one measured and decided. Needed because two fallbacks make
    # "the last node that ran" a wrong answer: CRAG keeps the fast-path
    # chunks when grading finds nothing better, and multi-agent keeps CRAG's
    # window when verification rejects every candidate.
    retrieval_stage: str
    retrieval_path: List[dict]
    # Set by traced_node while a node runs; stamped onto trace events.
    _node: Optional[str]
    session_id: Optional[str]
    username: Optional[str]

    context: str

    agent_destination: str

    prompt: str

    answer: str
    
    uploaded_files: List[str]
    generated_files: List[str]
    # Python written by the coding agent, shown in the UI's Code tab.
    generated_code: Optional[str]
    # Verified {answer, details} from the sandbox, for metric cards.
    structured: Optional[dict]
    model_override: Optional[str]
    
    trajectory_log: List[dict]