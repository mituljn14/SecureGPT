from typing import List

from retrieval_pipeline.retrieval.schemas import RetrievedChunk
from retrieval_pipeline.retrieval.config import MAX_CONTEXT_CHUNKS

from retrieval_pipeline.context.formatter import format_chunk


def build_context(
    chunks: List[RetrievedChunk]
) -> str:

    if not chunks:
        return ""

    # Defense-in-depth cap: individual retrieval nodes already bound how
    # many chunks they hand off (e.g. multi_agent_retrieval_node trims to
    # 5), but this was declared in config.py and never actually enforced
    # anywhere -- so a future node that forgets to trim its own output
    # could silently blow up the prompt. This is the single choke point
    # every path funnels through before hitting the LLM.
    chunks = chunks[:MAX_CONTEXT_CHUNKS]

    context = []

    for i, chunk in enumerate(chunks, start=1):

        context.append(

            f"========== Context {i} ==========\n"

        )

        context.append(

            format_chunk(chunk)

        )

    return "\n\n".join(context)