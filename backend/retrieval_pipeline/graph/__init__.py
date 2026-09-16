"""
Graph package.

`graph` is exposed lazily rather than imported at package level.

Eagerly running `from .workflow import graph` here made importing ANY
submodule of this package (e.g. `retrieval_pipeline.graph.logger`) pull in
the entire workflow. That created a circular import the moment anything
reached llm.provider first:

    meeting_agent -> llm.provider -> graph.logger -> graph/__init__
      -> graph.workflow -> graph.nodes -> llm.provider  (still initialising)

It only stayed hidden because the API happens to import the workflow
before anything else. Deferring the import to first attribute access keeps
`from retrieval_pipeline.graph import graph` working while letting
submodules be imported on their own.
"""

__all__ = ["graph"]


def __getattr__(name):
    if name == "graph":
        from .workflow import graph as _graph
        return _graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
