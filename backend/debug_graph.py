import sys
import logging
logging.basicConfig(level=logging.DEBUG)
from retrieval_pipeline.graph.workflow import graph

print("Graph loaded.")

initial_state = {
    "query": "calculate net cash from operations from 2018-2022",
    "role": "CEO",
    "department": "Executive_Corporate",
    "trajectory_log": [],
    "uploaded_files": [],
    "generated_files": [],
    "model_override": None,
    "attempt": 0
}

print("Invoking graph...")
final_state = graph.invoke(initial_state)
print("SUCCESS!")
print(final_state.get('final_answer'))
