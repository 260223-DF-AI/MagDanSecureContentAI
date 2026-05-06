import os
from dotenv import load_dotenv

load_dotenv()  # must run BEFORE agent imports

from agents.retriever import retriever_node
from agents.analyst import analyst_node
from agents.fact_checker import fact_checker_node

state = {
    "user_question": "What are uses of philosophy?",
    "plan": ["What are uses of philosophy?"],
    "current_task_index": 0,
}
state.update(retriever_node(state))
state.update(analyst_node(state))
state.update(fact_checker_node(state))

print("Overall confidence:", state["confidence_score"])
print("Needs HITL:", state["hitl_required"])
for v in state["fact_check_results"]["verdicts"]:
    print(f"  [{v['verdict']}] {v['claim'][:60]}...")
