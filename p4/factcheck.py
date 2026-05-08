"""Test fact_checker agent functionality with this script"""

# Move to project root when using to avoid import adjustment
from agents.analyst import analyst_node
from agents.fact_checker import fact_checker_node
from agents.retriever import retriever_node
from dotenv import load_dotenv

load_dotenv()  # must run BEFORE agent imports

state = {
    "user_question": "What is Utilitarianism?",
    "plan": ["What is Utilitarianism?"],
    "current_task_index": 0,
}
state.update(retriever_node(state))
state.update(analyst_node(state))
state.update(fact_checker_node(state))

print("Overall confidence:", state["confidence_score"])
print("Needs HITL:", state["hitl_required"])
for v in state["fact_check_results"]["verdicts"]:
    print(f"  [{v['verdict']}] {v['claim'][:60]}...")
