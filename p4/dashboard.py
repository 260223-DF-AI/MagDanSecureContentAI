"""
ResearchFlow Mission Control Dashboard

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

import streamlit as st

from agents.supervisor import (
    DEFAULT_MAX_ITERATIONS,
    build_supervisor_graph,
    build_thread_config,
    get_thread_history,
    resume_from_hitl,
)


st.set_page_config(
    page_title="ResearchFlow Mission Control",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ ResearchFlow Mission Control")
st.caption("Multi-Agent Research Assistant — LangGraph + RAG + HITL")


@st.cache_resource
def load_graph():
    """Compile the supervisor graph once for the Streamlit session."""
    return build_supervisor_graph()


app = load_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"mission-control-{uuid4()}"

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "question" not in st.session_state:
    st.session_state.question = ""


# =========================
# Sidebar Controls
# =========================

with st.sidebar:
    st.header("Mission Controls")

    st.session_state.question = st.text_area(
        "Research Question",
        value=st.session_state.question,
        height=120,
        placeholder="Example: What is utilitarianism?",
    )

    max_iterations = st.slider(
        "Max Self-Refinement Iterations",
        min_value=1,
        max_value=5,
        value=DEFAULT_MAX_ITERATIONS,
    )

    run_button = st.button("🚀 Launch ResearchFlow", use_container_width=True)

    st.divider()

    st.write("**Thread ID**")
    st.code(st.session_state.thread_id)

    if st.button("🔄 New Mission Thread", use_container_width=True):
        st.session_state.thread_id = f"mission-control-{uuid4()}"
        st.session_state.latest_result = None
        st.rerun()


# =========================
# Run Graph
# =========================

if run_button:
    if not st.session_state.question.strip():
        st.warning("Enter a research question first.")
    else:
        with st.spinner("ResearchFlow agents are running..."):
            config = build_thread_config(st.session_state.thread_id)

            result = app.invoke(
                {
                    "user_question": st.session_state.question,
                    "messages": [],
                    "scratchpad": [],
                    "iteration_count": 0,
                    "max_iterations": max_iterations,
                },
                config=config,
            )

            st.session_state.latest_result = result


result = st.session_state.latest_result


# =========================
# Empty State
# =========================

if result is None:
    st.info("Enter a question and launch ResearchFlow to view the mission dashboard.")
    st.stop()


# =========================
# Dashboard Data
# =========================

confidence = result.get("confidence_score", 0.0)
iteration_count = result.get("iteration_count", 0)
hitl_required = result.get("hitl_required", False)
critique_decision = result.get("critique_decision", "unknown")
current_task = result.get("current_task") or {}
active_agent = current_task.get("task_type", "complete")

retrieved_chunks = result.get("retrieved_chunks", [])
scratchpad = result.get("scratchpad", [])
messages = result.get("messages", [])
fact_check_results = result.get("fact_check_results", {})
final_answer = result.get("final_answer", "")


# =========================
# Top Metrics
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Active Agent", active_agent.title())

with col2:
    st.metric("Confidence", f"{confidence:.2f}")

with col3:
    st.metric("Retry Count", iteration_count)

with col4:
    st.metric("HITL Status", "Required" if hitl_required else "Not Required")


# =========================
# Agent Status Row
# =========================

st.subheader("Agent Pipeline")

agents = [
    ("retrieve", "📚 Retriever"),
    ("analyze", "🧠 Analyst"),
    ("fact_check", "🕵️ Fact Checker"),
    ("critique", "🧾 Critique Node"),
]

agent_cols = st.columns(len(agents))

for col, (key, label) in zip(agent_cols, agents):
    with col:
        if active_agent == key:
            st.success(f"ACTIVE\n\n{label}")
        else:
            st.info(label)


# =========================
# Main Dashboard Layout
# =========================

left, middle, right = st.columns([1.2, 1.4, 1.2])


# =========================
# Retrieved Sources Panel
# =========================

with left:
    st.subheader("📚 Retrieved Sources")

    if not retrieved_chunks:
        st.write("No retrieved chunks yet.")
    else:
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            filename = (
                chunk.get("filename")
                or chunk.get("metadata", {}).get("filename")
                or chunk.get("source")
                or "Unknown Source"
            )

            score = (
                chunk.get("relevance_score")
                or chunk.get("score")
                or 0.0
            )

            text = (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("metadata", {}).get("text")
                or chunk.get("metadata", {}).get("source")
                or ""
            )

            with st.expander(f"{idx}. {filename} — score {score:.3f}"):
                st.write(text[:1200])


# =========================
# Final Answer + Reasoning Trace
# =========================

with middle:
    st.subheader("🧠 Final Research Output")

    if final_answer:
        st.write(final_answer)
    else:
        analysis_output = result.get("analysis_output", {})
        if isinstance(analysis_output, dict):
            st.write(analysis_output.get("answer", "No final answer yet."))
        else:
            st.write(analysis_output or "No final answer yet.")

    st.divider()

    st.subheader("🧭 Reasoning Trace / Scratchpad")

    if not scratchpad:
        st.write("No scratchpad entries yet.")
    else:
        for idx, entry in enumerate(scratchpad, start=1):
            st.markdown(f"**Step {idx}:** {entry}")


# =========================
# HITL + Fact Check Panel
# =========================

with right:
    st.subheader("🧑‍⚖️ HITL Approval Box")

    if hitl_required:
        st.warning("Human review is required.")

        approval_choice = st.selectbox(
            "Reviewer Action",
            ["approve", "revise", "retry_retriever", "retry_analyst"],
        )

        revised_answer = ""

        if approval_choice == "revise":
            revised_answer = st.text_area(
                "Human revised final answer",
                height=160,
            )

        feedback = st.text_area(
            "Reviewer feedback",
            placeholder="Optional reviewer notes...",
        )

        if st.button("Submit Human Review", use_container_width=True):
            reviewer_feedback = {
                "action": approval_choice,
                "feedback": feedback,
            }

            if approval_choice == "revise":
                reviewer_feedback["final_answer"] = revised_answer

            with st.spinner("Resuming graph after HITL review..."):
                updated_result = resume_from_hitl(
                    app=app,
                    thread_id=st.session_state.thread_id,
                    reviewer_feedback=reviewer_feedback,
                )

                st.session_state.latest_result = updated_result
                st.rerun()

    else:
        st.success("No human review required.")

    st.divider()

    st.subheader("🕵️ Fact Check Results")

    if fact_check_results:
        st.json(fact_check_results)
    else:
        st.write("No fact-check results yet.")

    st.divider()

    st.subheader("💬 Message Window")

    if messages:
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            st.markdown(f"**{role}:** {content[:500]}")
    else:
        st.write("No message history yet.")


# =========================
# Checkpoint Timeline
# =========================

st.divider()
st.subheader("⏱️ Checkpoint Timeline")

try:
    history = get_thread_history(app, st.session_state.thread_id)

    if not history:
        st.write("No checkpoints yet.")
    else:
        timeline_cols = st.columns(min(len(history), 6))

        for idx, checkpoint in enumerate(history):
            with timeline_cols[idx % len(timeline_cols)]:
                st.caption(f"Checkpoint {idx}")
                st.code(f"next={checkpoint.next}")

except Exception as exc:
    st.warning(f"Could not load checkpoint history: {exc}")


# =========================
# Raw State Debugger
# =========================

with st.expander("Raw Graph State Debugger"):
    st.json(result)