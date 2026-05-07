"""
ResearchFlow — Main Entry Point

Parses CLI arguments and invokes the Supervisor graph to answer
a research question against the ingested document corpus.
"""

import argparse

from agents.supervisor import DEFAULT_MAX_ITERATIONS, build_supervisor_graph
from dotenv import load_dotenv
from middleware.guardrails import detect_injection, sanitize_input
from middleware.pii_masking import mask_pii


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ResearchFlow: Adaptive Multi-Agent Research Assistant"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="The research question to answer.",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="default",
        help="User ID for cross-thread memory (Store interface).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable step-wise scratchpad logging.",
    )
    return parser.parse_args()


def main() -> None:
    """
    High-level flow:

    1. Load environment variables.
    2. Initialize the Supervisor graph (see agents/supervisor.py).
    3. Invoke the graph with the user's question.
    4. Print the structured research report.
    """
    load_dotenv()
    args = parse_args()

    # Initialize the Supervisor StateGraph
    app = build_supervisor_graph()

    # TODO: Build the initial graph state from args
    question = args.question
    if detect_injection(question):
        raise ValueError("Input rejected: possible prompt injection")
    question = sanitize_input(question)
    question = mask_pii(question)

    # Invoke graph with the cleaned question
    config = []  # TODO: replace w real config, if needed else remove
    result = app.invoke(
        {
            "user_question": question,
            "scratchpad": [],
            "iteration_count": 0,
            "max_iterations": DEFAULT_MAX_ITERATIONS,
        },
        config=config,
    )

    # TODO: Invoke the graph and collect the final state
    # TODO: Pretty-print the structured research report
    result = ""  # replace with supervisor final state
    answer = result["analysis"]["answer"]
    answer = mask_pii(answer)  # belt-and-suspenders on output too

    raise NotImplementedError("Wire up the Supervisor graph here.")


if __name__ == "__main__":
    main()
