"""
ResearchFlow — Main Entry Point

Parses CLI arguments and invokes the Supervisor graph to answer
a research question against the ingested document corpus.
"""

import argparse
import os

from dotenv import load_dotenv


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

    # TODO: Initialize the Supervisor StateGraph
    # TODO: Build the initial graph state from args
    # TODO: Invoke the graph and collect the final state
    # TODO: Pretty-print the structured research report

    raise NotImplementedError("Wire up the Supervisor graph here.")


if __name__ == "__main__":
    main()
