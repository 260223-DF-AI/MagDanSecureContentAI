"""
ResearchFlow — RAGAS Evaluation Pipeline

Loads a golden dataset and runs a formal RAGAS evaluation measuring
faithfulness, answer relevancy, and context precision.

Usage:
    python scripts/evaluate.py --golden-dataset ./data/golden_dataset.json
    - OR -
    python -m scripts.evaluate --golden-dataset ./data/golden_dataset.json
"""

import argparse
import json
import time

from agents.supervisor import build_supervisor_graph
from datasets import Dataset
from dotenv import load_dotenv
from infrastructure.instances import _get_embedder, _get_llm
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation.")
    parser.add_argument(
        "--golden-dataset",
        type=str,
        required=True,
        help="Path to the golden dataset JSON file.",
    )
    return parser.parse_args()


def load_golden_dataset(filepath: str) -> list[dict]:
    """
    Load the golden dataset from a JSON file.

    Expected format: see data/golden_dataset.json for the schema.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_contexts(ctxs: list[str]) -> list[str]:
    """Remove text that indicates uncertainty."""
    return [
        c
        for c in ctxs
        if "no verbatim" not in c.lower()
        and "not directly mentioned" not in c.lower()
        and "cannot extract" not in c.lower()
    ]


def generate_predictions(dataset: list[dict]) -> list[dict]:
    """
    Run each question through the ResearchFlow pipeline and collect predictions.

    - For each entry in the dataset, invoke the Supervisor graph.
    - Capture the generated answer and the retrieved contexts.
    - Return a list of dicts with keys: question, answer, contexts.
    """
    graph = build_supervisor_graph()
    out = []
    for i, entry in enumerate(dataset):
        config = {"configurable": {"thread_id": f"eval-{i}"}}
        try:
            result = graph.invoke(
                {"user_question": entry["question"], "user_id": "evaluator"},
                config=config,
            )
        except Exception as e:
            print(f"  [warn] entry {i} failed: {e}")
            out.append({"question": entry["question"], "answer": "", "contexts": []})
            continue
        raw_contexts = [c["content"] for c in result.get("retrieved_chunks", [])]
        contexts = _clean_contexts(raw_contexts)
        analysis = result.get("analysis_output", "")
        out.append(
            {
                "question": entry["question"],
                "answer": analysis.get("answer", ""),
                "contexts": contexts,
                "reference": entry["reference"],
            }
        )
        print(f"  [{i + 1}/{len(dataset)}] done")
    return out


def run_ragas_evaluation(predictions: list[dict], golden: list[dict]) -> dict:
    """
    Evaluate predictions against the golden dataset using RAGAS.

    - Construct a RAGAS Dataset from predictions and ground truth.
    - Evaluate with metrics: faithfulness, answer_relevancy, context_precision.
    - Return a dict of metric_name → score.
    """
    ds = Dataset.from_list(predictions)
    _llm = LangchainLLMWrapper(_get_llm())
    _embedder = LangchainEmbeddingsWrapper(_get_embedder())

    result = evaluate(
        ds,
        llm=_llm,
        embeddings=_embedder,
        metrics=[
            Faithfulness(llm=_llm),
            AnswerRelevancy(llm=_llm, embeddings=_embedder),
            ContextPrecision(llm=_llm),
        ],
    )

    # Aggregate per-row scores to get mean scores
    import numpy as np

    scores = {}
    for metric, values in result._scores_dict.items():
        if isinstance(values, list):
            scores[metric] = float(np.nanmean(values))
        else:
            scores[metric] = float(values)
    return scores


def main() -> None:
    """Orchestrate the evaluation pipeline."""
    print(time.time())
    load_dotenv()
    args = parse_args()

    golden = load_golden_dataset(args.golden_dataset)
    predictions = generate_predictions(golden)
    results = run_ragas_evaluation(predictions, golden)

    # ---- Write to report file ----
    report_path = f"ragas_report{time.time()}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("📄 RAGAS Evaluation Report\n")
        f.write("=" * 50 + "\n\n")

        f.write("🔮 Predictions:\n")
        f.write("-" * 50 + "\n")
        f.write(str(predictions) + "\n\n")

        f.write("📊 Evaluation Results:\n")
        f.write("-" * 50 + "\n")
        for metric, score in results.items():
            f.write(f"{metric:<25} {score:.4f}\n")
        f.write("-" * 50 + "\n")

    # ---- Console output (optional) ----
    print("\n📊 RAGAS Evaluation Results:")
    print("-" * 40)
    for metric, score in results.items():
        print(f"  {metric:<25} {score:.4f}")
    print("-" * 40)

    print(f"\n📝 Full report saved to: {report_path}")


if __name__ == "__main__":
    main()
