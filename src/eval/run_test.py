import argparse
import asyncio
import logging
from typing import Any

from langsmith import Client
from langsmith.evaluation import aevaluate
from langsmith.schemas import Example

from src.eval.graders import (
    eval_answer_relevancy,
    eval_contextual_recall,
    eval_faithfulness,
    evaluate_precision_at_1,
    evaluate_precision_at_3,
    evaluate_precision_at_10,
    evaluate_recall_at_1,
    evaluate_recall_at_3,
    evaluate_recall_at_10,
    evaluate_word_f1,
)
from src.eval.wrappers import rag_eval_wrapper

logger = logging.getLogger(__name__)


async def run_ab_experiment(
    dataset_name: str,
    experiment_prefix: str,
    config_overrides: dict[str, Any],
) -> None:
    """
    Runs an evaluation for RAG  using LangSmith's evaluation framework.

    Args:
        dataset_name: Имя датасета в LangSmith (напр. "RAG_Gold_Benchmark_v1").
        experiment_prefix: Название для A/B теста (напр. "v2_large_chunking").
        config_overrides: Конфиги (температура, промпт), которые мы прокидываем в пайплайн.
    """

    eval_data: str | list[Example]

    logger.info(f"🚀 PROD MODE: Running full benchmark on {dataset_name}...")
    eval_data = dataset_name
    logger.info(f"Starting A/B test: {experiment_prefix} on dataset {dataset_name}")

    experiment_results = await aevaluate(
        rag_eval_wrapper,
        data=eval_data,
        evaluators=[
            eval_contextual_recall,
            eval_faithfulness,
            eval_answer_relevancy,
            evaluate_word_f1,
            evaluate_precision_at_1,
            evaluate_precision_at_3,
            evaluate_precision_at_10,
            evaluate_recall_at_1,
            evaluate_recall_at_3,
            evaluate_recall_at_10,
        ],
        experiment_prefix=experiment_prefix,
        metadata=config_overrides,
        max_concurrency=2,
    )

    logger.info(
        f"Experiment {experiment_prefix} completed! View results in LangSmith UI."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run RAG evaluation experiment with specified dataset and configuration."
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Prefix for the experiment name (e.g. 'v2_large_chunking'). "
        "This helps identify the experiment in LangSmith.",
    )

    args = parser.parse_args()

    asyncio.run(
        run_ab_experiment(
            dataset_name="RAG_Gold_Benchmark_v1",
            experiment_prefix=args.prefix,
            config_overrides={"temperature": 0.2, "retriever_k": 5},
        )
    )
