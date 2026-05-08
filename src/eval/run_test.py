import argparse
import asyncio
import logging
import os
from typing import Any

import mlflow
import pandas as pd
from dotenv import load_dotenv
from langsmith.evaluation import aevaluate

from src.eval.graders import (
    eval_answer_relevancy,
    eval_contextual_recall,
    eval_faithfulness,
    evaluate_document_precision,
    evaluate_document_recall,
    evaluate_word_f1,
)
from src.eval.wrappers import make_rag_eval_wrapper

logger = logging.getLogger(__name__)
load_dotenv()

# 5000 for API, 5050 for UI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "SEC_10K_RAG_Optimization")

# Initialize MLflow connection
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


async def run_ab_experiment(
    dataset_name: str,
    experiment_prefix: str,
    config_overrides: dict[str, Any],
) -> None:
    """
    Runs a RAG evaluation using LangSmith and logs hyperparameters,
    metadata, and aggregated metrics to MLflow for A/B testing.

    Args:
        dataset_name: Name of the dataset in LangSmith (e.g. "RAG_Gold_Benchmark_v1").
        experiment_prefix: Name for the A/B test run (e.g. "v2_large_chunking").
        config_overrides: Configs (temperature, retriever_k) passed to the pipeline.
    """
    logger.info(f"🚀 PROD MODE: Running full benchmark on {dataset_name}...")
    logger.info(f"Starting A/B test: '{experiment_prefix}'")

    # Start an MLflow tracking run
    with mlflow.start_run(run_name=experiment_prefix) as run:
        # ---------------------------------------------------------
        # 1. LOG INPUTS (Hyperparameters & Metadata)
        # ---------------------------------------------------------
        logger.info("Logging hyperparameters to MLflow...")
        mlflow.log_params(config_overrides)
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param(
            "embedding_model", "text-embedding-3-small"
        )  # Add static context

        # ---------------------------------------------------------
        # 2. EXECUTE EVALUATION (LangSmith)
        # ---------------------------------------------------------
        logger.info("Triggering LangSmith async evaluation...")
        try:
            custom_eval_wrapper = make_rag_eval_wrapper(config_overrides)
            experiment_results = await aevaluate(
                custom_eval_wrapper,
                data=dataset_name,
                evaluators=[
                    eval_contextual_recall,
                    eval_faithfulness,
                    eval_answer_relevancy,
                    evaluate_word_f1,
                    evaluate_document_precision,
                    evaluate_document_recall,
                ],
                experiment_prefix=experiment_prefix,
                # Wrap config_overrides in a 'config' dict so LangGraph's RunnableConfig can extract it
                metadata={"config": config_overrides},
                max_concurrency=4,
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            mlflow.set_tag("status", "failed")
            raise e

        # ---------------------------------------------------------
        # 3. PROCESS & LOG OUTPUTS (Metrics to MLflow)
        # ---------------------------------------------------------
        logger.info("Parsing LangSmith results and computing aggregate metrics...")

        # Convert LangSmith results to Pandas DataFrame for easy aggregation
        df: pd.DataFrame = experiment_results.to_pandas()
        metrics_to_log = {}

        # Extract columns starting with "feedback." (LangSmith's default for metrics)
        for col in df.columns:
            if col.startswith("feedback."):
                # Clean name: "feedback.Recall@3" -> "Recall@3"
                clean_metric_name = col.replace("feedback.", "")
                clean_metric_name = clean_metric_name.replace("@", "_at_")
                # Calculate the mean across all dataset examples
                mean_value = df[col].dropna().mean()
                metrics_to_log[clean_metric_name] = float(mean_value)

                # Calculate standard deviation to measure stability/variance
                std_value = df[col].dropna().std()
                if pd.notna(std_value):
                    metrics_to_log[f"{clean_metric_name}_std"] = float(std_value)

                logger.info(
                    f"   -> {clean_metric_name}: {mean_value:.4f} (std: {std_value:.4f})"
                )

        # Bulk log all aggregated metrics to MLflow
        if metrics_to_log:
            mlflow.log_metrics(metrics_to_log)
        else:
            logger.warning(
                "No metrics found to log. Did the evaluators run successfully?"
            )

        # ---------------------------------------------------------
        # 4. TRACEABILITY (Link MLflow to LangSmith)
        # ---------------------------------------------------------
        langsmith_url = (
            f"https://smith.langchain.com/projects/p?name={experiment_prefix}"
        )
        mlflow.set_tag("langsmith_project_url", langsmith_url)
        mlflow.set_tag("status", "completed")

        logger.info(f"✅ Experiment '{experiment_prefix}' completed!")
        logger.info(f"🔗 MLflow Run ID: {run.info.run_id}")
        logger.info(
            "View detailed traces in LangSmith UI and aggregated metrics in MLflow UI."
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Run RAG evaluation experiment with specified dataset and configuration."
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Prefix for the experiment name (e.g. 'v2_large_chunking'). "
        "This helps identify the experiment in LangSmith and MLflow.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to use for evaluation.",
    )

    args = parser.parse_args()

    asyncio.run(
        run_ab_experiment(
            dataset_name=args.dataset,
            experiment_prefix=args.prefix,
            config_overrides={"temperature": 0.2, "retriever_k": 5},
        )
    )
