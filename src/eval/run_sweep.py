import argparse
import asyncio
import logging
import subprocess
import sys
from typing import Any, Dict, List

from src.eval.run_test import run_ab_experiment

logging.basicConfig(level=logging.INFO)


# Top-K and Temperature (Generation constraints)
generation_experiments: List[Dict[str, Any]] = [
    {"prefix": "Gen_K3_Temp0", "config": {"retriever_k": 3, "temperature": 0.0}},
    {"prefix": "Gen_K5_Temp0", "config": {"retriever_k": 5, "temperature": 0.0}},
    {"prefix": "Gen_K10_Temp0", "config": {"retriever_k": 10, "temperature": 0.0}},
    {"prefix": "Gen_K5_Temp0.5", "config": {"retriever_k": 5, "temperature": 0.5}},
]

# Search Algorithms (Retrieval performance)
retrieval_experiments: List[Dict[str, Any]] = [
    {
        "prefix": "Search_Dense_Only_K5",
        "config": {"search_algorithm": "dense", "retriever_k": 5, "critic": False},
    },
    {
        "prefix": "Search_BM25_Only_K5",
        "config": {"search_algorithm": "bm25", "retriever_k": 5, "critic": False},
    },
    {
        "prefix": "Search_Hybrid_RRF_K5",
        "config": {"search_algorithm": "hybrid", "retriever_k": 5, "critic": False},
    },
]

# Prompt Engineering (Agent behavior)
prompt_experiments: List[Dict[str, Any]] = [
    {
        "prefix": "Prompt_v3_Baseline",
        "config": {
            "prompt_version": "v3_baseline",
            "search_algorithm": "hybrid",
            "retriever_k": 5,
            "collection_name": "sec_reports",
            "critic": False,
        },
    },
    {
        "prefix": "Prompt_v4_Avoid_Duplicate_Searches",
        "config": {
            "prompt_version": "v4_avoid_duplicate_searches",
            "search_algorithm": "hybrid",
            "retriever_k": 5,
            "collection_name": "sec_reports",
            "critic": False,
        },
    },
]

# Planner vs No Planner
planner_experiments: List[Dict[str, Any]] = [
    {
        "prefix": "Agent_Standard_Single_Shot",
        "config": {
            "prompt_version": "v4_strict_metadata",
            "use_planner": False,
            "retriever_k": 5,
            "search_algorithm": "hybrid",
            "collection_name": "sec_reports",
        },
    },
    {
        "prefix": "Agent_With_Query_Decomposition",
        "config": {
            "prompt_version": "v4_strict_metadata",
            "use_planner": True,
            "retriever_k": 5,
            "search_algorithm": "hybrid",
            "collection_name": "sec_reports",
        },
    },
]

critic_experiments: List[Dict[str, Any]] = [
    {
        "prefix": "Agent_Critic_Disabled_Fast",
        "config": {
            "use_critic": False,
            "prompt_version": "v4_strict_metadata",  # Use your best prompt here
            "retriever_k": 5,  # Use your best K here
            "search_algorithm": "hybrid",  # Use your best search here
            "collection_name": "sec_reports",  # Use your best chunking here
        },
    },
    {
        "prefix": "Agent_Critic_Enabled_Accurate",
        "config": {
            "use_critic": True,
            "prompt_version": "v4_strict_metadata",
            "retriever_k": 5,
            "search_algorithm": "hybrid",
            "collection_name": "sec_reports",
        },
    },
]

#  Chunking Configs
chunking_configs: List[Dict[str, Any]] = [
    {"size": 8000, "overlap": 400},
    {"size": 2000, "overlap": 200},
    {"size": 4000, "overlap": 400},
]

# Experiment set with the 'ingest' key to trigger ingestion
chunking_experiments: List[Dict[str, Any]] = [
    {
        "prefix": f"Chunk_{cfg['size']}_{cfg['overlap']}",
        "config": {
            "collection_name": f"sec_reports_cs{cfg['size']}_co{cfg['overlap']}",
            "retriever_k": 5,
            "use_critic": False,
        },
        "ingest_params": cfg,
    }
    for cfg in chunking_configs
]

EXPERIMENT_SUITES = {
    "chunking": chunking_experiments,
    "retrieval": retrieval_experiments,
    "generation": generation_experiments,
    "prompt": prompt_experiments,
    "planner": planner_experiments,
    "critic": critic_experiments,
    "all": chunking_experiments
    + retrieval_experiments
    + generation_experiments
    + prompt_experiments
    + planner_experiments
    + critic_experiments,
}


async def run_unified_sweep(suite_name: str) -> None:
    dataset = "RAG_Gold_Benchmark_v2"

    if suite_name not in EXPERIMENT_SUITES:
        print(f"❌ Error: Suite '{suite_name}' not found.")
        print(f"Available suites: {', '.join(EXPERIMENT_SUITES.keys())}")
        sys.exit(1)

    active_experiments = EXPERIMENT_SUITES[suite_name]

    print(f"🚀 Initializing Unified Sweep for {len(active_experiments)} experiments...")

    for exp in active_experiments:
        print(f"\n{'=' * 60}\n🚀 STARTING: {exp['prefix']}\n{'=' * 60}")

        # ingestion check
        if "ingest_params" in exp:
            params = exp["ingest_params"]
            coll_name = exp["config"]["collection_name"]

            print(f"🏗️  PRE-STEP: Ingesting data into {coll_name}...")
            try:
                # We use 'uv run' to ensure we stay in the same GCP environment
                subprocess.run(
                    [
                        "uv",
                        "run",
                        "python",
                        "-m",
                        "src.data.ingest",
                        "--chunk-size",
                        str(params["size"]),
                        "--chunk-overlap",
                        str(params["overlap"]),
                        "--collection",
                        coll_name,
                    ],
                    check=True,
                )
                print("✅ Ingestion successful.")
            except subprocess.CalledProcessError as e:
                print(
                    (
                        f"❌ Ingestion failed for {coll_name}, "
                        f" skipping experiment. Error: {e}"
                    )
                )
                continue

        # RUN EVALUATION
        try:
            await run_ab_experiment(
                dataset_name=dataset,
                experiment_prefix=str(exp["prefix"]),
                config_overrides=exp["config"],
            )
        except Exception as e:
            print(f"❌ Evaluation Error in {exp['prefix']}: {e}")
            continue


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run RAG Evaluation Sweeps")
    parser.add_argument(
        "suite",
        type=str,
        choices=list(EXPERIMENT_SUITES.keys()),
        help="Which suite of experiments to run",
    )

    args = parser.parse_args()

    # Run the selected suite
    asyncio.run(run_unified_sweep(args.suite))
