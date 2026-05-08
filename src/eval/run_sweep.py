import asyncio
import logging

from src.eval.run_test import run_ab_experiment

logging.basicConfig(level=logging.INFO)


async def run_grid_search() -> None:
    dataset = "RAG_Gold_Benchmark_v2"

    # EXPERIMENT 1: Top-K and Temperature (Generation constraints)
    generation_experiments = [
        {"prefix": "Gen_K3_Temp0", "config": {"retriever_k": 3, "temperature": 0.0}},
        {"prefix": "Gen_K5_Temp0", "config": {"retriever_k": 5, "temperature": 0.0}},
        {"prefix": "Gen_K10_Temp0", "config": {"retriever_k": 10, "temperature": 0.0}},
        {"prefix": "Gen_K5_Temp0.5", "config": {"retriever_k": 5, "temperature": 0.5}},
    ]

    # EXPERIMENT 2: Search Algorithms (Retrieval performance)
    retrieval_experiments = [
        {
            "prefix": "Search_Dense_Only_K5",
            "config": {"search_algorithm": "dense", "retriever_k": 5},
        },
        {
            "prefix": "Search_BM25_Only_K5",
            "config": {"search_algorithm": "bm25", "retriever_k": 5},
        },
        {
            "prefix": "Search_Hybrid_RRF_K5",
            "config": {"search_algorithm": "hybrid", "retriever_k": 5},
        },
    ]

    # EXPERIMENT 3: Prompt Engineering (Agent behavior)
    prompt_experiments = [
        {
            "prefix": "Prompt_v3_Baseline",
            "config": {
                "prompt_version": "v3_baseline",
                "search_algorithm": "hybrid",
                "retriever_k": 5,
            },
        },
        {
            "prefix": "Prompt_v4_Avoid_Duplicate_Searches",
            "config": {
                "prompt_version": "v4_avoid_duplicate_searches",
                "search_algorithm": "hybrid",
                "retriever_k": 5,
            },
        },
    ]

    # EXPERIMENT: Planner vs No Planner
    planner_experiments = [
        {
            "prefix": "Agent_Standard_Single_Shot",
            "config": {
                "prompt_version": "v4_strict_metadata",
                "use_planner": False,
                "retriever_k": 5,
                "search_algorithm": "hybrid",
            },
        },
        {
            "prefix": "Agent_With_Query_Decomposition",
            "config": {
                "prompt_version": "v4_strict_metadata",
                "use_planner": True,
                "retriever_k": 5,
                "search_algorithm": "hybrid",
            },
        },
    ]

    # which experiments to run? control with these flags
    active_experiments = planner_experiments

    print(f"🚀 Initializing Grid Search for {len(active_experiments)} experiments...")

    for exp in active_experiments:
        print(f"\n{'=' * 50}\n🚀 STARTING EXPERIMENT: {exp['prefix']}\n{'=' * 50}")
        try:
            await run_ab_experiment(
                dataset_name=dataset,
                experiment_prefix=str(exp["prefix"]),
                config_overrides=exp["config"],  # type: ignore[arg-type]
            )
        except Exception as e:
            print(f"❌ Error in {exp['prefix']}: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(run_grid_search())
