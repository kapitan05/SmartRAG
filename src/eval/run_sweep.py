import asyncio
import logging

from src.eval.run_test import run_ab_experiment

logging.basicConfig(level=logging.INFO)


async def run_grid_search() -> None:
    experiments1 = [
        {"prefix": "k3_temp0", "config": {"retriever_k": 3, "temperature": 0.0}},
        {"prefix": "k5_temp0", "config": {"retriever_k": 5, "temperature": 0.0}},
        {"prefix": "k10_temp0", "config": {"retriever_k": 10, "temperature": 0.0}},
        {"prefix": "k5_temp0.5", "config": {"retriever_k": 5, "temperature": 0.5}},
    ]
    experiments2 = [
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

    for exp in experiments2:
        print(f"\n{'=' * 50}\n🚀 STARTING EXPERIMENT: {exp['prefix']}\n{'=' * 50}")
        try:
            await run_ab_experiment(
                dataset_name="RAG_Gold_Benchmark_v1",
                experiment_prefix=str(exp["prefix"]),
                config_overrides=exp["config"],  # type: ignore[arg-type]
            )
        except Exception as e:
            print(f"❌ Error in {exp['prefix']}: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(run_grid_search())
