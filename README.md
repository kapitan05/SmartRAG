# 📊 SEC Insight Agent: Enterprise-Grade Financial RAG System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red.svg)](https://qdrant.tech/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-green.svg)](https://langchain.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2.svg)](https://mlflow.org/)

## 📝 Overview
**SEC Insight Agent** is a production-ready, agentic Retrieval-Augmented Generation (RAG) system engineered to autonomously analyze and extract insights from dense SEC 10-K financial reports. 

Built with strict MLOps best practices, this system moves beyond naive vector search by implementing **Hybrid Search (Dense + Sparse BM25)**, **Multi-hop reasoning** via LangGraph, and a comprehensive **CI/CD Evaluation Pipeline** to guarantee highly accurate, hallucination-free financial data extraction.

## 💼 Business Value
Extracting actionable insights from 150+ page SEC filings (e.g., MSFT, NVDA) is highly manual and error-prone. This system automates the workflow while maintaining enterprise standards of accuracy:
* **High-Fidelity Retrieval:** Standard semantic search fails at exact-keyword matching (e.g., "Form S-1", specific tickers). Implementing Reciprocal Rank Fusion (RRF) ensures high recall for both conceptual queries and exact financial figures.
* **Autonomous Multi-Hop Reasoning:** The LangGraph agent dynamically executes multiple tool calls per query, allowing it to natively cross-reference and compare multiple companies in a single pass.
* **Quantifiable Reliability:** Continuous A/B testing and LLM-as-a-Judge evaluations prove system reliability before deployment.

---

## 🏗️ System Architecture

### 1. Ingestion & Indexing Pipeline
* **Document Parsing:** Utilizes the Docling API to parse complex financial PDFs into structured Markdown.
* **Contextual Enrichment:** Condenses tables and dynamically injects hierarchical metadata (Ticker, Year, Quarter, Section Path) into every chunk to eliminate context fragmentation.
* **Hybrid Vector Store:** Upserts both Dense embeddings (`text-embedding-3-small`) and Sparse vectors (`Qdrant/bm25` via FastEmbed) into a local Qdrant container.

### 2. Agentic Workflow
* **Orchestration:** Powered by `LangGraph` for stateful, cyclic agent execution.
* **Dynamic Tooling:** Custom tools autonomously adjust retrieval parameters (e.g., switching search algorithms) based on real-time configuration contexts.

### 3. MLOps & Evaluation Pipeline
* **Experiment Tracking:** `MLflow` logs hyperparameter sweeps (A/B testing search algorithms, `chunk_size`, `top_k`).
* **Telemetry:** `LangSmith` traces multi-stage agent trajectories and tool invocations.

---

## 📊 Production & Business Metrics
Evaluating autonomous agents requires moving beyond basic chunk-level matching. This system implements a robust suite of custom and LLM-assisted metrics to measure true business utility.

### 1. Retrieval Metrics (Custom Document-Level Eval)
*Standard `Precision@K` fails for agents that make multiple tool calls. These custom metrics evaluate the holistic retrieval session.*
* **Document Recall:** *Did the agent find the correct source file?* Measures the percentage of expected golden documents successfully retrieved by the agent across all tool calls. 
* **Document Precision:** *How much noise did the agent pull?* Measures the percentage of unique retrieved documents that were actually relevant to the query.

### 2. Generation Metrics (LLM-as-a-Judge via DeepEval)
* **Faithfulness:** Measures hallucination rates. Ensures every claim made by the LLM is directly backed by the retrieved SEC context.
* **Answer Relevancy:** Ensures the final output directly answers the user's prompt without unnecessary verbosity.
* **Contextual Recall:** Evaluates if the retrieved context was sufficient for the LLM to formulate a complete answer.

### 🏆 Benchmark Results (Example)
Transitioning from Dense-only search to **Hybrid Search (RRF)** yielded significant improvements on the internal Gold Benchmark dataset:

| Search Strategy | Document Recall | Document Precision | Faithfulness | Answer Relevancy |
|-----------------|-----------------|--------------------|--------------|------------------|
| Dense Only      | 0.65            | 0.40               | 0.88         | 0.82             |
| BM25 Only       | 0.58            | 0.45               | 0.85         | 0.78             |
| **Hybrid (RRF)**| **0.94** | **0.88** | **0.99** | **0.96** |

---

## 🛠️ Tech Stack
* **AI/LLM Framework:** LangChain, LangGraph, OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)
* **Vector Database:** Qdrant, FastEmbed (Local BM25)
* **MLOps & Eval:** MLflow, DeepEval, LangSmith
* **Infrastructure:** Docker, Docker Compose, `uv` (Python package manager)

---

## 🚀 Quick Start

### Prerequisites
* Docker & Docker Compose
* Python 3.10+
* OpenAI & LangChain API Keys

### Installation & Execution

1. **Clone the repository & Set Environment:**
   ```bash
   git clone [https://github.com/yourusername/SEC-Insight-Agent.git](https://github.com/yourusername/SEC-Insight-Agent.git)
   cd SEC-Insight-Agent
   
   # Create .env file
   echo "OPENAI_API_KEY=sk-your-key" >> .env
   echo "LANGCHAIN_API_KEY=ls-your-key" >> .env
   echo "LANGCHAIN_TRACING_V2=true" >> .env
   echo "LANGCHAIN_PROJECT=SEC_RAG_Eval" >> .env
   echo "QDRANT_HOST=localhost" >> .env
   echo "QDRANT_PORT=6333" >> .env

write it in markdown format 
"2. **Start Infrastructure:**
   ```bash
   docker compose up -d --build
Run Ingestion (Populate Qdrant):

Bash

docker compose exec api uv run python -m src.data.ingest
Run MLOps Evaluation Sweeps:

Bash

docker compose exec api uv run python -m src.eval.run_sweep
🔮 Future Roadmap
[ ] Two-Stage Retrieval (Cross-Encoder): Fine-tune a lightweight open-source model (e.g., Llama-3-8B or BGE-M3) using LoRA/PEFT to rerank the broad candidate pool retrieved by Qdrant.
[ ] Data Interpreter Tool: Integrate a Python REPL tool allowing the agent to perform complex tabular data analysis (e.g., YoY growth calculations) natively via Pandas.
[ ] Self-Hosted Inference (vLLM): Transition away from proprietary API calls by hosting a quantized small language model locally or on a rented GPU instance using vLLM for high-throughput, cost-effective inference.
[ ] RL Post-Training for Financial Reasoning: Align and fine-tune the self-hosted model specifically for financial data extraction and logical reasoning using state-of-the-art Reinforcement Learning techniques such as DPO (Direct Preference Optimization) or GRPO (Group Relative Policy Optimization).
[ ] Cloud-Native Deployment: Migrate infrastructure to AWS ECS / RunPod Serverless for scalable API hosting.
Built by [Your Name] — Always open to discussing AI Engineering, MLOps, and Agentic workflows."