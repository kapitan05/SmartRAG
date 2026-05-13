# Financial Agentic RAG system (FARS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-Fast_Pip-purple.svg)](https://github.com/astral-sh/uv)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![GCP](https://img.shields.io/badge/Google_Cloud-4285F4?style=flat&logo=google-cloud&logoColor=white)](https://cloud.google.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red.svg)](https://qdrant.tech/)
[![Docling](https://img.shields.io/badge/Docling-PDF_Parsing-orange.svg)](https://github.com/DS4SD/docling)
[![OpenAI](https://img.shields.io/badge/OpenAI-Models-412991.svg?logo=openai&logoColor=white)](https://openai.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-gray.svg)](https://langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-green.svg)](https://langchain.com/)
[![vLLM](https://img.shields.io/badge/vLLM-Fast_Inference-blueviolet.svg)](https://docs.vllm.ai/)
[![DeepEval](https://img.shields.io/badge/DeepEval-LLM_Eval-FF4B4B.svg)](https://github.com/confident-ai/deepeval)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-black.svg)](https://smith.langchain.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2.svg)](https://mlflow.org/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Telemetry-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800?logo=grafana&logoColor=white)](https://grafana.com/)
## Overview
**FARS** is a production-ready, agentic Retrieval-Augmented Generation (RAG) system engineered to autonomously analyze and extract insights from financial reports of 10-Q form. FARS is fully hosted on Google Cloud Platform and exposed on external IP.

## System in Action
![SEC Agent UI](docs/assets/ui_screenshot.png)

## Key Capabilities
* **Multi-Hop Reasoning:** Planner agent node can execute multiple searches to compare different companies or fiscal years in a single pass
* **Hybrid Search:** Semantic + BM25
* **Self-Correcting Critic:** Critic agent node evaluates draft answers. If data is missing, it routes the agent to fetch more data before showing the user.
* **Quantifiable Reliability:** LLM-as-a-judge metrics guarantee high faithfulness and low hallucinations.

## System Architecture
```mermaid
graph TD
    %% User Flow
    User((User)) -->|Query| UI[Streamlit UI]
    UI -->|API Call| FA[FastAPI Backend]

    %% LangGraph Flow
    subgraph LangGraph Agentic Workflow
        FA --> Planner[Planner Node]
        Planner -->|Decomposes Query| Tools[Retrieval Tools]
        Tools --> Agent[Agent Node]
        
        Agent --> Critic[Critic Node<br>Self-Reflection]
        
        Critic -->|Approved| Output((Final Response))
        Critic -->|Missing Data| Planner
        Critic -->|Logic/Format Error| Agent
    end

    %% Data Layer
    subgraph Data & Infra
        Tools <-->|Hybrid Search +<br>Metadata Pre-Filtering| Qdrant[(Qdrant DB<br>Dense + BM25)]
        Tools <-->|Parse PDFs| Docling[Docling Engine]
    end

    %% Telemetry Layer
    subgraph Observability
        Agent -.->|Traces| LS(LangSmith)
        Critic -.->|Traces| LS
        FA -.->|Metrics| Prom(Prometheus) --> Graf(Grafana)
    end
```


## Evaluation

### Prompts testing

Using MLflow comparing a standard prompt (`v4_strict_metadata`) against improved one (`agent`) we achieve better results.

| Evaluation Metric | `v4_strict_metadata` | `agent`|
| :--- | :--- | :--- |
| **Answer Relevancy** | 0.941 | **0.985** |
| **Contextual Recall** | 0.739 | **0.833** |
| **Doc Precision** | 0.617 | **0.783** |
| **Doc Recall** | 0.950 | 0.950 |
| **Faithfulness** | **0.857** | 0.804 |
| **Word F1** | 0.357 | **0.379** |

<details>
<summary>View Raw MLflow Experiment Logs</summary>
<br>

<img src="docs/assets/mlflow_prompt1.png" width="800" alt="MLflow Prompt Comparison">
<img src="docs/assets/mlflow_prompt2.png" width="800" alt="MLflow Prompt Comparison">
</details>


Running 20+ experiment runs to find the best configuration. The following was fond:

| Component | Experiments Run | Winning Configuration |
| :--- | :--- | :--- |
| **Search Strategy** | Dense vs. BM25 vs. Hybrid | **Hybrid** |
| **Retrieval Depth** | k=3, 5, 10 | **k=10** |
| **Generation** | Temp 0.0, 0.1, 0.3 | **Temp 0.0** |
| **Query Logic** | Standard vs. Planner | **Planner** |
| **Verification** | Critic On vs. Off | **Critic off** |
| **Data Chunking** | 2k, 4k, 8k | **8000, 800 overlap** |

---



## Tech Stack
* **LLM Framework:** `LangChain`, `LangGraph`, `OpenAI`
* **Vector Database:** `Qdrant`, `FastEmbed`
* **MLOps & Eval:** `MLflow`, `DeepEval`, `LangSmith`, `Prometheus`, `Grafana`
* **Infrastructure:** `Docker`, `Google Cloud Platform`, `FastAPI`, `uv`
* **UI:** `Streamlit`

---

## Quick Start

### Installation & Execution

1. **Clone the repository & Set Environment:**
   ```bash
   git clone [https://github.com/yourusername/SEC-Insight-Agent.git](https://github.com/yourusername/SEC-Insight-Agent.git)
   cd SEC-Insight-Agent
   
   # Create .env file
   echo "OPENAI_API_KEY=sk-your-key" >> .env
   echo "LANGCHAIN_API_KEY=ls-your-key" >> .env
   echo "LANGCHAIN_TRACING_V2=true" >> .env
   echo "LANGCHAIN_PROJECT=your_name" >> .env
   echo "LANGSMITH_ENDPOINT=your_endpoint" >> .env
   echo"ACCESS_TOKEN_EXPIRE_MINUTES=1440" >> .env
   echo "JWT_SECRET_KEY=your_key" >> .env
   echo "JWT_ALGORITHM=HS256" >> .env
   echo "QDRANT_HOST=qdrant" >> .env
   echo "QDRANT_PORT=6333" >> .env
   echo "MONGO_URI=mongodb://mongodb:27017" >> .env

 **Start Infrastructure:**

   ```bash
   docker compose up -d --build
   ```
Run Ingestion (Populate Qdrant):
   ```bash
docker compose exec api uv run python -m src.data.ingest
   ```
Run MLOps Evaluation Sweeps:
   ```bash
docker compose exec api uv run python -m src.eval.run_sweep
   ```


## 🔮 Future Roadmap
* **Two-Stage Retrieval (Cross-Encoder):** Fine-tune a lightweight open-source model (e.g., Llama-3-8B or BGE-M3) using LoRA/PEFT to rerank the broad candidate pool retrieved by Qdrant.
* **Data Interpreter Tool:** Integrate a Python REPL tool allowing the agent to perform complex tabular data analysis (e.g., YoY growth calculations) natively via Pandas.
* **Self-Hosted Inference (vLLM):** Transition away from proprietary API calls by hosting a quantized small language model locally or on a rented GPU instance using vLLM for high-throughput, cost-effective inference.
* **RL Post-Training for Financial Reasoning:** Align and fine-tune the self-hosted model specifically for financial data extraction and logical reasoning using state-of-the-art Reinforcement Learning techniques such as DPO (Direct Preference Optimization) or GRPO (Group Relative Policy Optimization).