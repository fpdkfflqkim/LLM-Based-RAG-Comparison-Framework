# 🔍 LLM-Based RAG Comparison Framework
This project evaluates and compares the retrieval and response quality of four different approaches under the same LLM environment:

- Baseline LLM (no retrieval)
- Vector RAG
- Graph RAG
- Hybrid RAG (Vector + Graph)

The system implements the full pipeline end-to-end, including data preprocessing, database indexing, retrieval, and response generation.

A Streamlit-based interactive UI is provided to visually compare the outputs of each pipeline in real time, enabling qualitative analysis of retrieval strategies and response quality.

## 🗺️ Framework Architecture
![alt text](<LLM-Based RAG Comparison Framework.png>)

## ⚙️ Installation

### Prerequisites
| Tool | Purpose |
|---|---|
| `Python 3.12+` | Runtime environment |
| [`uv`](https://docs.astral.sh/uv/) | Dependency management |
| [`Ollama`](https://ollama.com) | Local LLM serving |
| [`Docker`](https://www.docker.com) | Required for Neo4j (for Graph RAG) |

### Setup
**1. Install Dependencies**
```bash
$ uv sync
```

**2. Pull Ollama Models**

Pull the LLM and embedding models used in this project.  
Model names must match those defined in `config.py`. 

```bash
# LLM  (set MODEL_NAME in src/config.py)
$ ollama pull <llm-model>
 
# Embedding model  (set EMBEDDING_MODEL in src/config.py)
$ ollama pull <embedding-model>
```

## 🧰 Tech Stack

| Category | Stack |
|---|---|
| Language | `Python 3.12` |
| Dependency manager | `uv` |
| LLM | `Ollama` |
| Framework | `LangChain` |
| Vector Database | `ChromaDB` |
| Graph Database | `Neo4j` |
| Containerization | `Docker` |
| UI / Demo | `Streamlit` |

## 🚀 Usage Guide

Both Vector RAG and Graph RAG follow the same three-step pipeline.  

Run each module in order:  
**① Data Preprocessing**  
**② Indexing**  
**③ RAG Q&A**  

### 1️⃣ Vector RAG (`src/vectorrag/`)

#### > Running the Framework

| Module | Description | Run Command |
|---|---|---|
| `data_processing.py` | Data Preprocessing | `$ python -m src.vectorrag.data_processing` |
| `db.py` | Vector DB Indexing | `$ python -m src.vectorrag.db` |
| `query_answer.py` | RAG Q&A | `$ python -m src.vectorrag.query_answer` |


#### > Configuration (`config.py`)

| Parameter | Description | Default |
|---|---|---|
| `data_path` | Path to documents for indexing | `./src/vectorrag/input_doc` |
| `ollama_model` | LLM for response generation | `gemma4:31b-cloud` |
| `embd_model` | Embedding model | `embeddinggemma` |
| `prompt_path` | Path to prompt file | `./src/vectorrag/prompt` |
| `persist_directory` | Vector DB storage path | `./src/vectorrag/db` |
| `collection_name` | DB collection name | `vstore_vs_graph_py` |


### 2️⃣ Graph RAG (`src/graphrag/`)

#### > Docker Compose

Graph RAG requires `Neo4j`.  
Make sure to start the Docker container before running any scripts.

```bash
$ docker compose up -d
```
- Compose file: `docker-compose.yml`  
- Credentials: `neo4j_auth.txt` (format: `admin/password`)  
- Dashboard URL : `http://localhost:7474/` 

#### > Running the Framework

| Module | Description | Run Command |
|---|---|---|
| `connect_neo4j.py` | Neo4j Connection Check | `$ python -m src.graphrag.connect_neo4j` |
| `data_processing.py` | Data Preprocessing | `$ python -m src.graphrag.data_processing` |
| `indexing.py` | Graph DB Indexing | `$ python -m src.graphrag.indexing` |
| `retrieve.py` | Retrieval | `$ python -m src.graphrag.retrieve` |
| `query_answer.py` | RAG Q&A | `$ python -m src.graphrag.query_answer` |

#### > Configuration (`config.py`)

| Parameter | Description | Default |
|---|---|---|
| `data_path` | Path to documents for indexing | `./src/graphrag/input_doc` |
| `prompt_path` | Path to prompt file | `./src/graphrag/prompt` |
| `ollama_model` | LLM for response generation | `gemma4:31b-cloud` |
| `embd_model` | Embedding model | `embeddinggemma` |

## 📁 Project Structure

This project is structured into modular pipelines for Vector RAG and Graph RAG, enabling systematic comparison of retrieval strategies.

```bash
Project Root
├── src/
│   ├── vectorrag/                  # Vector RAG pipeline
│   │   ├── input_doc/                # Raw documents for indexing
│   │   ├── prompt/                   # Prompt template files
│   │   ├── config.py                 # Configuration (paths, models, etc.)
│   │   ├── data_processing.py        # Data preprocessing
│   │   ├── db.py                     # Vector DB indexing 
│   │   ├── prompts.py                # Prompt loading utilities
│   │   └── query_answer.py           # RAG query & response generation
│   └── graphrag/                   # Graph RAG pipeline
│       ├── input_doc/                # Raw documents for indexing
│       ├── prompt/                   # Prompt template files
│       ├── config.py                 # Configuration (paths, models, etc.)
│       ├── connect_neo4j.py          # Neo4j connection check
│       ├── data_processing.py        # Data preprocessing
│       ├── indexing.py               # Graph DB indexing
│       ├── prompts.py                # Prompt loading utilities
│       ├── query_answer.py           # RAG query & response generation
│       └── retrieve.py               # Graph-based retrieval
├── test/
│   └── comparing_methods.py        # Performance comparison across 4 approaches
├── docker-compose.yml              # Neo4j container configuration
├── neo4j_auth.txt                  # Neo4j authentication credentials
├── pyproject.toml                  # Project dependencies definition
├── streamlit_app.py                # Interactive comparison UI
└── uv.lock                         # Dependency lock file
```

## 🔬 Testing & Demo Interface

### 1️⃣ Comparison (`test/`)

Compare the performance of each RAG approach.  
All experiments use the same underlying LLM to ensure a fair comparison.

| Module File  | Description | Run Command |
|---|---|---|
| `comparing_methods.py` | Comparing Inference Result<br>(Plain LLM / Vector RAG / Graph RAG / Hybrid RAG)| `$ python -m test.comparing_methods` |

### 2️⃣ Interactive Demo (Streamlit UI)

A lightweight Streamlit-based interface is provided to interactively inspect and compare results.

```bash
$ streamlit run streamlit_app.py
```