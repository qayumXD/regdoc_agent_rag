# Regulatory Document RAG Agent with Sub-Agents

A cost-efficient, sequential multi-agent RAG (Retrieval-Augmented Generation) system built using **Google ADK** (Agent Development Kit), a local **ChromaDB** vector database, and **GitHub Models** (`gpt-4o` and `gpt-4o-mini` via LiteLLM routing).

## 🚀 Features
- **GitHub Models LLM Backend**: Routes queries through GitHub Models (`gpt-4o-mini` for fast retrieval tool calls, `gpt-4o` for deep synthesis).
- **Sequential Multi-Agent Pipeline**:
  - `retriever_agent`: Strictly retrieves matches from ChromaDB without summarizing (preventing early hallucinations).
  - `synthesizer_agent`: Synthesizes a grounded final answer citing the exact source document and page number.
- **Semantic Text Splitting**: Uses `langchain` recursive text splitters for context-aware chunking while embedding `--- PAGE X ---` markers to preserve legal citation metadata.
- **Empirical RAG Benchmark Suite (`evaluate_rag.py`)**: Includes a built-in evaluator comparing Naive RAG vs Multi-Agent RAG on citation accuracy and groundedness.

## 📊 Benchmark Results (Naive RAG vs Multi-Agent RAG)

| Metric | Naive RAG (Single-Pass) | Multi-Agent RAG (GitHub Models) |
| :--- | :---: | :---: |
| **Citation Precision (%)** | **0.0%** | **100.0%** |
| **Groundedness / Anti-Hallucination** | 100.0% | 80.0% |
| **Average Query Latency** | 3.41s | 5.86s |

## 📁 Project Structure
- `regdoc_agent/agent.py`: Agent logic containing both sub-agents and the sequential orchestrator.
- `regdoc_agent/ingest.py`: Parses PDF files inside the `data/` directory and indexes them into ChromaDB.
- `regdoc_agent/create_dummy_pdf.py`: Script to generate dummy PDF data for clinical evaluation of Software as a Medical Device (SaMD).
- `evaluate_rag.py`: Automated benchmark evaluation suite.

## 🛠️ Setup & Running
1. Clone the repository and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Create a `.env` file inside `regdoc_agent/` with your GitHub PAT:
   ```env
   GOOGLE_GENAI_USE_VERTEXAI=FALSE
   OPENAI_API_KEY=your-github-personal-access-token
   OPENAI_API_BASE=https://models.inference.ai.azure.com
   ```
3. Generate dummy data and ingest documents:
   ```bash
   cd regdoc_agent
   python create_dummy_pdf.py
   python ingest.py
   cd ..
   ```
4. Run the agent via CLI:
   ```bash
   adk run regdoc_agent "What does Section 2 say about clinical evidence?"
   ```
5. Run the Empirical Benchmark Suite:
   ```bash
   python evaluate_rag.py
   ```
