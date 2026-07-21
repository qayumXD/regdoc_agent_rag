# Regulatory Document RAG Agent with Sub-Agents

A cost-efficient, sequential multi-agent RAG (Retrieval-Augmented Generation) system built using **Google ADK** (Agent Development Kit), a local **ChromaDB** vector database, and the **Qwen API** (powered by `qwen-max` via LiteLLM).

## Features
- **Local Embedding & Storage**: Uses ChromaDB running locally with default embeddings, keeping indexation entirely free.
- **Cost-efficient LLM Backend**: Queries `qwen-max` via an OpenAI-compatible endpoint using LiteLLM routing.
- **Sequential Multi-Agent Pipeline**:
  - `retriever_agent`: Strictly retrieves matches from ChromaDB without summarizing (preventing early hallucinations).
  - `synthesizer_agent`: Synthesizes a grounded final answer citing the source document and page number.
- **Semantic Text Splitting**: Uses `langchain` recursive text splitters for context-aware chunking, enhancing retrieval accuracy compared to naive page-based splits.

## Project Structure
- `agent.py`: Agent logic containing both sub-agents and the sequential orchestrator.
- `ingest.py`: Parses PDF files inside the `data/` directory and indexes them into ChromaDB.
- `create_dummy_pdf.py`: Script to generate a dummy PDF for clinical evaluation of Software as a Medical Device (SaMD).
- `data/`: Directory for regulatory PDFs.

## Setup & Running
1. Clone the repository and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Create a `.env` file inside `regdoc_agent/` with:
   ```env
   GOOGLE_GENAI_USE_VERTEXAI=FALSE
   OPENAI_API_KEY=your-qwen-api-key
   OPENAI_API_BASE=https://ws-jhfrt7owiw2bvmy4.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
   ```
3. Generate dummy data and ingest documents:
   ```bash
   cd regdoc_agent
   python create_dummy_pdf.py
   python ingest.py
   ```
4. Run the agent via CLI:
   ```bash
   adk run . "What does Section 2 say about clinical evidence?"
   ```
5. Or run the Web UI:
   ```bash
   adk web .
   ```