from google.adk.agents import Agent, SequentialAgent
import chromadb
from chromadb.utils import embedding_functions

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "chroma_db")
_client = chromadb.PersistentClient(path=db_path)
_embed_fn = embedding_functions.DefaultEmbeddingFunction()
_collection = _client.get_or_create_collection("regdocs", embedding_function=_embed_fn)

def search_knowledge_base(query: str) -> dict:
    """Searches indexed regulatory documents for passages relevant to the query.

    Args:
        query: The search query.

    Returns:
        A dict with matched passages and their source document/page.
    """
    results = _collection.query(query_texts=[query], n_results=4)
    matches = [
        {"text": doc, "source": meta["source"], "chunk": meta["chunk"]}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
    return {"matches": matches}

retriever_agent = Agent(
    name="retriever_agent",
    model="openai/gpt-4o-mini",
    instruction=(
        "Given the user's question, call search_knowledge_base with the question as the query. "
        "Return the raw matches exactly as returned — do not summarize or answer yourself."
    ),
    tools=[search_knowledge_base],
    output_key="retrieved_chunks",
)

synthesizer_agent = Agent(
    name="synthesizer_agent",
    model="openai/gpt-4o",
    instruction=(
        "The retrieved passages are in {retrieved_chunks}. Draft a clear, grounded answer to the "
        "user's question, citing the source document and page number for every claim. (Note: "
        "page markers are embedded within the text as '--- PAGE X ---'). If the passages "
        "don't actually answer the question, say so explicitly instead of guessing."
    ),
)

root_agent = SequentialAgent(
    name="regdoc_agent",
    sub_agents=[retriever_agent, synthesizer_agent],
)
