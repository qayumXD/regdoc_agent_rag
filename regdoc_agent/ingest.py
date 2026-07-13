import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import os

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "chroma_db")
client = chromadb.PersistentClient(path=db_path)
embed_fn = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection("regdocs", embedding_function=embed_fn)

DATA_DIR = os.path.join(current_dir, "data")
for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".pdf"):
        continue
    reader = PdfReader(os.path.join(DATA_DIR, fname))
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or not text.strip():
            continue
        collection.add(
            documents=[text],
            metadatas=[{"source": fname, "page": i + 1}],
            ids=[f"{fname}-{i}"],
        )
print(f"Indexed {collection.count()} chunks")
