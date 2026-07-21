import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "chroma_db")
client = chromadb.PersistentClient(path=db_path)

# Clear existing collection to avoid duplicate chunks
try:
    client.delete_collection("regdocs")
except ValueError:
    pass # Collection doesn't exist yet

embed_fn = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection("regdocs", embedding_function=embed_fn)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

DATA_DIR = os.path.join(current_dir, "data")
for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".pdf"):
        continue
    reader = PdfReader(os.path.join(DATA_DIR, fname))
    
    full_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            # Embed page marker directly into text so the LLM can cite it
            full_text += f"\n\n--- PAGE {i+1} ---\n\n" + text
            
    chunks = text_splitter.split_text(full_text)
    
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            metadatas=[{"source": fname, "chunk": i}],
            ids=[f"{fname}-chunk-{i}"],
        )

print(f"Indexed {collection.count()} semantic chunks")
