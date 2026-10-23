
import re
import sys
from pathlib import Path

import pymupdf
import weaviate
import yaml
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_weaviate import WeaviateVectorStore

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "docs" / "sources.yaml"
RAW_DIR = ROOT / "docs" / "raw"
COLLECTION = "ProductDocs"
EMBED_MODEL = "nomic-embed-text"

# Load docs

def base_metadata(entry: dict) -> dict:
    return {
        "doc_id": entry["id"],
        "product": entry.get("product") or "general",
        "brand": entry["brand"],
        "doc_type": entry["doc_type"],
        "source_url": entry["url"],
    }


def load_pdf(path: Path, meta: dict) -> list[Document]:
    with pymupdf.open(path) as pdf:
        pages = [page.get_text().strip() for page in pdf]
    text = "\n".join(p for p in pages if p)
    return [Document(page_content=text, metadata=meta)] if text else []


def load_html(path: Path, meta: dict) -> list[Document]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    root = soup.find("main") or soup.body or soup
    lines = (line.strip() for line in root.get_text("\n").splitlines())
    text = "\n".join(line for line in lines if line)
    return [Document(page_content=text, metadata={**meta, "page": 1})] if text else []


def load_documents() -> list[Document]:
    entries = yaml.safe_load(SOURCES.read_text()) or []
    docs = []
    for entry in entries:
        files = list(RAW_DIR.glob(f"{entry['id']}.*"))
        if not files:
            print(f"missing   {entry['id']} (run download_docs.py first)")
            continue
        path, meta = files[0], base_metadata(entry)
        loaded = load_pdf(path, meta) if path.suffix == ".pdf" else load_html(path, meta)
        print(f"loaded    {path.name}")
        docs.extend(loaded)
    return docs

def split_documents(docs: list[Document]) -> list[Document]:
    return [
        Document(page_content=chunk, metadata=doc.metadata)
        for doc in docs
        for chunk in mixed_chunking(doc.page_content)
    ]

# Chunk docs
 
# Heading titles and work caps
HEADING_PATTERN = r"^[A-Z][A-Z /&'-]{3,}$"
MIN_WORDS = 25
MAX_WORDS = 300 
CHUNK_OVERLAP = 30
 
 
def mixed_chunking(source_text):
    chunks = re.split(f"(?m)(?={HEADING_PATTERN})", source_text)
 
    new_chunks = []
    chunk_buffer = ""
    for chunk in chunks:
        new_buffer = chunk_buffer + chunk
        if len(new_buffer.split()) < MIN_WORDS:
            chunk_buffer = new_buffer
        else:
            new_chunks.append(new_buffer)
            chunk_buffer = ""
    if chunk_buffer.strip():
        new_chunks.append(chunk_buffer)
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_WORDS,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda t: len(t.split()),
        separators=["\n\n", "\n", ". ", " "],
    )
    final = []
    for chunk in new_chunks:
        if len(chunk.split()) > MAX_WORDS:
            final.extend(splitter.split_text(chunk))
        else:
            final.append(chunk)
    return final

# Store chunks

def store_chunks(chunks: list[Document]) -> None:
    client = weaviate.connect_to_local()
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    try:
        if client.collections.exists(COLLECTION):
            client.collections.delete(COLLECTION)

        WeaviateVectorStore.from_documents(
            chunks,
            embeddings,
            client=client,
            index_name=COLLECTION,
            text_key="text",
        )
        count = client.collections.get(COLLECTION).aggregate.over_all(total_count=True).total_count
        print(f"Stored {count} chunks in '{COLLECTION}'")
    finally:
        client.close()
        embeddings._client.close()


def main() -> None:

    docs = load_documents()

    chunks = split_documents(docs=docs)
    if not chunks:
        sys.exit("No documents to embed.")

    store_chunks(chunks=chunks)


if __name__ == "__main__":
    main()