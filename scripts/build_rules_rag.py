<<<<<<< HEAD
import os
import requests
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import Config
import torch

# BM25 specific
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = str(DATA_DIR / "chroma_db")
SRD_FILE = DATA_DIR / "srd_rules.md"
SRD_URL = "https://raw.githubusercontent.com/oznogon/cc-srd5/main/cc-srd5.md" # doesn't work

def download_srd():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading D&D 5e Markdown Rules (CC Version)...")
    try:
        response = requests.get(SRD_URL, timeout=10)
        response.raise_for_status() # This stops the script if it gets a 404
        with open(SRD_FILE, "w", encoding="utf-8") as f:
            f.write(response.text)
        print("Download complete!")
    except Exception as e:
        print(f"FATAL ERROR: Could not download SRD. {e}")
        exit(1)

def inject_header_chain(documents):
    """
    Add hierarchical header chain metadata to each document chunk.
    
    Creates a 'header_chain' field like: "Combat > Making an Attack > Attack Rolls"
    This helps with context-aware retrieval and display.
    """
    for doc in documents:
        chain = []
        
        # Build chain in hierarchical order
        for header_level in ["Header 1", "Header 2", "Header 3"]:
            if header_level in doc.metadata and doc.metadata[header_level]:
                chain.append(doc.metadata[header_level])
        
        # Add the chain to metadata
        if chain:
            doc.metadata["header_chain"] = " > ".join(chain)
        
        # Ensure all header levels exist in metadata (even if None)
        for header_level in ["Header 1", "Header 2", "Header 3"]:
            doc.metadata.setdefault(header_level, None)
    
    return documents

def build_vector_store():
    print(f"Reading {SRD_FILE}...")
    with open(SRD_FILE, "r", encoding="utf-8") as f:
        raw_md_text = f.read()
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    
    print("Splitting by header hierarchy...")
    md_header_splits = markdown_splitter.split_text(raw_md_text)
    
    md_header_splits = inject_header_chain(md_header_splits)
    
    final_splits = md_header_splits

    print(f"Created {len(final_splits)} chunks total")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": device}
    )
    
    print(f"Building Chroma vector store at {CHROMA_DIR}...")
    vectorstore = Chroma.from_documents(
        documents=final_splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    print("Vector store built successfully!")
    return vectorstore

def build_bm25_index():
    print(f"Reading {SRD_FILE}...")
    with open(SRD_FILE, "r", encoding="utf-8") as f:
        raw_md_text = f.read()
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    
    print("Splitting by header hierarchy...")
    md_header_splits = markdown_splitter.split_text(raw_md_text)
    
    md_header_splits = inject_header_chain(md_header_splits)

    final_splits = md_header_splits
    
    print(f"Building BM25 index with {len(final_splits)} chunks...")
    
    corpus = [doc.page_content for doc in final_splits]
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    print("BM25 index built")
    return bm25, final_splits

def peek_chunks(chunks, n=20):
    """
    Debug method to peek chunks
    n = number of chunks
    """
    for i, chunk in enumerate(chunks[:n]):
        print(f"\nChunk {i}")
        print(f"Chain: {chunk.metadata.get('header_chain', 'N/A')}")
        print(f"Content: {chunk.page_content[:150]}...")
        print()

def get_parent_header_chunks(retrieved_chunk, all_chunks):
    h1 = retrieved_chunk.metadata.get("Header 1")
    h2 = retrieved_chunk.metadata.get("Header 2")
    
    parent_chunks = []
    
    # Find H1 parent chunk (has H1 but no H2)
    if h1:
        h1_chunk = [
            c for c in all_chunks
            if c.metadata.get("Header 1") == h1
            and c.metadata.get("Header 2") is None  # No H2 = it's the H1 intro
        ]
        if h1_chunk:
            print(f"\n H1 parent: {h1}")
            print(f"   Content: {h1_chunk[0].page_content[:200]}...\n")
            parent_chunks.extend(h1_chunk)
    
    # Find H2 parent chunk (has H1 and H2, but no H3)
    if h1 and h2:
        h2_chunk = [
            c for c in all_chunks
            if c.metadata.get("Header 1") == h1
            and c.metadata.get("Header 2") == h2
            and c.metadata.get("Header 3") is None  # No H3 = it's the H2 intro
        ]
        if h2_chunk:
            print(f"\n H2 parent: {h2}")
            print(f"   Content: {h2_chunk[0].page_content[:200]}...\n")
            parent_chunks.extend(h2_chunk)
    
    if not parent_chunks:
        print("No parent header chunks found")
    
    return parent_chunks

if __name__ == "__main__":
    # download_srd()
    build_vector_store()
    # bm25, final_splits = build_bm25_index()
    # peek_chunks(final_splits)

    # retrieved = final_splits[5]
    # get_parent_header_chunks(retrieved, final_splits)
    
=======
import os
import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
SRD_FILE = DATA_DIR / "5esrd.md"

sys.path.append(str(BASE_DIR))
from config import Config

def build_vector_store():
    # Check if the file actually exists locally
    if not os.path.exists(SRD_FILE):
        print(f"FATAL ERROR: {SRD_FILE} not found!")
        print("Please download a D&D 5e SRD Markdown file manually and place it in the 'data' folder.")
        exit(1)

    # Split by Markdown headers
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    with open(SRD_FILE, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    print("Chunking rules logically...")
    md_header_splits = markdown_splitter.split_text(md_text)
    
    # Secondary split for very long sections
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    final_splits = text_splitter.split_documents(md_header_splits)

    print(f"Building local database with {len(final_splits)} chunks...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={'device': device}
    )
    
    # Save to ChromaDB
    vectorstore = Chroma.from_documents(
        documents=final_splits,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR) # Cast to string to prevent Pathlib errors
    )
    print("Success! The Rules Arbiter now has a working brain.")

if __name__ == "__main__":
    # Bypassing the volatile internet download step entirely
    build_vector_store()
>>>>>>> 13719273790fe9b374b2d6522abe615c9974edf2
