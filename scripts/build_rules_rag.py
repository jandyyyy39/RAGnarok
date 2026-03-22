import os
import requests
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path
import re

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

def download_srd():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading D&D 5e Markdown Rules (CC Version)...")

def inject_header_chain(documents):
    """
    Add hierarchical header chain metadata to each document chunk.

    Creates a 'header_chain' field like: "Combat > Making an Attack > Attack Rolls"
    """
    for doc in documents:
        chain = []
        
        # Build chain in hierarchical order
        for header_level in ["Header 1", "Header 2", "Header 3", "Header 4"]:
            if header_level in doc.metadata and doc.metadata[header_level]:
                chain.append(doc.metadata[header_level])
        
        # Add the chain to metadata
        if chain:
            doc.metadata["header_chain"] = " > ".join(chain)
        
        # Ensure all header levels exist in metadata (even if None)
        for header_level in ["Header 1", "Header 2", "Header 3", "Header 4"]:
            doc.metadata.setdefault(header_level, None)
    
    return documents

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
    h3 = retrieved_chunk.metadata.get("Header 3")
    
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
    
    if h1 and h2 and h3:
        h3_chunk = [
            c for c in all_chunks if c.metadata.get("Header 1") == h1 
            and c.metadata.get("Header 2") == h2 
            and c.metadata.get("Header 3") == h3 
            and c.metadata.get("Header 4") is None
        ]
        if h3_chunk:
            print(f"\n H2 parent: {h2}")
            print(f"   Content: {h3_chunk[0].page_content[:200]}...\n")
            parent_chunks.extend(h3_chunk)

    if not parent_chunks:
        print("No parent header chunks found")
    
    return parent_chunks

def convert_setext_to_atx(md_text: str) -> str:
    """
    Convert Setext-style headers to ATX-style headers so that
    MarkdownHeaderTextSplitter can recognise them.

    Lines underlined with === become # (H1).
    Lines underlined with --- become ## (H2).

    Edge-cases handled:
    - Skips blank "header" lines (pure whitespace before the underline).
    - Requires the underline to be at least 2 characters long (avoids
      false positives on HR-style separators that are only 1 char, though
      those are non-standard anyway).
    - Leaves fenced code blocks untouched by tracking ``` fences.
    """
    lines = md_text.splitlines()
    output = []
    in_code_block = False
    i = 0

    while i < len(lines):
        line = lines[i]

        # Track fenced code blocks – never touch content inside them.
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            output.append(line)
            i += 1
            continue

        if not in_code_block and i + 1 < len(lines):
            next_line = lines[i + 1]
            stripped_next = next_line.strip()

            # Must be a pure underline of = or - with at least 2 chars.
            is_h1_underline = bool(re.fullmatch(r"={2,}", stripped_next))
            is_h2_underline = bool(re.fullmatch(r"-{2,}", stripped_next))

            if (is_h1_underline or is_h2_underline) and line.strip():
                prefix = "#" if is_h1_underline else "##"
                output.append(f"{prefix} {line.strip()}")
                i += 2          # consume both the heading text and the underline
                continue

        output.append(line)
        i += 1

    return "\n".join(output)

def _load_and_chunk_documents() -> list:
    """Shared chunking logic used by both the vector store and BM25 index."""
    with open(SRD_FILE, "r", encoding="utf-8") as f:
        raw_md_text = f.read()

    # Preprocess Setext headers
    md_text = convert_setext_to_atx(raw_md_text)

    # Split by Markdown headers
    headers_to_split_on = [
        ("#",   "Header 1"),
        ("##",  "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = markdown_splitter.split_text(md_text)

    # Secondary split for oversized chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n", 
            "\n", 
            "</table>", 
            "</tr>", 
            " ", 
            ""
        ]
    )
    final_splits = text_splitter.split_documents(md_header_splits)

    # Enrich metadata AFTER all splitting is done
    final_splits = inject_header_chain(final_splits)

    print(f"Created {len(final_splits)} chunks total")
    return final_splits

def build_vector_store():
    print(f"Reading and processing {SRD_FILE}...")
    
    final_splits = _load_and_chunk_documents()
    
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
    print(f"Reading and processing {SRD_FILE}...")
    
    # Use the shared logic!
    final_splits = _load_and_chunk_documents()
    
    print(f"Building BM25 index with {len(final_splits)} chunks...")
    
    corpus = [doc.page_content for doc in final_splits]
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    print("BM25 index built")
    return bm25, final_splits

if __name__ == "__main__":
    # download_srd()
    build_vector_store()
    # bm25, final_splits = build_bm25_index()
    # peek_chunks(final_splits)

    # retrieved = final_splits[5]
    # get_parent_header_chunks(retrieved, final_splits)