import os
import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
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
    markdown_splitter   = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    with open(srd_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    print("Chunking rules by Markdown headers...")
    md_header_splits = markdown_splitter.split_text(md_text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    final_splits  = text_splitter.split_documents(md_header_splits)
    print(f"  {len(final_splits)} chunks created from {srd_path.name}")

    print(f"Building local database with {len(final_splits)} chunks...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    embeddings = HuggingFaceEmbeddings(
        model_name   = Config.EMBEDDING_MODEL,
        model_kwargs = {'device': device},
    )
    
    # Save to ChromaDB
    vectorstore = Chroma.from_documents(
        documents=final_splits,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR) # Cast to string to prevent Pathlib errors
    )
    print(f"Vector store built at {CHROMA_DIR}")


if __name__ == "__main__":
    # Bypassing the volatile internet download step entirely
    build_vector_store()