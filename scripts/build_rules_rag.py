import os
import requests
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
SRD_FILE = DATA_DIR / "srd_rules.md"
SRD_URL = "https://raw.githubusercontent.com/oznogon/cc-srd5/main/cc-srd5.md"

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

def build_vector_store():
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
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=final_splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print("Success! The Rules Arbiter now has a working brain.")

if __name__ == "__main__":
    download_srd()
    build_vector_store()