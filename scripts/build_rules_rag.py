import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import requests
import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import Config
from pathlib import Path
import torch

# for RAG
import pickle
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
SRD_FILE = DATA_DIR / "srd_rules.md"
SRD_URL = "https://raw.githubusercontent.com/BTMorton/dnd-5e-srd/master/5esrd.md"

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


    # For Advanced RAG 
    # Save raw chunks for BM25 and evaluation
    chunk_texts = [doc.page_content for doc in final_splits]
    chunk_metadata = [doc.metadata for doc in final_splits]

    with open(DATA_DIR / 'chunks_raw.pkl', 'wb') as f:
        pickle.dump({'texts': chunk_texts, 'metadata': chunk_metadata}, f)
    print(f"Saved {len(chunk_texts)} raw chunks to chunks_raw.pkl")

    # Build BM25 keyword index
    tokenized_corpus = [text.lower().split() for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(DATA_DIR / 'bm25_index.pkl', 'wb') as f:
        pickle.dump(bm25, f)
    print("BM25 index built and saved.")


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
        persist_directory=str(CHROMA_DIR)
        persist_directory=str(CHROMA_DIR) # Cast to string to prevent Pathlib errors
    )
    print("Success! The Rules Arbiter now has a working brain.")

if __name__ == "__main__":
    # Bypassing the volatile internet download step entirely
    build_vector_store()