import os
import requests
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path
from config import Config
import torch

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = Config.DATA_DIR
CHROMA_DIR = Config.CHROMA_DIR
SRD_URL    = "https://raw.githubusercontent.com/oznogon/cc-srd5/main/cc-srd5.md"


def resolve_srd_file() -> Path:
    if Config.ESRD_FILE.exists():
        print(f"Using local SRD: {Config.ESRD_FILE}")
        return Config.ESRD_FILE
    if Config.SRD_FILE.exists():
        print(f"Using cached SRD: {Config.SRD_FILE}")
        return Config.SRD_FILE
    return download_srd()


def download_srd() -> Path:
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading D&D 5e SRD from GitHub...")
    try:
        response = requests.get(SRD_URL, timeout=10)
        response.raise_for_status()
        with open(Config.SRD_FILE, "w", encoding="utf-8") as f:
            f.write(response.text)
        print("Download complete.")
        return Config.SRD_FILE
    except Exception as e:
        print(f"FATAL: Could not download SRD and no local copy found. {e}")
        exit(1)


def build_vector_store(srd_path: Path):
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter   = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    with open(srd_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    print("Chunking rules by Markdown headers...")
    md_header_splits = markdown_splitter.split_text(md_text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    final_splits  = text_splitter.split_documents(md_header_splits)
    print(f"  {len(final_splits)} chunks created from {srd_path.name}")

    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name   = Config.EMBEDDING_MODEL,
        model_kwargs = {'device': device},
    )

    Chroma.from_documents(
        documents         = final_splits,
        embedding         = embeddings,
        persist_directory = str(CHROMA_DIR),
    )
    print(f"Vector store built at {CHROMA_DIR}")


if __name__ == "__main__":
    srd_path = resolve_srd_file()
    build_vector_store(srd_path)
