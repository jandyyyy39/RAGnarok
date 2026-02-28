from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "data/chroma_db"

def test_retrieval(query: str):
    print(f"\n--- Testing Query: '{query}' ---")
    
    # Load the same embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Connect to your local ChromaDB
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    
    # Retrieve the top 3 most relevant chunks
    docs = vectorstore.similarity_search(query, k=3)
    
    if not docs:
        print("No results found. Did you build the database?")
        return

    for i, doc in enumerate(docs):
        print(f"\n[Result {i+1}]")
        # Print the metadata to prove the Markdown headers worked!
        print(f"Headers: {doc.metadata}")
        print(f"Content snippet: {doc.page_content[:200]}...\n")

if __name__ == "__main__":
    # Let's test a few classic D&D mechanics
    test_retrieval("How do I grapple someone?")
    test_retrieval("What happens when I drop to 0 hit points?")