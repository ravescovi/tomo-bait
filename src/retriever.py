import sys
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
# Make sure these match what you used in Phase 1
DB_PATH = "./chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# ---

def get_documentation_retriever():
    """
    Initializes and returns a retriever for our ChromaDB.
    """
    print(f"Loading embedding model: {MODEL_NAME}")
    # Initialize the same embedding model
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    print(f"Connecting to vector store at: {DB_PATH}")
    # Connect to the existing, persisted database
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    
    print("✅ Retriever is ready.")
    # Create a retriever object that can search for 3 relevant docs
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Test Block ---
if __name__ == "__main__":
    """
    This lets us test the retriever function by running:
    python retriever.py "your test question"
    """
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\n--- Testing Retriever ---")
        print(f"Query: '{query}'")
        
        retriever = get_documentation_retriever()
        
        # 'invoke' runs the retriever and gets the docs
        results = retriever.invoke(query)
        
        print(f"\nFound {len(results)} relevant documents:")
        for i, doc in enumerate(results):
            print(f"\n--- Document {i+1} ---")
            print(doc.page_content)
            print(f"(Source: {doc.metadata.get('source', 'unknown')})")
            print("------------------")
    else:
        print("Usage: python retriever.py \"Your test query here\"")