# Imports
from src.data_loader import load_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGsearch



# Example usage

if __name__ == "__main__":
   
    #documents = load_documents('data')
    store = FaissVectorStore('faiss_store')
    #store.vectorstore(documents=documents)
    store.load()
    rag_search = RAGsearch()
    query = "Who is the author of Weak annotation of Human Activity Recognition datasets using Vision Language models?"
    Summary = rag_search.search_and_summarize(query, top_k=3)
    print('Summary: ', Summary )
