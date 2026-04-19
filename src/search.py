# Import
import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGsearch:
    def __init__(self, persist_dir : str = 'faiss_store', embedding_model : str ='all-MiniLM-L6-v2', llm_model: str = 'llama-3.1-8b-instant'):
        self.vectorstore =  FaissVectorStore(persist_dir=persist_dir, embedding_model=embedding_model)
        
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, 'faiss.index')
        meta_path =  os.path.join(persist_dir, 'metadata.pkl')

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_documents
            docs = load_documents('data')
            self.vectorstore.vectorstore(docs)

        else:
            self.vectorstore.load()
            api = os.getenv('API')
            self.llm = ChatGroq(api_key=api, model=llm_model)
            print(f"[INFO] Groq LLM initialized: {llm_model}")
        

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k)
        texts = [r["metadata"].get("text", "") for r in results if r['metadata']]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found"
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n\n{context}\n\nSummmary: """
        response = self.llm.invoke([prompt])
        return response.content