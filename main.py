# Imports
from src.data_loader import load_documents
from src.embedder import EmbeddingPipeline



# Example usage

if __name__ == "__main__":
    documents = load_documents('data')
    chunks = EmbeddingPipeline().chunk_documents(documents=documents)
    embeddings = EmbeddingPipeline().embed_chunks(chunks=chunks)
    print(embeddings)