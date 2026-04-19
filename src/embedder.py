from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_documents

class EmbeddingPipeline:

    """ This class handels the operations of chunking and embedding the chunks into vectors"""

    # Constructor to initialize model, chunk size and chunk overlap
    def __init__(self, model_name: str = 'all-MiniLm-L6-v2', chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f'[INFO] Embedding model loaded: {model_name}')

    # Function to perform chunking operation
    def chunk_documents(self, documents:List[Any]) -> List[Any]: #Takes in the documents in the form of list and outputs a list of chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
    
    # Function to embed the chunks into vectors
    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:  # Takes in chunks in the form of a list and outputs embeddings in the form of numpy array of n dimensions
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks ...")

        embeddings = self.model.encode(texts, show_progress_bar = True)
        print(f"[INFO] Embedding shape : {embeddings.shape}")
        return embeddings