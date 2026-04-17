# Import libraries

from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader


# Function to load documents in a directory

def load_documents(directory:str) -> List[Any]:         # input data directory strictly in string format and it returns a list of any types of data type 
    ''' 
    Loads all supported files from the directory and converts to Langchain document structure.
    Supported file formats: PDF, Txt, CSV, Excel, Word, JSON
    '''

    # Use project root path
    data_path = Path(directory).resolve()
    print(f"[DEBUG] Data Path: {directory}")
    
    # List to store document files
    documents = []

    # PDF files
    pdf_files = list(data_path.glob("**/*.pdf"))              # Pattern search for all files with .pdf extension
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    
    # Loop through all available files and load them one by one
    for file in pdf_files:
        print(f"[DEBUG] Loading PDF: {file}")

        try:
            loader = PyPDFLoader(str(file))                       # PDF loader
            loaded = loader.load()                                

            print(f"[DEBUG] Loaded {len(loaded)} PDF documents from {pdf_files}")

            documents.extend(loaded)                              # Appending loaded files into list documents
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {file}: {e}")
    
    return documents


        
    # Text Files

    # CSV Files

    # SQL Files