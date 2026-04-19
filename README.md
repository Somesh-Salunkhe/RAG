# RAG — Retrieval-Augmented Generation Pipeline

A modular Retrieval-Augmented Generation (RAG) pipeline built with LangChain, FAISS, and Groq LLM. 
Designed to load documents (PDFs), embed them into a vector store, and answer natural language 
queries with context-aware summarization.


<img width="1325" height="802" alt="Screenshot 2026-04-19 164022" src="https://github.com/user-attachments/assets/9d13b74f-cba0-41b1-bdff-faed4163227f" />


---

## 🗂️ Project Structure
RAG/
├── src/
│ ├── data_loader.py # Document loading and preprocessing
│ ├── vectorstore.py # FAISS vector store creation and loading
│ └── search.py # RAG search and summarization logic
├── notebook/ # Jupyter notebooks for exploration
├── main.py # Entry point for running the RAG pipeline
├── requirements.txt # Python dependencies
├── RAG.svg # Architecture diagram
└── .gitignore


## ⚙️ How It Works

1. **Document Loading** — PDFs are loaded and parsed using `PyPDF` / `PyMuPDF`.
2. **Text Splitting** — Documents are chunked using LangChain's text splitters.
3. **Embedding & Indexing** — Chunks are embedded using `sentence-transformers` and indexed with FAISS.
4. **Retrieval & Generation** — On query, the top-k relevant chunks are retrieved and passed to a Groq-hosted LLM for summarization.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Somesh-Salunkhe/RAG.git
cd RAG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare your data

Place your PDF documents in a `data/` directory at the root of the project.

### 4. Build the vector store

Uncomment the vectorstore build lines in `main.py`:

```python
documents = load_documents('data')
store.vectorstore(documents=documents)
```

Then run:

```bash
python main.py
```

### 5. Query the pipeline

Update the `query` variable in `main.py` with your question:

```python
query = "Your question here"
```

Run again to get a context-aware summary:

```bash
python main.py
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langchain`, `langchain-core`, `langchain-community` | Core RAG framework |
| `langchain-text-splitters` | Document chunking |
| `langchain-groq` | Groq LLM integration |
| `pypdf`, `pymupdf` | PDF loading |
| `sentence-transformers` | Text embeddings |
| `faiss-cpu` | Vector similarity search |
| `chromadb` | Alternative vector store |
| `ipykernel` | Jupyter notebook support |

Install all with:

```bash
pip install -r requirements.txt
```

---

## 🔍 Example Query

```python
query = "Who is the author of Weak annotation of Human Activity Recognition datasets using Vision Language models?"
Summary = rag_search.search_and_summarize(query, top_k=3)
print('Summary:', Summary)
```

---

## 🧩 Architecture

See [`RAG.svg`](./RAG.svg) for a visual overview of the pipeline architecture.

---

## 📄 License

This project is open-source. Feel free to use and adapt it for your research and applications.
