# RAG Chatbot Research Assistant

Retrieval-Augmented Generation (RAG) powered chatbot that allows researchers and professionals to interact with large sets of documents using natural language queries. Built using LangChain, OpenAI, ChromaDB, and Gradio, this assistant enables semantic search and contextual Q&A from embedded PDFs.

## Features

- **Semantic Search**: Uses OpenAI embeddings (`text-embedding-3-large`) to map document text into a vector space for relevant retrieval.
- **Natural Language Chat Interface**: Ask questions in plain English and receive source-grounded answers.
- **RAG Workflow**: Combines retrieval with generation for accurate and context-aware responses.
- **Multi-Document Support**: Load and query multiple PDFs from the `data/` directory.
- **Streamed LLM Responses**: Uses GPT-4 with streaming output for fast interaction.
- **Customizable for Research Domains**: Can be adapted to scientific, medical, legal, or business documents.

---

## Tech Stack

| Component | Description |
|----------|-------------|
| OpenAI GPT-4 | Large Language Model for generating contextual responses |
| LangChain | Framework for document ingestion, retrieval, and orchestration |
| OpenAI Embeddings | Transforms text into semantic vectors |
| ChromaDB | Persistent vector store for document chunks |
| Gradio | Interactive UI for chatbot interface |
| PyPDFLoader | Loads academic PDFs from the `/data` folder |

---
