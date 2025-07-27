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
## How It Works

1. **Load PDFs**  
   PDFs from the `data` folder are loaded and parsed.

2. **Text Splitting**  
   Each document is split into ~300-character chunks (with overlap) to preserve context.

3. **Embedding & Storage**  
   Chunks are embedded via OpenAI and stored in ChromaDB for fast retrieval.

4. **RAG Query Flow**  
   - User types a question into Gradio.
   - Top 5 relevant chunks are retrieved via semantic search.
   - Prompt with retrieved content is sent to GPT-4.
   - Answer is streamed to the UI in real time.

---
## User Guide

1. **Load PDFs**
   - Place all documents into the `data` directory.
2. **Set up environment variables**
   - Create a `.env` file with your OpenAI key:
     ```bash
     OPENAI_API_KEY=your-api-key
3. **Run the app**
   ```bash
   python chatbot.py
