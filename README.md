# RAG Application with Qwen3-VL and ChromaDB

A simple terminal-based Retrieval-Augmented Generation (RAG) application to test the context window of Ollama's Qwen3-VL:8B model.

## Features

- 📄 Extracts text from PDF files
- 🗄️ Stores embeddings in ChromaDB vector store
- 🔍 Semantic search for relevant context
- 🤖 Queries Qwen3-VL model with retrieved context
- 💬 Interactive terminal interface

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running with qwen3-vl:8b model:
```bash
ollama list  # Should show qwen3-vl:8b
```

## Usage

Simply run the script:
```bash
python main.py
```

The app will:
1. Find PDF files in the current directory
2. Index the PDF into ChromaDB (first run only)
3. Start an interactive session

### Commands

- Type your question to query the PDF
- `stats` - Show collection statistics
- `quit`, `exit`, or `q` - Exit the app

## Example

```
❓ Question: What are the main topics covered in this document?
🔍 Retrieved 5 relevant chunks (2450 chars)
📊 Total prompt length: 2650 characters
🤖 Querying Ollama Qwen3-VL...

💡 Answer:
[AI response based on the PDF content]
```

## Testing Context Window

The app displays:
- Number of chunks retrieved
- Total character count of context
- Full prompt length sent to the model

This helps you understand how much context the model is processing and test its context window limits.
