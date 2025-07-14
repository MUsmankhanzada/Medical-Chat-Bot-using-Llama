# Medical-Chat-Bot-using-Llama

A Retrieval-Augmented Generation (RAG) based medical chatbot using Llama models, LangChain, and Pinecone.

## Overview
This project implements a medical chatbot that leverages Retrieval-Augmented Generation (RAG) to provide accurate and context-aware answers to user queries. The system uses a quantized Llama model for efficient inference on resource-constrained hardware, and integrates with Pinecone for vector storage and retrieval.

## Features
- **RAG Pipeline:** Combines LLM generation with retrieval from a local medical PDF knowledge base.
- **Quantized Llama Model:** Uses a lightweight, quantized Llama model (GGUF format) for fast, memory-efficient inference.
- **Pinecone Vector Store:** Stores and retrieves document embeddings for context-aware responses.
- **Web Chat Interface:** Simple Flask-based web UI for user interaction.
- **Model Experimentation:** Tested multiple Llama models; selected a quantized model due to system limitations.

## Architecture
- **Frontend:** HTML/CSS chat interface (`templates/chat.html`, `static/style.css`)
- **Backend:** Flask app (`app.py`) serving the chatbot and handling user queries
- **RAG Pipeline:**
  - PDF ingestion and chunking (`src/helper.py`)
  - Embedding generation using HuggingFace models
  - Vector storage and retrieval with Pinecone
  - Llama model inference via LangChain

## Model Selection
- **Tested:**
  - Full-size Llama models (e.g., Llama-2, Llama-3)
  - Quantized TinyLlama (1.1B) for resource efficiency
- **Final Choice:**
  - Used a quantized, lightweight Llama model due to limited system resources (no GPU, limited RAM)

## Tested Models
The following models were tested during development:

- **TinyLLaMA-1.1B-Chat-v1.0-GGUF** (quantized):
  - Downloaded using Hugging Face CLI:
    ```bash
    huggingface-cli download TheBloke/TinyLLaMA-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir ./models --local-dir-use-symlinks False
    ```
- **Other Llama Models:**
  - Larger Llama models (e.g., Llama-2, Llama-3) were also tested, but due to system constraints, the quantized TinyLLaMA model was selected for deployment.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repo-url>
cd Medical-Chat-Bot-using-Llama
```

### 2. Install Dependencies
Install Python 3.8+ and run:
```bash
pip install -r requirement.txt
```

### 3. Prepare Environment Variables
Create a `.env` file with your Pinecone API key:
```
PINECONE_API_KEY=your_pinecone_api_key
```

### 4. Add Medical Knowledge Base
Place your medical PDF(s) in the `data/` directory (e.g., `data/Medical_book.pdf`).

### 5. Build the Vector Index
```bash
python store_index.py
```

### 6. Run the Chatbot
```bash
python app.py
```
Visit [http://localhost:8080](http://localhost:8080) in your browser.

## File Structure
- `app.py` - Main Flask app and RAG pipeline
- `src/helper.py` - PDF loading, text splitting, embedding
- `src/prompt.py` - Prompt template for Llama
- `store_index.py` - Builds Pinecone vector index
- `templates/chat.html` - Web chat UI
- `static/style.css` - UI styles
- `data/` - Medical PDFs
- `experiment/trials.ipynb` - Model experiments and testing

## Notebooks & Experiments
- `experiment/trials.ipynb` contains experiments with different Llama models and embedding strategies. Due to hardware constraints, a quantized model was selected for deployment.

## Requirements
See `requirement.txt` for all dependencies. Notable packages:
- `flask`, `langchain`, `pinecone-client`, `langchain-community`, `sentence-transformers`, `llama-cpp-python`, `pypdf`, `python-dotenv`, `langchain-pinecone`, `huggingface-hub`, `torch`

## Notes
- The chatbot is for educational/demo purposes and should not be used for real medical advice.
- Quantized models are used to enable local inference on systems without a powerful GPU.

## License
MIT License