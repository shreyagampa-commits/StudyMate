# StudyMate AI — RAG-Based PDF Question Answering Assistant

StudyMate AI is an interview-ready major project that answers questions from uploaded PDF documents using a custom-trained Transformer encoder, semantic retrieval, vector search, Flask APIs, and optional LLM answer generation.

## What makes this a strong project

- Custom Transformer Encoder built in PyTorch instead of only using a ready-made embedding model.
- Triplet-loss training for semantic retrieval: anchor question, positive context, negative context.
- Larger included training dataset with 220 triplets across AI, NLP/RAG, DBMS, OS, networks, cloud, and software engineering.
- PDF ingestion pipeline: PDF extraction → chunking → embeddings → vector index → retrieval.
- FAISS vector search with NumPy fallback if FAISS is unavailable.
- Flask REST APIs for upload and question answering.
- Evaluation script to show retrieval quality.
- Optional OpenRouter LLM integration for final context-aware answers.

## Project Architecture

```text
PDF Upload
   ↓
Text Extraction using PyMuPDF
   ↓
Chunking with overlap
   ↓
Custom Transformer Encoder creates embeddings
   ↓
FAISS / vector similarity search retrieves top chunks
   ↓
LLM generates answer using retrieved context
   ↓
Answer + source chunks shown to user
```

## Backend Setup

Open CMD from the project folder:

```bat
cd C:\Users\shrey\Desktop\studymate-ai\backend
..\venv\Scripts\activate
pip install -r requirements.txt
```

If `faiss-cpu` gives an installation issue on Windows, remove `faiss-cpu` from `requirements.txt` and run pip install again. The project will still work using NumPy fallback retrieval.

## Train the Custom Encoder

```bat
python train_encoder.py --epochs 18
```

This creates:

```text
backend/artifacts/encoder_model.pth
backend/artifacts/tokenizer.pkl
backend/artifacts/training_history.json
```

For faster testing:

```bat
python train_encoder.py --epochs 3
```

For a stronger run:

```bat
python train_encoder.py --epochs 30 --batch-size 16
```

## Evaluate Retrieval

```bat
python evaluate_encoder.py
```

You should see pairwise retrieval accuracy and sample top matches.

## Run Backend

```bat
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

## API Usage

### Upload PDF

Use frontend or Postman with form-data:

```text
POST http://127.0.0.1:5000/upload
key: pdf
value: your_file.pdf
```

### Ask Question

```text
POST http://127.0.0.1:5000/ask
Content-Type: application/json
```

Body:

```json
{
  "question": "What is semantic search?",
  "top_k": 4
}
```

## Optional LLM Setup

Create a `.env` file inside `backend`:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Without this key, the project still runs and returns retrieved context.

## How to Explain in Interview

**Problem:** Students upload long PDFs and waste time searching for exact answers.

**Solution:** I built a RAG-based study assistant that retrieves the most relevant document chunks and generates context-aware answers.

**Model:** I trained a custom Transformer Encoder using triplet loss. The model learns to bring related question-context pairs closer in embedding space and push unrelated content farther away.

**Retrieval:** After PDF chunking, each chunk is converted into an embedding. At query time, the question is embedded and compared against document embeddings using FAISS/vector similarity search.

**Why RAG:** Instead of asking an LLM directly, the system grounds the answer in uploaded PDF content, reducing hallucination and improving document-specific accuracy.

## Resume Points

- Built a Retrieval-Augmented Generation (RAG) based study assistant for answering questions from PDF documents using semantic search.
- Developed a custom Transformer-based retrieval pipeline with PyTorch, vector search, and Flask APIs for context-aware question answering.
