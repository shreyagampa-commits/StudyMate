# StudyMate AI Interview Notes

## One-line explanation
StudyMate AI is a RAG-based PDF question-answering system that uses a custom Transformer Encoder for semantic retrieval and an LLM for grounded answer generation.

## Core modules
1. PDF extraction using PyMuPDF.
2. Chunking with overlap to preserve context.
3. Custom Transformer Encoder trained using triplet loss.
4. Embedding generation for questions and document chunks.
5. Vector retrieval using FAISS or NumPy fallback.
6. Flask APIs for upload and ask operations.
7. Optional OpenRouter LLM for final answer generation.

## Why triplet loss?
Triplet loss uses an anchor, positive, and negative sample. It trains the model so that the anchor question is closer to the relevant answer/context than to irrelevant context.

## Why not only use an LLM?
The LLM may hallucinate if it does not know the document content. RAG first retrieves relevant PDF chunks, so the answer is grounded in the uploaded document.

## Improvements possible
- Add reranking using a cross-encoder.
- Add persistent FAISS indexes per user/document.
- Add authentication and document history.
- Fine-tune on real educational Q&A datasets.
- Add citation highlighting inside the frontend.
