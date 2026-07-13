import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from pdf_processor import extract_text_from_pdf
from chunker import chunk_text
from rag_pipeline import RAGPipeline
from llm_api import generate_answer

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    rag = RAGPipeline()
    startup_error = None
except Exception as e:
    rag = None
    startup_error = str(e)


@app.route("/")
def home():
    return jsonify({"message": "StudyMate AI Backend Running", "model_loaded": rag is not None, "error": startup_error})


@app.route("/upload", methods=["POST"])
def upload_pdf():
    global rag
    if rag is None:
        return jsonify({"error": startup_error, "fix": "Run python train_encoder.py --epochs 18 first."}), 500

    if "pdf" not in request.files:
        return jsonify({"error": "No PDF uploaded. Use form field name 'pdf'."}), 400

    file = request.files["pdf"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    pages = extract_text_from_pdf(file_path)
    chunks = chunk_text(pages)
    rag.add_documents(chunks)

    return jsonify({"message": "PDF processed successfully", "pages": len(pages), "total_chunks": len(chunks)})


@app.route("/ask", methods=["POST"])
def ask_question():
    if rag is None:
        return jsonify({"error": startup_error, "fix": "Run python train_encoder.py --epochs 18 first."}), 500

    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    retrieved = rag.retrieve(question, top_k=int(data.get("top_k", 4)))
    context = "\n\n".join([chunk["text"] for _, chunk in retrieved])
    answer = generate_answer(question, context)

    sources = [
        {"page": chunk.get("page"), "score": round(float(score), 4), "text": chunk["text"][:350]}
        for score, chunk in retrieved
    ]
    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
