import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def generate_answer(question, context):
    if not context.strip():
        return "I could not find relevant content in the uploaded document."

    if not API_KEY:
        return (
            "OpenRouter API key is not configured, so I am returning retrieved context only.\n\n"
            + context[:1200]
        )

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    prompt = f"""
You are StudyMate AI, a helpful academic assistant.
Answer only from the provided context. If the answer is not present, say that it is not available in the document.
Give a clear, concise answer suitable for a student.

Context:
{context}

Question: {question}
Answer:
"""
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM request failed: {e}\n\nRetrieved context:\n{context[:1200]}"
