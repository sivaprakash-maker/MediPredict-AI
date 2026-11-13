
from pathlib import Path
import requests

from rag_retriever import retrieve_documents

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:1b"

def ask_medipredict(question):

    print("\nSearching project knowledge...")

    results = retrieve_documents(question, top_k=2)

    # Reject unsupported questions
    if not results or results[0]["score"] < 0.08:
        return "I couldn't find reliable information about this in the MediPredict AI project documents."

    context = "\n\n".join(
        result["content"] for result in results
    )

    prompt = f"""
You are MediPredict AI, a project information assistant.

Answer the user's question using ONLY the project context below.

Rules:
1. Do not invent project features or results.
2. If the context does not contain the answer, say so.
3. Keep the answer short and clear.
4. This project uses synthetic data.
5. Do not provide medical diagnosis or treatment advice.

PROJECT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "2m",
        "options": {
            "num_ctx": 2048,
            "num_predict": 60,
            "temperature": 0.1
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=(10, 600)
        )

        response.raise_for_status()

        answer = response.json().get("response", "").strip()

        return answer if answer else "No answer generated."

    except requests.exceptions.RequestException as error:
        return f"Ollama connection error: {error}"


if __name__ == "__main__":

    print("=" * 50)
    print("MEDIPREDICT AI - RAG ASSISTANT")
    print("=" * 50)

    while True:

        question = input("\nAsk MediPredict AI (type exit to quit): ")

        if question.lower().strip() == "exit":
            print("Thank you for using MediPredict AI!")
            break

        answer = ask_medipredict(question)

        print("\nMediPredict AI Answer:")
        print("-" * 40)
        print(answer)
