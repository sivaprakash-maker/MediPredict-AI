
from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_FILE = (
    BASE_DIR / "data" / "rag" / "medipredict_knowledge.txt"
)


def load_documents():

    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge file not found: {KNOWLEDGE_FILE}"
        )

    text = KNOWLEDGE_FILE.read_text(encoding="utf-8")

    # Split knowledge into meaningful sections
    sections = re.split(
        r"(?=^(?:OVERVIEW:|MODULE \d+:|TECHNOLOGY STACK:|PROJECT LIMITATIONS:))",
        text,
        flags=re.MULTILINE
    )

    documents = [
        section.strip()
        for section in sections
        if section.strip()
    ]

    return documents


def retrieve_documents(question, top_k=3):

    documents = load_documents()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    document_vectors = vectorizer.fit_transform(documents)

    question_vector = vectorizer.transform([question])

    similarity_scores = cosine_similarity(
        question_vector,
        document_vectors
    ).flatten()

    ranked_indices = similarity_scores.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        score = similarity_scores[index]

        if score > 0:
            results.append({
                "content": documents[index],
                "score": round(float(score), 4)
            })

    return results


if __name__ == "__main__":

    print("\nMEDIPREDICT AI - RAG RETRIEVER")
    print("=" * 40)

    question = input("\nEnter your question: ")

    results = retrieve_documents(question)

    if not results:
        print("\nNo relevant information found.")
    else:
        for number, result in enumerate(results, start=1):

            print(f"\nResult {number}")
            print("-" * 40)
            print("Similarity:", result["score"])
            print(result["content"])
