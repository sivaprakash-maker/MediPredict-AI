
import requests

url = "http://localhost:11434/api/generate"

project_context = """
You are MediPredict AI Assistant.

MediPredict is a medical e-commerce and insurance
intelligence demo built using Python and AI/ML.

Project modules:

1. Order quantity prediction using Random Forest Regression.
2. Insurance claim review classification.
3. Customer segmentation using K-Means clustering.
4. Product recommendation using collaborative filtering.
5. Local LLM integration using Llama 3.2 1B.

Important:
- This project uses synthetic data.
- Never provide medical diagnosis or prescriptions.
- Insurance predictions are only for human review support.
- Explain technical concepts in simple language.
"""

question = input("Ask MediPredict AI: ")

payload = {
    "model": "llama3.2:1b",
    "system": project_context,
    "prompt": question,
    "stream": False
}

response = requests.post(url, json=payload)
response.raise_for_status()

result = response.json()

print("\nMediPredict AI Assistant")
print("=" * 40)
print(result["response"])
