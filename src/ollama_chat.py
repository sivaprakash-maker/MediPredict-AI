
import requests

URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:1b"

system_prompt = """
You are MediPredict AI Assistant.

Answer accurately and briefly using this project information.

1. Regression:
Random Forest predicts order quantity.

2. Insurance:
Random Forest Classification.
0 = Routine
1 = Review Flag

3. Customer Segmentation:
K-Means clustering.
Features:
order_count, total_spend, avg_order_value,
total_quantity, avg_discount, unique_products,
cancelled_orders, returned_orders.

4. Product Recommendation:
Collaborative filtering using cosine similarity.

5. LLM:
Llama 3.2 1B using Ollama.

All datasets are synthetic.

Never invent features.
Never provide medical diagnosis or prescriptions.
Insurance predictions support human review only.
"""

messages = [
    {"role": "system", "content": system_prompt}
]

print("\n========================================")
print(" MEDIPREDICT AI - LIGHTWEIGHT CHATBOT")
print("========================================")
print("Type 'exit' to stop.\n")

while True:

    question = input("You: ").strip()

    if question.lower() in ["exit", "quit", "bye"]:
        print("MediPredict AI: Goodbye!")
        break

    if not question:
        continue

    messages.append({
        "role": "user",
        "content": question
    })

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": "2m",
        "options": {
            "num_ctx": 2048,
            "num_predict": 80,
            "temperature": 0.2
        }
    }

    try:
        response = requests.post(
            URL,
            json=payload,
            timeout=(10, 600)
        )

        response.raise_for_status()

        answer = response.json()["message"]["content"]

        print("\nMediPredict AI:", answer, "\n")

        messages.append({
            "role": "assistant",
            "content": answer
        })

        # Keep only the latest two conversation turns.
        messages = [messages[0]] + messages[-4:]

    except requests.exceptions.RequestException as error:
        print("\nAPI Error:", error)
        messages.pop()

    except (KeyError, ValueError) as error:
        print("\nResponse Error:", error)
        messages.pop()
