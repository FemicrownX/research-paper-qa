import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


def ask_gemini(context_docs, question):
    context = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = f"""You are a research assistant. Answer the question below using only the context provided.
If the answer is not in the context, say: "I could not find an answer in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:"""

    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })

    return response.json()["response"].strip()