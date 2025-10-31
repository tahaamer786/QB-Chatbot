import os
from dotenv import load_dotenv
from pymongo import MongoClient
import numpy as np
from numpy.linalg import norm
from numpy import dot
from sentence_transformers import SentenceTransformer
import requests

# ============ Load ENV ============
load_dotenv(dotenv_path="C:/Users/New/PyCharmMiscProject/.env")
groq_api_key = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")

# ============ MongoDB Setup ============
client = MongoClient(mongo_uri)
db = client["qd_chatbot"]
collection = db["documents"]

# ============ Load Embedding Model ============
print("Loading local embedding model (same as used for DB)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded successfully!\n")

# ============ Helpers ============
def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b) + 1e-10)

def get_embedding_local(text: str):
    return model.encode(text)

def find_top_docs(query, top_k=5):
    """Return top K most relevant chunks instead of just one"""
    query_emb = get_embedding_local(query)
    docs = list(collection.find({}, {"content": 1, "embedding": 1}))
    scored_docs = []

    for d in docs:
        d_emb = np.array(d.get("embedding", [0.0] * 384))
        score = cosine_similarity(query_emb, d_emb)
        scored_docs.append((score, d))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_docs = [d for s, d in scored_docs[:top_k] if s > 0.08]  # lower threshold to capture all relevant data
    return top_docs

# ============ Groq API Response ============
def generate_answer(prompt, context=""):
    if not groq_api_key:
        return "Error: Missing GROQ_API_KEY in .env file."

    prompt_lower = prompt.lower()

    # === Friendly greetings ===
    if any(word in prompt_lower for word in ["hi", "hello", "hey", "salam"]):
        return (
            "Hello. Welcome to Qubit Dynamics Academy. "
            "How can I assist you today? Are you interested in learning more "
            "about our community, job openings, or perhaps our research collaborations in AI?"
        )
    if "how are you" in prompt_lower:
        return "I'm doing great, thank you for asking! How can I assist you today?"
    if "thank" in prompt_lower:
        return "You're very welcome! 😊 Always happy to help."

    # === No relevant content ===
    if not context.strip():
        return "I don't know."

    # === GROQ API Request ===
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}

    system_prompt = (
        "You are an intelligent assistant for Qubit Dynamics Academy. "
        "Use ONLY the provided content below to answer clearly and completely. "
        "If the query mentions a person (e.g., Dr Hammad Armaghan) similar to a name in the content "
        "(e.g., Dr Ammar Armaghan), treat them as the same person and provide details. "
        "If the question cannot be answered using the provided text, say exactly 'I don't know.' "
        "Do NOT mention 'website' or 'context' or 'provided data'. "
        "Be concise, natural, and professional. Include all relevant information available."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Website Content:\n{context}\n\nUser Question: {prompt}"}
    ]

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.25,
        "max_tokens": 1000
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=60)
        if res.status_code != 200:
            print("Groq error:", res.status_code, "-", res.text[:200])
            return "Sorry, I couldn't process your question right now."
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Groq request failed:", e)
        return "Error during response generation."

# ============ Terminal Chat ============
def chat():
    print("\nQubit Dynamics Chatbot (Terminal Mode)")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Take top 3–5 chunks for better context
        top_docs = find_top_docs(user_input, top_k=5)
        context = "\n\n".join([d["content"] for d in top_docs]) if top_docs else ""

        answer = generate_answer(user_input, context)
        print(f"\nBot: {answer}\n")

# ============ Run ============
if __name__ == "__main__":
    chat()
