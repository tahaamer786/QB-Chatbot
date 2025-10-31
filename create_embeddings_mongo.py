import os
import json
from tqdm import tqdm
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ============ Load .env File ============ #
load_dotenv(dotenv_path="C:/Users/New/PyCharmMiscProject/.env")

# ============ MongoDB Setup ============ #
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError(" MONGO_URI not found in .env file. Please check the path or formatting.")

client = MongoClient(mongo_uri)
db = client["qd_chatbot"]
collection = db["documents"]

# Clear previous data (optional)
collection.delete_many({})
print(" Cleared old data from MongoDB.")

# ============ Load Scraped Data ============ #
json_path = "C:/Users/New/PyCharmMiscProject/qd_full_data.json"
if not os.path.exists(json_path):
    raise FileNotFoundError(" qd_full_data.json not found. Make sure scrape_site.py was run successfully.")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ============ Load Sentence Transformer Model ============ #
print(" Loading local embedding model (this may take a moment)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print(" Model loaded successfully.")

# ============ Helper: Chunking Function ============ #
CHUNK_SIZE = 800  # each chunk ~800 characters
CHUNK_OVERLAP = 100  # small overlap for context continuity

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split long text into smaller chunks with slight overlap for better embedding coverage"""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ============ Process and Embed All Pages ============ #
texts = []
metadata = []

print("\n Splitting pages into text chunks...")
for item in tqdm(data, desc="Preparing text chunks"):
    text = item.get("content", "")
    url = item.get("url", "")
    if not text.strip():
        continue

    chunks = chunk_text(text)
    for chunk in chunks:
        texts.append(chunk)
        metadata.append({"url": url, "content": chunk})

print(f" Created {len(texts)} total chunks from {len(data)} pages.\n")

# ============ Generate Embeddings in Batches ============ #
print(" Generating embeddings (this may take several minutes)...")
embeddings = model.encode(texts, batch_size=16, show_progress_bar=True)

# ============ Insert into MongoDB ============ #
print(" Saving embeddings to MongoDB...")
docs_to_insert = []
for emb, meta in zip(embeddings, metadata):
    docs_to_insert.append({
        "url": meta["url"],
        "content": meta["content"],
        "embedding": emb.tolist()
    })

# bulk insert for speed
if docs_to_insert:
    collection.insert_many(docs_to_insert)

print(f"\n Done! Inserted {len(docs_to_insert)} embedded chunks into MongoDB.")
print(" Embeddings created and stored successfully!")
