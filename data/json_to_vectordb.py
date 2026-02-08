import json
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv
import os
import torch

# -------------------------
# Check GPU
# -------------------------
print("CUDA disponible :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX = os.getenv("INDEX_v2")
BATCH_SIZE = 64  

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY manquant")

# -------------------------
# Load JSON data
# -------------------------
with open("reviews_processed2.json", "r", encoding="utf-8") as f:
    reviews = json.load(f)

# -------------------------
# Init Pinecone
# -------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX)

# -------------------------
# Init E5 model
# -------------------------
print("🔄 Chargement du modèle E5-large-v2...")
model = SentenceTransformer("intfloat/e5-large-v2", device="cuda")

# -------------------------
# Prepare vectors
# -------------------------
vectors = []

for idx, review in enumerate(tqdm(reviews, desc="Préparation des embeddings")):
    text = review.get("review", "")
    
    # Embedding
    embedding = model.encode(text, normalize_embeddings=True)
    
    # Pinecone vector
    vectors.append({
        "id": str(idx),
        "values": embedding.tolist(),
        "metadata": {
            "review": text,
            "year": review.get("year"),
            "model": review.get("model"),
            "engine": review.get("engine"),
            "rating": review.get("rating")
        }
    })

# -------------------------
# Upsert par batch
# -------------------------
print("🚀 Envoi vers Pinecone...")
for i in tqdm(range(0, len(vectors), BATCH_SIZE), desc="Upserting batches"):
    batch = vectors[i:i+BATCH_SIZE]
    index.upsert(vectors=batch)

print(f"\n✅ Terminé ! {len(reviews)} reviews indexées dans Pinecone.")
