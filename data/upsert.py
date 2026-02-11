import json
import uuid
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
# Load JSON data (CHANGE ICI)
# -------------------------
JSON_FILES = ["data/json/.json", "data/json/.json",]
print("🔄 Chargement du modèle E5-large-v2...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("intfloat/e5-large-v2", device=device)

for JSON_FILE in JSON_FILES:

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    # -------------------------
    # Init Pinecone
    # -------------------------
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX)

    vectors = []

    for review in tqdm(reviews, desc="Préparation des embeddings"):
        text = review.get("review", "").strip()
        if not text:
            continue

        # E5 expects "passage:" for documents
        text_for_embedding = f"passage: {text}"

        embedding = model.encode(
            text_for_embedding,
            normalize_embeddings=True
        )

        vectors.append({
            "id": str(uuid.uuid4()),  # ✅ ID UNIQUE → PAS D'ÉCRASEMENT
            "values": embedding.tolist(),
            "metadata": {
                "review": text,
                "year": review.get("year"),
                "model": review.get("model"),
                "engine": review.get("engine"),
                "rating": review.get("rating"),
                "source": JSON_FILE
            }
        })

    # -------------------------
    # Upsert by batch
    # -------------------------
    print("🚀 Envoi vers Pinecone...")
    for i in tqdm(range(0, len(vectors), BATCH_SIZE), desc="Upserting batches"):
        index.upsert(vectors=vectors[i:i + BATCH_SIZE])

    print(f"\n✅ Terminé ! {len(vectors)} reviews ajoutées à Pinecone (sans overwrite).")
