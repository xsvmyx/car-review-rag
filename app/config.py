import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq



# -------------------------
# Config
# -------------------------
load_dotenv()

print("⏳ System init  (E5 + Pinecone + Groq)...", flush=True)

model = SentenceTransformer("intfloat/e5-large-v2", device="cuda")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("INDEX_v2"))

GROQ_KEYS = [os.getenv(f"GROQ_API_KEY{i}") for i in range(1, 11) if os.getenv(f"GROQ_API_KEY{i}")]
current_key_index = 0
client = Groq(api_key=GROQ_KEYS[current_key_index])

print(f"✅ {len(GROQ_KEYS)} clés Groq disponibles")

