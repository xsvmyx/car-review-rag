import os
import json
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








# -------------------------
# Groq key management
# -------------------------
def switch_groq_key():
    """Passe à la clé Groq suivante"""
    global current_key_index, client
    current_key_index = (current_key_index + 1) % len(GROQ_KEYS)
    client = Groq(api_key=GROQ_KEYS[current_key_index])
    print(f"🔄 Changement vers clé Groq #{current_key_index + 1}")


def call_groq_with_retry(messages, model, temperature, max_tokens):
    """Groq call with automatic key switch"""
    attempts = 0
    
    while attempts < len(GROQ_KEYS):
        try:
            completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content.strip()
        
        except Exception as e:
            error_str = str(e)
            
            # Détecter rate limit
            if "rate_limit_exceeded" in error_str or "429" in error_str:
                print(f"⚠️  Rate limit atteint (clé #{current_key_index + 1})")
                attempts += 1
                
                # Si toutes les clés sont épuisées
                if attempts >= len(GROQ_KEYS):
                    raise Exception(f"❌ Toutes les {len(GROQ_KEYS)} clés Groq sont épuisées. Réessayez plus tard.")
                
                # Changer de clé et réessayer
                switch_groq_key()
                continue
            
            # Autre erreur - lever l'exception
            raise e
    
    raise Exception("Failed after many attempts")





# -------------------------
# Extract vehicles from user query with Groq
# -------------------------
def extract_vehicles_from_query(user_query):
    """
    Groq will extract the year and vehicule name in a smart way
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a vehicle extraction expert.\n"
                "Extract ALL vehicles mentioned in the user query.\n"
                "Return ONLY a JSON array of vehicle strings.\n"
                "Format: year + make + model (e.g., '2007 Toyota 4Runner')\n"
                "If no vehicle is mentioned, return an empty array [].\n"
                "Examples:\n"
                "- 'is the 2007 Toyota Solara reliable?' → ['2007 Toyota Solara']\n"
                "- 'compare 2007 4Runner and 2009 Xterra' → ['2007 Toyota 4Runner', '2009 Nissan Xterra']\n"
                "Return ONLY the JSON array, nothing else."
            )
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    response = call_groq_with_retry(
        messages=messages,
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=100
    )
    
    try:
        vehicles = json.loads(response)
        return vehicles if isinstance(vehicles, list) else []
    except:
        return []


# -------------------------
# Prepare the context for Groq response generation
# -------------------------
def build_context(filtered_matches):
    """Construit le contexte depuis les matches"""
    context_entries = []
    for match in filtered_matches:
        md = match.metadata
        entry = (
            f"[SIMILARITY SCORE: {match.score:.3f}]\n"
            f"MODEL: {md.get('model', 'unknown')}\n"
            f"YEAR: {md.get('year', 'unknown')}\n"
            f"ENGINE: {md.get('engine', 'unknown')}\n"
            f"RATING: {md.get('rating', 'unknown')}\n"
            f"REVIEW:\n{md.get('review', '')}"
        )
        context_entries.append(entry)
    return "\n\n---\n\n".join(context_entries)


# -------------------------
# Groq Response generation prompt with the context
# -------------------------
def generate_response(user_query, context, is_comparison):
    """Génère la réponse avec Groq"""
    
    if is_comparison:
        task = (
            "TASK:\n"
            "- Compare the vehicles based on the reviews\n"
            "- Summarize strengths and weaknesses for EACH vehicle\n"
            "- Highlight key differences\n"
            "- Include 1-2 short user quotes per vehicle\n"
            "- Base everything strictly on the reviews provided"
        )
    else:
        task = (
            "TASK:\n"
            "- Summarize strengths\n"
            "- Summarize common problems (ONLY if mentioned)\n"
            "- Include 1–3 short user quotes\n"
            "- Base everything strictly on the reviews provided"
        )
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a car reliability expert summarizing REAL user reviews.\n"
                "You must follow these rules strictly:\n"
                "1. Use ONLY information available in the user reviews provided.\n"
                "2. If a problem is NOT mentioned in the reviews, DO NOT invent it.\n"
                "3. If reviews are overwhelmingly positive, say so clearly.\n"
                "4. If opinions differ, mention the disagreement.\n"
                "5. If you do not have enough information to answer, reply exactly: I don't know about this.\n"
                "6. Do NOT mention sources, datasets, or context explicitly.\n"
                "7. Pay attention to the YEAR and MODEL in each review - use the exact vehicles asked about."
            )
        },
        {
            "role": "user",
            "content": f"REVIEWS:\n{context}\n\nUSER QUESTION:\n{user_query}\n\n{task}"
        }
    ]
    
    return call_groq_with_retry(
        messages=messages,
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=800
    )


# -------------------------
# main RAG function
# -------------------------
def ask_car_reviews_bot_free_text(user_query, top_k=5, min_score=0.65):

    # 1️⃣ Extraction des véhicules (JUSTE pour affichage et détection de comparaison)
    vehicles = extract_vehicles_from_query(user_query)
    if vehicles:
        print(f"📌 About: {', '.join(vehicles)}")
    
    # 2️⃣ Embedding de la query
    query_vector = model.encode(
        ["query: " + user_query],
        normalize_embeddings=True
    )[0].tolist()
    
    # 3️⃣ Recherche Pinecone pure
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    # 4️⃣ Filtrage par score UNIQUEMENT
    filtered_matches = [m for m in results.matches if m.score >= min_score]
    
    if not filtered_matches:
        return "I don't know."
    
    # 5️⃣ Construction du contexte
    context = build_context(filtered_matches)
    
    # 6️⃣ Génération (comparaison si plusieurs véhicules détectés)
    is_comparison = len(vehicles) > 1
    return generate_response(user_query, context, is_comparison)













# -------------------------
# 6️⃣ Boucle interactive
# -------------------------
print("\n" + "="*50)
print("CAR REVIEWS BOT READY")
print("="*50 + "\n")

while True:
    print("Entrez 'exit' ou 'quit' pour quitter.")
    user_input = input("👤 You: ").strip()
    if user_input.lower() in ['exit', 'quit']:
        break

    if not user_input:
        continue

    print("🤖 Thinking...", end="\r", flush=True)
    try:
        response = ask_car_reviews_bot_free_text(user_input)
        print(" " * 50, end="\r")
        print(f"🤖 CarFinder:\n{response}\n")
    except Exception as e:
        print(f"\n❌ Erreur : {e}\n")