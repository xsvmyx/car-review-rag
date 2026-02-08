import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq
from app.schemas import QueryRequest, QueryResponse

# -------------------------
# 1. Configuration & Setup
# -------------------------
load_dotenv()

app = FastAPI(
    title="Classic Car RAG API",
    description="API de recherche sémantique pour voitures de collection"
)

print("⏳ Initialisation du système (Modèle E5 + API)...", flush=True)


model = SentenceTransformer("intfloat/e5-large-v2")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("INDEX_NAME2"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))




def get_classic_car_response(user_query: str, top_k: int):
    
    query_for_embedding = f"query: {user_query}"
    query_vector = model.encode([query_for_embedding], normalize_embeddings=True)[0].tolist()

   
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    
    context_entries = []
    for res in results['matches']:
        m = res['metadata']
        details = []
        for key, value in m.items():
            if value:
                val_str = ", ".join(value) if isinstance(value, list) else str(value)
                details.append(f"{key.replace('_', ' ').capitalize()}: {val_str}")
        context_entries.append("\n".join(details))

    context = "\n---\n".join(context_entries) if context_entries else "No relevant information found."

    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in classic cars. Your goal is to answer questions "
                "based ONLY on the provided context. \n"
                "RULES:\n"
                "1. If the answer is not in the context, reply exactly there's no data about this.\n"
                "2. Be technical and precise.\n"
                "3. Stay in the language of the user's question."
            )
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{user_query}"
        }
    ]

    completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=500
    )

    return completion.choices[0].message.content.strip()




# -------------------------
# Endpoints API
# -------------------------

@app.get("/")
async def health_check():
    return {"status": "online", "model": "e5-large-v2", "llm": "llama-3.3-70b"}

@app.post("/ask", response_model=QueryResponse)
async def ask_car_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="La requête ne peut pas être vide.")
    
    try:
        answer = get_classic_car_response(request.query, request.top_k)
        return QueryResponse(answer=answer)
    except Exception as e:
        print(f"❌ Erreur Serveur: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne lors du traitement de la requête.")
