from fastapi import FastAPI, HTTPException
from app.schemas import QueryRequest, QueryResponse
import app.config 
from app.Services.ragService import ask_car_reviews_bot_free_text

app = FastAPI(
    title="Car review RAG API"
)




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
        answer = ask_car_reviews_bot_free_text(request.query,request.top_k)
        return QueryResponse(answer=answer)
    except Exception as e:
        print(f"❌ Erreur Serveur: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne lors du traitement de la requête.")
