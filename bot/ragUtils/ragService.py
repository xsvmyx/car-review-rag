from config import model, index
from bot.ragUtils.groqPrompt import build_context, generate_response
from bot.ragUtils.extractVehicles import extract_vehicles_from_query



# -------------------------
# main RAG function
# -------------------------
def ask_car_reviews_bot_free_text(user_query, top_k=5, min_score=0.65):

    # Extraction des véhicules (JUSTE pour affichage et détection de comparaison)
    vehicles = extract_vehicles_from_query(user_query)
    if vehicles:
        print(f"📌 About: {', '.join(vehicles)}")
    
    # Embedding de la query
    query_vector = model.encode(
        ["query: " + user_query],
        normalize_embeddings=True
    )[0].tolist()
    
    # Recherche Pinecone 
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    # Filtrage par score UNIQUEMENT
    filtered_matches = [m for m in results.matches if m.score >= min_score]
    
    if not filtered_matches:
        return "I don't know."
    
    # Construction du contexte
    context = build_context(filtered_matches)
    
    # Génération (comparaison si plusieurs véhicules détectés)
    is_comparison = len(vehicles) > 1
    return generate_response(user_query, context, is_comparison)