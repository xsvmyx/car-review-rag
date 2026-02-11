from bot.ragUtils.groqCall import call_groq_with_retry


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
                "7. Pay attention to the YEAR and MODEL in each review - use the exact vehicles asked about.\n"
                "8. if you don't know don't say anything at all, just say \"I don't know about this.\"\n"
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
