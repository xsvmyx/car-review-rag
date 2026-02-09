from app.utilsLLM.groqCall import call_groq_with_retry
import json



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