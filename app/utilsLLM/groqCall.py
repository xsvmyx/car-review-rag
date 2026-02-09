from groq import Groq
from app.config import GROQ_KEYS, current_key_index, client



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

