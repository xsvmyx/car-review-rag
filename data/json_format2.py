import pandas as pd
import re
import json

df = pd.read_csv("data/train_car.csv") 

# Fonctions d'extraction (inchangées)
def extract_year(title):
    match = re.search(r'(\d{4})', title)
    return int(match.group(1)) if match else None

def extract_engine(title):
    match = re.search(r'\((.*?)\)', title)
    return match.group(1) if match else "unknown"

def extract_model(title):
    model = re.sub(r'\d{4}', '', title)
    model = re.sub(r'\(.*?\)', '', model)
    return model.strip()

json_list = []

for _, row in df.iterrows():
    vehicle_title = str(row['Vehicle_Title'])
    review_title = str(row['Review_Title']) # On récupère le titre de l'avis
    review_content = str(row['Review'])      # Le corps de l'avis
    rating = row['Rating']

    year = extract_year(vehicle_title)
    model = extract_model(vehicle_title)
    engine = extract_engine(vehicle_title)

    # Concaténation demandée : Title : Review_Title : Review_Content
    full_review_text = f"{vehicle_title} : {review_title} : {review_content}"

    review_json = {
        "year": year,
        "model": model,
        "engine": engine,
        "rating": rating,
        "review": full_review_text # Texte concaténé
    }

    json_list.append(review_json)

with open("reviews_processed2.json", "w", encoding="utf-8") as f:
    json.dump(json_list, f, ensure_ascii=False, indent=2)

print(f"✅ JSON généré avec {len(json_list)} reviews !")