import os
import json
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np

app = FastAPI()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/embeddings/sentence-transformers/all-MiniLM-L6-v2"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

def get_embedding(text):
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": text})
    if response.status_code != 200:
        raise Exception(f"Error from HF API: {response.status_code} - {response.text}")
    return response.json()["embedding"]

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def cargar_embeddings(archivo):
    with open(archivo, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/ask")
async def responder_pregunta(req: Request):
    body = await req.json()
    pregunta = body.get("question")
    tema = body.get("topic")

    if not pregunta or not tema:
        return JSONResponse({"error": "Missing data"}, status_code=400)

    archivos = [f for f in os.listdir() if f.endswith(".json")] if tema.lower() == "all" else [f"{tema}.json"]

    mejor_score = -1
    mejor_texto = ""
    pregunta_vec = get_embedding(pregunta)

    for archivo in archivos:
        try:
            datos = cargar_embeddings(archivo)
            for item in datos:
                score = cosine_similarity(pregunta_vec, item["embedding"])
                if score > mejor_score:
                    mejor_score = score
                    mejor_texto = item["text"]
        except:
            continue

    return {
        "respuesta": mejor_texto,
        "similitud": round(mejor_score, 4)
    }
