from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

app = FastAPI()
modelo = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def cargar_embeddings(archivo):
    with open(archivo, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@app.post("/ask")
async def responder_pregunta(req: Request):
    body = await req.json()
    pregunta = body.get("question")
    tema = body.get("topic")

    if not pregunta or not tema:
        return JSONResponse({"error": "Missing data"}, status_code=400)

    if tema.lower() == "all":
        archivos = [f for f in os.listdir() if f.endswith(".json")]
    else:
        archivos = [f"{tema}.json"]

    mejor_score = -1
    mejor_texto = ""

    pregunta_vec = modelo.encode(pregunta)

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
