from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

app = FastAPI()

# Activar CORS para que el frontend pueda comunicarse
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, podés reemplazar * por tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

modelo = SentenceTransformer("all-MiniLM-L6-v2")

# Función para calcular similitud por producto punto (coseno)
def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Cargar los embeddings desde archivo JSON
def cargar_embeddings(archivo):
    with open(archivo, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@app.post("/ask")
async def responder_pregunta(req: Request):
    body = await req.json()
    pregunta = body.get("question")
    tema = body.get("topic")

    if tema == "all":
        archivos = [f for f in os.listdir("output_embeddings") if f.endswith(".json")]
    else:
        archivos = [f"{tema}.json"]

    mejor_respuesta = ""
    mayor_similitud = -1

    pregunta_embedding = modelo.encode(pregunta)

    for archivo in archivos:
        path = os.path.join("output_embeddings", archivo)
        embeddings = cargar_embeddings(path)

        for item in embeddings:
            sim = cosine_similarity(pregunta_embedding, item["embedding"])
            if sim > mayor_similitud:
                mayor_similitud = sim
                mejor_respuesta = item["text"]

    return JSONResponse(content={
        "respuesta": mejor_respuesta,
        "similitud": round(mayor_similitud, 4)
    })
