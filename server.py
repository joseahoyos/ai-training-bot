from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import requests
import json
import os

app = FastAPI()

# Activar CORS para que el frontend pueda comunicarse
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TOKEN de HuggingFace (se debe poner como variable de entorno en Render)
HF_TOKEN = os.getenv("HF_TOKEN")

# Función para obtener embedding desde la API
def get_embedding(texto):
    response = requests.post(
        "https://api-inference.huggingface.co/embeddings/sentence-transformers/all-MiniLM-L6-v2",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": texto},
    )
    response.raise_for_status()
    return response.json()["embedding"]

# Producto punto (coseno)
def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Cargar los embeddings desde archivo JSON
def cargar_embeddings(archivo):
    with open(archivo, "r", encoding="utf-8") as f:
        return json.load(f)

# Endpoint principal
@app.post("/ask")
async def responder_pregunta(req: Request):
    body = await req.json()
    pregunta = body.get("question")
    tema = body.get("topic")

    if not pregunta or not tema:
        return JSONResponse(status_code=400, content={"error": "Missing question or topic"})

    archivos = (
        [f"{tema}.json"] if tema != "all"
        else [f for f in os.listdir("output_embeddings") if f.endswith(".json")]
    )

    mejor_respuesta = ""
    mayor_similitud = -1

    try:
        pregunta_embedding = get_embedding(pregunta)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Error getting embedding", "details": str(e)})

    for archivo in archivos:
        path = os.path.join("output_embeddings", archivo)
        data = cargar_embeddings(path)

        for item in data:
            sim = cosine_similarity(pregunta_embedding, item["embedding"])
            if sim > mayor_similitud:
                mayor_similitud = sim
                mejor_respuesta = item["text"]

    return JSONResponse(content={
        "respuesta": mejor_respuesta,
        "similitud": round(mayor_similitud, 4)
    })
