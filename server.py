from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import requests
import json
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")

def get_embedding(texto):
    if not HF_TOKEN:
        raise ValueError("🚨 HuggingFace API token (HF_TOKEN) is missing!")

    response = requests.post(
        "https://api-inference.huggingface.co/embeddings/sentence-transformers/all-MiniLM-L6-v2",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": texto},
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(f"🛑 HuggingFace API error {response.status_code}: {response.text}")

    data = response.json()
    if "embedding" not in data:
        raise RuntimeError("❌ 'embedding' not found in HuggingFace response")

    return data["embedding"]

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def cargar_embeddings(archivo):
    with open(archivo, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/ask")
async def responder_pregunta(req: Request):
    try:
        body = await req.json()
        pregunta = body.get("question")
        tema = body.get("topic")

        if not pregunta or not tema:
            return JSONResponse(status_code=400, content={"error": "Missing 'question' or 'topic'"})

        archivos = (
            [f"{tema}.json"] if tema != "all"
            else [f for f in os.listdir("output_embeddings") if f.endswith(".json")]
        )

        pregunta_embedding = get_embedding(pregunta)
        mejor_respuesta = ""
        mayor_similitud = -1

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

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": "Internal server error",
            "details": str(e)
        })
