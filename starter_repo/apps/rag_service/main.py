from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np, faiss, json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline

app = FastAPI(title="RAG Service", version="0.1.0")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "models" / "index" / "faiss.index"
TEXTS_PATH = PROJECT_ROOT / "models" / "index" / "texts.json"
_index = None
_texts: List[str] = []
_model: SentenceTransformer | None = None
_gen = None  # text2text-generation pipeline (Flan-T5)

def embed(text: str) -> np.ndarray:
    assert _model is not None, "Embedding model not loaded"
    v = _model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
    return v

def generate_answer(question: str, contexts: List[str]) -> str:
    if not contexts:
        return "I don't have enough context to answer."
    # Compose a simple prompt instructing model to use only provided context
    ctx = "\n".join(f"- {c}" for c in contexts)
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the context.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )
    try:
        out = _gen(prompt, max_new_tokens=128, do_sample=False, clean_up_tokenization_spaces=True)
        text = out[0]["generated_text"].strip()
        return text
    except Exception as e:
        return f"Generator error: {e}"

@app.on_event("startup")
def load_index():
    global _index, _texts, _model, _gen
    if INDEX_PATH.exists() and TEXTS_PATH.exists():
        _index = faiss.read_index(str(INDEX_PATH))
        _texts = json.loads(TEXTS_PATH.read_text(encoding="utf-8"))
        try:
            _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print(f"Index loaded with {len(_texts)} chunks from {INDEX_PATH.parent}. Embedding model ready.")
        except Exception as e:
            print(f"Warning: Failed loading embedding model: {e}")
        # Load local generator (no API key). Small model for fast CPU inference.
        try:
            _gen = hf_pipeline("text2text-generation", model="google/flan-t5-small")
            print("Generation model ready: google/flan-t5-small")
        except Exception as e:
            print(f"Warning: Failed loading generation model: {e}")
    else:
        print(f"Index not found at {INDEX_PATH} or texts at {TEXTS_PATH}. Run scripts/ingest_data.py and scripts/build_index.py first.")

class Query(BaseModel):
    question: str
    k: int = 5

@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": bool(_index), "chunks": len(_texts)}

@app.post("/query")
def query_rag(q: Query):
    if _index is None or not _texts:
        return {"error": "Index not loaded. Build it first."}
    if _model is None:
        return {"error": "Embedding model not loaded. Ensure internet access and restart service."}
    qv = embed(q.question)
    D, I = _index.search(np.expand_dims(qv, 0), min(q.k, len(_texts)))
    results = [{"rank": i+1, "score": float(D[0][i]), "text": _texts[idx]} for i, idx in enumerate(I[0])]
    if _gen is not None:
        top_ctx = [r["text"] for r in results]
        answer = generate_answer(q.question, top_ctx)
    else:
        answer = "Top matching chunks returned. (Generation model not loaded.)"
    return {"answer": answer, "context": results}
