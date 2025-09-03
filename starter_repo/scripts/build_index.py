"""
Build a FAISS index over processed chunks using real sentence-transformer embeddings.
Stage 2: replace fake embeddings with 'all-MiniLM-L6-v2'.
"""
import numpy as np, faiss, json
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROC    = PROJECT_ROOT / "data" / "processed"
IDX_DIR = PROJECT_ROOT / "models" / "index"
IDX_DIR.mkdir(parents=True, exist_ok=True)

def get_model() -> SentenceTransformer:
    # Explicit repo slug ensures correct model
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def main():
    print(f"[index] PROC:    {PROC}")
    print(f"[index] IDX_DIR: {IDX_DIR}")
    texts = []
    for p in PROC.glob("*.chunks.txt"):
        chunks = p.read_text(encoding="utf-8").split("\n\n")
        texts.extend([c.strip() for c in chunks if c.strip()])

    if not texts:
        print("[index] No processed chunks found. Run scripts/ingest_data.py first.")
        return

    model = get_model()
    dim = model.get_sentence_embedding_dimension()
    # Use cosine similarity via normalized embeddings with inner product index
    xb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(dim)
    index.add(xb)

    faiss.write_index(index, str(IDX_DIR / "faiss.index"))
    (IDX_DIR / "texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    print(f"[index] Built index with {len(texts)} chunks at {IDX_DIR}.")

if __name__ == "__main__":
    main()
