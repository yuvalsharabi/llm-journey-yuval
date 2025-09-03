import os, zipfile, pathlib

# Define structure
paths = [
    "apps/rag_service",
    "apps/finetune",
    "data/raw",
    "data/processed",
    "models",
    "scripts",
    "tests",
]

files = {
    "README.md": "# LLM Journey Starter – Yuval\n\nQuickstart for RAG + Fine-Tuning projects (AWS-first).",
    "requirements.txt": "fastapi\nuvicorn\npydantic\ntorch\ntransformers\ndatasets\naccelerate\npeft\nbitsandbytes\nfaiss-cpu\nnumpy\npython-dotenv\nboto3\n",
    "Dockerfile": "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nEXPOSE 8000\nCMD [\"uvicorn\",\"apps.rag_service.main:app\",\"--host\",\"0.0.0.0\",\"--port\",\"8000\"]",
    "Makefile": "build:\n\tdocker build -t llm-journey .\nrun:\n\tdocker run --rm -p 8000:8000 llm-journey",
    "apps/rag_service/main.py": "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/health')\ndef health():\n    return {'status':'ok'}",
    "apps/finetune/finetune.py": "print('QLoRA training skeleton here')",
    "scripts/ingest_data.py": "print('Ingest + chunk data skeleton')",
    "scripts/build_index.py": "print('Build FAISS index skeleton')",
    "scripts/eval.py": "print('Eval skeleton')",
    "scripts/deploy.py": "print('Deploy skeleton')",
    "tests/test_smoke.py": "def test_smoke():\n    assert True",
}

base = pathlib.Path("starter_repo")
for p in paths:
    (base / p).mkdir(parents=True, exist_ok=True)

for rel, content in files.items():
    f = base / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content)

# Create zip
zip_path = "llm_journey_starter.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in base.rglob("*"):
        if f.is_file():
            z.write(f, f.relative_to(base.parent))

print(f"Created {zip_path}")
