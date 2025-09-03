# Yuval's LLM Learning Path 🚀

## 🎯 Goal
Become an expert in **LLM Engineering** in order to **lead development teams** and deliver production-grade AI solutions.

---

## 📍 Stage 1 – Foundations (Hands-on RAG)
- [x] Set up Python environment + venv.
- [x] Create project structure (`apps/`, `scripts/`, `data/`, `models/`).
- [x] Ingest raw documents → chunk into smaller pieces.
- [x] Build FAISS index (currently with **fake embeddings**).
- [x] Expose FastAPI service (`/health`, `/query`) returning top matching chunks.

✅ Outcome: You have a **working Retrieval system** (the “R” in RAG).

---

## 📍 Stage 2 – Real Embeddings
- [ ] Install `sentence-transformers`.
- [ ] Replace fake embeddings with real embeddings (`all-MiniLM-L6-v2`).
- [ ] Rebuild the index and test retrieval on larger documents.
- [ ] Validate that questions return relevant context chunks.

✅ Outcome: Retrieval is now **semantic**, not random.

---

## 📍 Stage 3 – Generation (the “G” in RAG)
- [ ] Add a text generator model (start with **Flan-T5** locally).
- [ ] Connect generator to retrieved chunks to form full answers.
- [ ] Compare answers with and without retrieval.

✅ Outcome: End-to-end **RAG pipeline**: docs → chunks → embeddings → retrieval → generated answer.

---

## 📍 Stage 4 – Observability & Metrics
- [ ] Add logging for queries (latency, #tokens, top-k hits).
- [ ] Track retrieval quality (are chunks relevant?).
- [ ] Simple evaluation dataset for Q&A.

✅ Outcome: Ability to measure and improve system performance.

---

## 📍 Stage 5 – Scale & Production
- [ ] Replace local FAISS with a vector DB (Pinecone, Weaviate, or pgvector).
- [ ] Add guardrails (e.g., max query length, input sanitization).
- [ ] Containerize with Docker.
- [ ] Deploy to AWS (API Gateway + Lambda / ECS).
- [ ] Secure with Cognito (authn/authz).
- [ ] Automate pipeline (ingest → index → deploy).

✅ Outcome: Production-ready **RAG system**, cloud-deployed, scalable, and secure.

---

## 📍 Stage 6 – Advanced Topics
- Fine-tuning with **PEFT/QLoRA**.
- Knowledge distillation to smaller models.
- Hybrid retrieval (sparse + dense).
- Integrations with Bedrock, SageMaker, or Azure OpenAI.
- Monitoring hallucinations, adding guardrails.

---

## 📝 Notes
- This file is your “map” – update checkboxes as you progress.
- At each stage, ask your GPT assistant:  
  *“Guide me to complete the next stage in my Learning Path.”*
- Keep documenting learnings in `notes/` for future reference.

---
