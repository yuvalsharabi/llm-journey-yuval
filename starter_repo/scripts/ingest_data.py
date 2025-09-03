# Ingest + chunk simple text files from data/raw to data/processed (path-safe)
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW  = PROJECT_ROOT / "data" / "raw"
PROC = PROJECT_ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def chunk_text(t: str, max_len=800):
    words, cur, acc = t.split(), [], []
    cur_len = 0
    for w in words:
        wl = len(w) + 1
        if cur_len + wl > max_len and cur:
            acc.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(w)
        cur_len += wl
    if cur:
        acc.append(" ".join(cur))
    return acc

def main():
    print(f"[ingest] RAW:  {RAW}")
    print(f"[ingest] PROC: {PROC}")
    files = list(RAW.rglob("*.txt"))
    if not files:
        print("[ingest] No .txt files under data/raw. Add some and rerun.")
        return
    for p in files:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        txt = clean_text(txt)
        chunks = chunk_text(txt, max_len=800)
        out = PROC / f"{p.stem}.chunks.txt"
        out.write_text("\n\n".join(chunks), encoding="utf-8")
        print(f"[ingest] {p} -> {out} ({len(chunks)} chunks)")
    print("[ingest] Done.")

if __name__ == "__main__":
    main()
