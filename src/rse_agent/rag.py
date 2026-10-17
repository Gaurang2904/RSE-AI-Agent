"""Retrieval-augmented generation over the knowledge base (knowledge/*.md).

    python -m rse_agent.rag build            # (re)build the index
    python -m rse_agent.rag "why term insurance with dependants?"

- Chunking: one passage per "## " section, prefixed with its document title, so every passage
  has a citable source "Document § Section".
- Embeddings: nomic-embed-text through the local Ollama API (no data leaves the machine), with
  the model's "search_document:" / "search_query:" task prefixes; vectors are L2-normalised.
- Index: FAISS IndexFlatIP (exact cosine similarity). Stored in RSE_PRIVATE/rag_index together
  with a content hash; it rebuilds automatically when any knowledge file changes.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import faiss
import httpx
import numpy as np

from .config import KNOWLEDGE_DIR, OLLAMA_HOST, PRIVATE
from .guardrails import assert_local

EMBED_MODEL = "nomic-embed-text"
INDEX_DIR = PRIVATE / "rag_index"
MIN_SCORE = 0.45          # below this a passage is reported as weakly related


def load_passages(kb: Path = KNOWLEDGE_DIR) -> list[dict]:
    passages = []
    for path in sorted(kb.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        m = re.search(r"^# (.+)$", text, re.M)
        title = m.group(1).strip() if m else path.stem
        for sec in re.split(r"^## ", text, flags=re.M)[1:]:
            heading, _, body = sec.partition("\n")
            body = " ".join(body.split())
            if body:
                passages.append({"doc": path.name, "title": title, "section": heading.strip(),
                                 "source": f"{title} § {heading.strip()}", "text": body})
    return passages


def _kb_hash(kb: Path = KNOWLEDGE_DIR) -> str:
    h = hashlib.sha256(EMBED_MODEL.encode())
    for p in sorted(kb.glob("*.md")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def embed(texts: list[str], kind: str) -> np.ndarray:
    """kind = 'document' or 'query' (nomic task prefixes)."""
    assert_local(OLLAMA_HOST)
    r = httpx.post(f"{OLLAMA_HOST}/api/embed", timeout=60,
                   json={"model": EMBED_MODEL, "input": [f"search_{kind}: {t}" for t in texts]})
    r.raise_for_status()
    vecs = np.array(r.json()["embeddings"], dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs


def build(kb: Path = KNOWLEDGE_DIR, out: Path = INDEX_DIR) -> dict:
    passages = load_passages(kb)
    vecs = embed([f"{p['title']}. {p['section']}. {p['text']}" for p in passages], "document")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "kb.faiss"))
    meta = {"hash": _kb_hash(kb), "model": EMBED_MODEL, "passages": passages}
    (out / "kb.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    _load.cache_clear()
    return {"passages": len(passages), "documents": len({p["doc"] for p in passages}), "dim": vecs.shape[1]}


@lru_cache(maxsize=1)
def _load(out: Path = INDEX_DIR):
    meta_path = out / "kb.json"
    if not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8"))["hash"] != _kb_hash():
        build(out=out)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return faiss.read_index(str(out / "kb.faiss")), meta["passages"]


def search(query: str, k: int = 4) -> list[dict]:
    index, passages = _load()
    scores, rows = index.search(embed([query], "query"), min(k, len(passages)))
    return [{**passages[i], "score": round(float(s), 3)} for s, i in zip(scores[0], rows[0]) if i >= 0]


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        print(build())
    else:
        for hit in search(" ".join(sys.argv[1:])):
            print(f"{hit['score']:.3f}  {hit['source']}")
