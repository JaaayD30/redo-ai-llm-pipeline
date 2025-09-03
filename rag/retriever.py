import os
import json
import faiss
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

print("[RAG] Initializing BERT embedder...")

_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
_model = BertModel.from_pretrained("bert-base-uncased")
_model.eval()

print("[RAG] ✅ BERT ready")

_index = None
_metadata = None

def _embed_text(text: str):
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = _model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()

def build_faiss_index():
    """Build FAISS index (cached after first call)."""
    global _index, _metadata
    if _index is not None:
        return _index, _metadata

    path = os.path.join("data", "minimalist_logo_principles.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, metadata = [], []
    for category in data["categories"]:
        for principle in category["principles"]:
            texts.append(principle)
            metadata.append({
                "category_id": category["id"],
                "title": category["title"],
                "principle": principle
            })

    embeddings = np.array([_embed_text(t) for t in texts], dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    _index, _metadata = index, metadata
    print(f"[RAG] ✅ FAISS index cached with {len(texts)} principles")

    return index, metadata

def retrieve_context(prompt: str, k: int = 5):
    """Retrieve top-k context principles for prompt."""
    index, metadata = build_faiss_index()
    query_vec = _embed_text(prompt).astype("float32").reshape(1, -1)

    _, I = index.search(query_vec, k)
    return [metadata[i] for i in I[0]]
