import json
import os
import faiss
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

print("[FAISS] Initializing BERT embedder...")

_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
_model = BertModel.from_pretrained("bert-base-uncased")
_model.eval()

print("[FAISS] ✅ BERT ready")

def embed_text(text: str):
    """Return mean pooled embedding as numpy array (768-dim)."""
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = _model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()

def build_faiss_index():
    """Build FAISS index every time from minimalist_logo_principles.json."""
    data_path = os.path.join("data", "minimalist_logo_principles.json")
    with open(data_path, "r", encoding="utf-8") as f:
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

    embeddings = np.array([embed_text(t) for t in texts], dtype="float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    print(f"[FAISS] ✅ Index built with {len(texts)} principles")

    return index, metadata

def retrieve_top_k(prompt: str, k: int = 3):
    """Retrieve top-k design principles for a prompt."""
    index, metadata = build_faiss_index()
    query_emb = embed_text(prompt).astype("float32").reshape(1, -1)

    _, I = index.search(query_emb, k)
    return [metadata[i] for i in I[0]]
