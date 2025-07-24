# rag/retriever.py

import os
import json
import faiss
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# Load once
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()  # shape: (768,)

# Cache these in memory (can extend to save/load later)
_index = None
_metadata = None

def build_faiss_index():
    global _index, _metadata
    if _index is not None:
        return _index, _metadata

    path = os.path.join("data", "minimalist_logo_principles.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    metadata = []

    for category in data["categories"]:
        for principle in category["principles"]:
            texts.append(principle)
            metadata.append({
                "category_id": category["id"],
                "title": category["title"],
                "principle": principle
            })

    embeddings = np.array([embed_text(t) for t in texts], dtype=np.float32)

    index = faiss.IndexFlatL2(768)
    index.add(embeddings)

    _index = index
    _metadata = metadata

    return index, metadata

def retrieve_context(prompt, k=5):
    index, metadata = build_faiss_index()
    query_vec = embed_text(prompt).astype("float32").reshape(1, -1)

    D, I = index.search(query_vec, k)
    results = [metadata[i] for i in I[0]]

    return results
