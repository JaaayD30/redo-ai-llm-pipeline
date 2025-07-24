# rag/faiss_retriever.py

import json
import os
import faiss
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()  # shape: (768,)

def build_faiss_index():
    data_path = os.path.join("data", "minimalist_logo_principles.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    metadata = []

    for category in data["categories"]:
        cat_id = category["id"]
        cat_title = category["title"]
        for principle in category["principles"]:
            texts.append(principle)
            metadata.append({"category_id": cat_id, "title": cat_title, "principle": principle})

    # Embed all texts
    embeddings = np.array([embed_text(text) for text in texts], dtype="float32")

    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, metadata

def retrieve_top_k(prompt, k=3):
    index, metadata = build_faiss_index()
    query_emb = embed_text(prompt).astype("float32").reshape(1, -1)

    D, I = index.search(query_emb, k)
    results = [metadata[i] for i in I[0]]
    return results
