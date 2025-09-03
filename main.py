# main.py

import torch
from rag.retriever import retrieve_context
from deepseek.refiner import refine_prompt
from bert.embedding import embed_prompt
from utils.save_data import save_output

print("\n🚀 Starting REDO-AI Pipeline...")

# Let user provide inputs
design_brief = input("\n Enter your Design Brief: ")
instruction = input(" Enter your Instruction: ")

# Step 1: Retrieve principles
print("\n[MAIN] Retrieving related design principles...")
related = retrieve_context(instruction, k=5)
print(f"[MAIN] Retrieved {len(related)} related principles")

# Step 2: Refine using LLM
summary, detailed = refine_prompt(instruction, related, design_brief)
print("\n[MAIN] Refined Summary:", summary)

# Step 3: Embed refined prompt
embedding = embed_prompt(summary)
print("[MAIN] Embedding shape:", embedding.shape)

# Show first 5 dimensions of the vector
print("[MAIN] First 5 dims of embedding:", embedding[0][:5].tolist())

# Step 4: Save to dataset
save_output(instruction, design_brief, detailed, embedding)

print("\n✅ Pipeline finished successfully!")
