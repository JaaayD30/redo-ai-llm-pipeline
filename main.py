# main.py

import os
import json
from rag.retriever import retrieve_context
from deepseek.refiner import refine_prompt
from bert.embedding import embed_prompt

OUTPUT_PATH = os.path.join("outputs", "training_data.jsonl")
os.makedirs("outputs", exist_ok=True)

print("REDO AI Pipeline Started\n")

while True:
    print("Enter a logo instruction (or type 'exit' to quit):")
    instruction = input("Instruction: ").strip()
    if instruction.lower() == "exit":
        break

    design_brief = input("Design Brief: ").strip()

    # Step 1: Retrieve context
    context = retrieve_context(instruction)
    print("\n[1] Retrieved Context:")
    for item in context:
        print(f"- ({item['title']}) {item['principle']}")

    # Step 2: Refine prompt using DeepSeek
    summary_prompt, detailed_prompt = refine_prompt(instruction, context, design_brief)

    print("\n[2] 🪄 Refined Prompt (for user display):")
    print(summary_prompt)

    print("\n[2.1] Refined Details (sent to BERT):")
    print("=" * 30)
    print(detailed_prompt)
    print("=" * 30)

    # Step 3: Embed the detailed version
    vector = embed_prompt(detailed_prompt)[0].tolist()
    print("\n[3] BERT Embedding (first 5 dims):")
    print(vector[:5])

    # Step 4: Save training data
    entry = {
        "instruction": instruction,
        "design_brief": design_brief,
        "summary_prompt": summary_prompt,
        "refined_prompt": detailed_prompt,
        "embedding": vector
    }

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"\n✅ Saved to {OUTPUT_PATH}")
    print("\n" + "-"*60 + "\n")
