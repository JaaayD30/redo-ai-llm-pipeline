# utils/save_data.py

import os
import json

def save_output(instruction, brief, refined_prompt, embedding, save_path="outputs/training_data.jsonl"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = {
        "instruction": instruction,
        "brief": brief,
        "refined_prompt": refined_prompt,
        "embedding": embedding.tolist()  # convert tensor to list for JSON
    }

    with open(save_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")
