from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def refine_prompt(instruction, context_list, design_brief):
    context_text = "\n".join([f"- ({item['title']}) {item['principle']}" for item in context_list])

    full_prompt = f"""
You are an expert logo design assistant. Your task is to refine user instructions for a logo based on a design brief and related design principles.

Design Brief:
{design_brief}

Instruction:
{instruction}

Design Principles:
{context_text}

Return only one professional sentence summarizing the refined instruction for display to the user.
Then on the next line, write: ---
Then write the detailed breakdown of that prompt for internal use only.
""".strip()

    print("\n🧪 Prompt passed to DeepSeek:")
    print("=" * 60)
    print(full_prompt)
    print("=" * 60)

    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    # Debug logic
    print("\n🧪 LLM Response:")
    print("=" * 60)
    print(response)
    print("=" * 60)

    if "---" in response:
        summary_part, detail_part = response.split("---", 1)
        print("\n✅ [DEBUG] Summary extracted successfully.")
    else:
        summary_part = response.strip().split("\n")[0]
        detail_part = response.strip()
        print("\n⚠️ [DEBUG] Failed to split with '---'. Using fallback method.")

    print("\n[DEBUG] Summary Preview:")
    print(summary_part.strip())

    print("\n[DEBUG] Detail Preview (first 2 lines):")
    print("\n".join(detail_part.strip().splitlines()[:2]))

    return summary_part.strip(), detail_part.strip()
