from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("[DeepSeek] Initializing model deepseek-ai/deepseek-coder-1.3b-instruct...")

_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model.to(_device)
_model.eval()

print(f"[DeepSeek] ✅ Model ready (device: {_device})")

def refine_prompt(instruction: str, context_list: list, design_brief: str):
    """Refine a design instruction using DeepSeek LLM."""
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

    print("\n🧪 [DeepSeek] Prompt passed to model:")
    print("=" * 60)
    print(full_prompt)
    print("=" * 60)

    inputs = _tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True).to(_device)

    outputs = _model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,
        pad_token_id=_tokenizer.eos_token_id
    )

    response = _tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    print("\n🧪 [DeepSeek] Raw Response:")
    print("=" * 60)
    print(response)
    print("=" * 60)

    if "---" in response:
        summary_part, detail_part = response.split("---", 1)
    else:
        summary_part = response.strip().split("\n")[0]
        detail_part = response.strip()

    return summary_part.strip(), detail_part.strip()
