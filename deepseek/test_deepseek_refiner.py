from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Simulated chat message
messages = [
    {"role": "user", "content": "Who are you?"},
]

# Apply chat template
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

# Generate response
outputs = model.generate(**inputs, max_new_tokens=50)

# Decode response
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("\n🤖 DeepSeek Response:\n", response)
