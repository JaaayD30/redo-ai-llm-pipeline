from transformers import BertTokenizer, BertModel
import torch

print("[BERT] Initializing... (first run may take time)")

# Load once
_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
_model = BertModel.from_pretrained("bert-base-uncased")
_model.eval()

print("[BERT] ✅ Model loaded successfully")

def embed_prompt(text: str):
    """Return mean pooled embedding of text (torch tensor)."""
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = _model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # shape (1, 768)
