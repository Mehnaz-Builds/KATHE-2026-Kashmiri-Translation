import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = "mehii123/kathe-2026-nllb-kashmiri"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Loads the fine-tuned KATHE 2026 NLLB tokenizer and model weights from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(DEVICE)
    return model, tokenizer

if __name__ == "__main__":
    print(f"Loading tokenizer and model from {MODEL_ID}...")
    model, tokenizer = load_model()
    print("Model and tokenizer loaded successfully!")
