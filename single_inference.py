import sys
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "mehii123/kathe-2026-nllb-kashmiri" 

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(DEVICE)

def translate(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("kas_Arab"),
        max_new_tokens=128,
        num_beams=5,
        repetition_penalty=1.2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "These are some verities of human nature."
    print(f"Output: {translate(text)}")
