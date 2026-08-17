import sys
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "mehii123/kathe-2026-nllb-kashmiri"

def run_batch(input_csv: str, output_csv: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(DEVICE)

    df = pd.read_csv(input_csv)
    eng_col = [c for c in df.columns if c.lower() != 'id'][0]
    sentences = df[eng_col].astype(str).tolist()

    translations = []
    batch_size = 32

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("kas_Arab"),
                max_new_tokens=100,
                num_beams=2,
                length_penalty=1.0,
                early_stopping=True
            )
            translations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    sub_df = pd.DataFrame({
        'ID': df['ID'] if 'ID' in df.columns else df.index + 1,
        'kashmiri_text': translations
    })
    sub_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "englishdev.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "submission.csv"
    run_batch(inp, out)
