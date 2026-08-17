"""
Exploratory ensemble script for transparency into evaluation experiments.
Combines fine-tuned NLLB-200 with base IndicTrans2-1B.
Final submission uses standalone single_inference.py / batch_inference.py.
"""

import difflib
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

NLLB_MODEL_ID = "mehii123/kathe-2026-nllb-kashmiri"
INDIC_MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"

def is_bad_output(text, min_len=2):
    if not isinstance(text, str) or text.strip() == "":
        return True
    return len(text.strip()) < min_len

def words_similar(w1, w2, threshold=0.75):
    if not w1 or not w2:
        return False
    return difflib.SequenceMatcher(None, w1, w2).ratio() >= threshold

def dedupe_consecutive(words):
    if not words:
        return words
    cleaned = [words[0]]
    for w in words[1:]:
        if w != cleaned[-1]:
            cleaned.append(w)
    return cleaned

def word_level_ensemble(nllb_text, indic_text):
    if is_bad_output(indic_text):
        return nllb_text if not is_bad_output(nllb_text) else ""
    if is_bad_output(nllb_text):
        return indic_text

    nllb_words = nllb_text.split()
    indic_words = indic_text.split()
    if not indic_words:
        return nllb_text

    result = list(indic_words)
    if nllb_words:
        result[0] = nllb_words[0]

    sm = difflib.SequenceMatcher(None, nllb_words, indic_words)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ('equal', 'replace'):
            span = min(i2 - i1, j2 - j1)
            for k in range(span):
                ni, ji = i1 + k, j1 + k
                if ji == 0:
                    continue
                if ni < len(nllb_words) and words_similar(nllb_words[ni], indic_words[ji]):
                    result[ji] = nllb_words[ni]

    return " ".join(dedupe_consecutive(result))

if __name__ == "__main__":
    sample_text = "These are some verities of human nature"
    print(f"English Input: {sample_text}")
    print("Ensemble evaluation pipeline ready.")
