# KATHE 2026 — English to Kashmiri Machine Translation

Official repository for KATHE 2026 Competition: English to Kashmiri (Perso-Arabic script `kas_Arab`) Machine Translation.

## 1. Primary Model & Weights
- **Base Model:** `facebook/nllb-200-distilled-600M`
- **Fine-Tuned Target Language:** Kashmiri Perso-Arabic (`kas_Arab`)
- **Hugging Face Model Weights:** [mehii123/kathe-2026-nllb-kashmiri](https://huggingface.co/mehii123/kathe-2026-nllb-kashmiri)
- **Validation Score:** **11.03 BLEU**

## 2. File Overview
### Model Loading Interface
For modular integration or evaluation pipelines, the model can be instantiated directly via `load_model.py`:

```python
from load_model import load_model

model, tokenizer = load_model()
- `single_inference.py` — Translates a single English sentence directly in the terminal.
- `batch_inference.py` — Evaluator batch pipeline: parses input CSV and exports `submission.csv`.
- `ensemble_inference.py` — Exploratory word-level ensembling script (included for transparency).
- `requirements.txt` — Required Python dependencies.

## 3. How to Run

### Installation
```bash
pip install -r requirements.txt

```
### Single Sentence Inference
```bash
python single_inference.py "These are some verities of human nature"

```
### Batch Inference (Evaluator Pipeline)
```bash
python batch_inference.py englishdev.csv submission.csv

```
## 4. Generation Parameters
Decoding parameters optimized for BLEU score recovery:
 * **Beams (num_beams):** 2
 * **Length Penalty (length_penalty):** 1.0
 * **Early Stopping (early_stopping):** True
 * **Max New Tokens (max_new_tokens):** 100
## 5. Methodology & Ensembling Analysis
### Fine-Tuning
The base facebook/nllb-200-distilled-600M model was fine-tuned on the BPCC corpus using Hugging Face's Seq2SeqTrainer.
### Exploratory Ensembling
During development, we explored ensembling our fine-tuned NLLB model with a base IndicTrans2-1B model based on qualitative observations:
 * **Fine-Tuned NLLB:** Produced grammatically strong output with accurate diacritics, but occasionally drifted semantically on longer sentences.
 * **IndicTrans2 Base:** Preserved semantic intent well, but produced weaker or missing Perso-Arabic diacritics.
Two ensembling strategies were evaluated:
 1. **Sentence-Level Fallback (10.56 BLEU):** Defaulted to NLLB, falling back to IndicTrans2 only on blank or degenerate output. (Rarely triggered as NLLB failures were subtle drift rather than corruption).
 2. **Word-Level Positional Merge (10.64 BLEU):** Used IndicTrans2 as the structural base sentence, forced NLLB's first word, and aligned remaining tokens using difflib.SequenceMatcher. Words were substituted with NLLB's diacritic-rich tokens whenever positional similarity exceeded >= 0.75.
### Qualitative Output Comparison Table

| English Input | Fine-Tuned NLLB (11.03 BLEU) | Base IndicTrans2 (8.14 BLEU) | Word-Level Ensemble (10.64 BLEU) | Qualitative Observation |
|---|---|---|---|---|
| He is very popular actor. | سُ چُھ واریاہ مشہؤر اداکار۔ | سہ چھ واریاہ مشہؤر اداکار۔ | سُ چُھ واریاہ مشہؤر اداکار۔ | At every word position, NLLB and IndicTrans2 output the same underlying word, differing only in diacritic marks (سُ vs سہ, چُھ vs چھ). Since character-similarity between each pair was above the 0.75 threshold, the algorithm substituted NLLB's diacritic-marked form at every position — not just the first word. Result: the ensemble reproduces NLLB's full diacritic accuracy while having IndicTrans2's word choice validate that each substitution was safe. This is the ensemble logic working as designed: when both models agree on meaning, prefer NLLB's superior diacritic marking. |
### Metric vs. Quality Analysis & Final Decision
Both ensemble variants scored lower on automated metrics than the **standalone fine-tuned NLLB model (11.03 BLEU)**. This occurs because automated BLEU strictly rewards contiguous n-gram matches against reference text, and splicing tokens from two separate model outputs can disrupt exact n-gram sequences even when the result is human-readable.
