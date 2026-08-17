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
| English Input | Fine-Tuned NLLB (11.03 BLEU) | Base IndicTrans2-1B | Positional Ensemble (10.64 BLEU) | Qualitative Observation |
|---|---|---|---|---|
| *These are some verities of human nature* | یِم چھِ انسٲنی فطرتٕچ کینٛہہ حٔقیٖقتہٕ۔ | یِم چھِ اِنسان کٲرۍ کینٛہہ حٔقیٖقت | یِم چھِ انسٲنی فطرتٕچ کینٛہہ حٔقیٖقت | NLLB provides superior diacritic alignment; IndicTrans2 maintains structure. |
| *She was a true visionary.* | سۄ ٲس اَکھ پۆز دُور اندیش۔ | سۄ ٲس اَکھ دُور اندیش | سۄ ٲس اَکھ پۆز دُور اندیش۔ | Ensemble successfully restores NLLB's diacritic marker (*پۆز*). |
### Metric vs. Quality Analysis & Final Decision
Both ensemble variants scored lower on automated metrics than the **standalone fine-tuned NLLB model (11.03 BLEU)**. This occurs because automated BLEU strictly rewards contiguous n-gram matches against reference text, and splicing tokens from two separate model outputs can disrupt exact n-gram sequences even when the result is human-readable.
