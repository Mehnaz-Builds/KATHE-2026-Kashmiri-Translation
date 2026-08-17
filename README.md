# KATHE 2026 — English to Kashmiri Translation

Official submission repository for the KATHE 2026 Competition (English to Kashmiri Perso-Arabic machine translation).

## 1. Primary Model & Weights
- **Base Model:** `facebook/nllb-200-distilled-600M`
- **Target Language:** Kashmiri Perso-Arabic (`kas_Arab`)
- **Hugging Face Model Weights:** [mehii123/kathe-2026-nllb-kashmiri](https://huggingface.co/mehii123/kathe-2026-nllb-kashmiri)
- **Validation Score:** 11.03 BLEU

## 2. File Overview
- `single_inference.py` — Translates a single English input sentence.
- `batch_inference.py` — Takes an input CSV, translates in GPU batches, and exports `submission.csv`.
- `requirements.txt` — Python dependencies needed to execute the scripts.

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
## 4. Generation Parameters Optimization
During model evaluation, we tuned the NLLB text generation parameters to prevent repetitive looping and improve sentence structure in Kashmiri script:
 * **Beam Search (num_beams):** Increased to 5 (up from standard 2) to explore higher-probability translation paths.
 * **Repetition Penalty:** Set to 1.2 to actively prevent repeating duplicate words or phrases.
 * **Max New Tokens:** Set to 128 to allow complete coverage for long sentences.
## 5. Methodology & Ensembling Analysis
### Fine-Tuning Approach
facebook/nllb-200-distilled-600M was fine-tuned on the provided BPCC dataset using Hugging Face Seq2SeqTrainer.
### Ensembling Experiments
We also experimented with combining our fine-tuned NLLB model with **IndicTrans2 (1B)** using sequence alignment (difflib.SequenceMatcher).
### Self-Evaluation & Findings
 * **BLEU Score:** Standalone fine-tuned NLLB scored **11.03 BLEU**, whereas the word-level ensemble scored **10.64 BLEU**.
 * **Qualitative Analysis:** While the ensemble generated slightly smoother Perso-Arabic diacritics, standalone fine-tuned NLLB provided higher n-gram accuracy against reference translations.
 * **Final Decision:** Because NLLB achieved the higher BLEU score and requires zero gated authentication tokens, standalone fine-tuned NLLB was chosen for the official evaluation pipeline.
```

