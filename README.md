# Invoice Extraction Agent

French / Tunisian-Arabic invoice extraction: synthetic data generation,
OCR pipeline, zero-shot baseline harness, QLoRA fine-tuning (Qwen3-8B via
Unsloth), validation layer, and evaluation.

## Files
- `generator.py` — synthetic FR/AR/mixed invoice generator (text + labels)
- `render_images.py` — renders invoices to PNG images for OCR
- `ocr_pipeline.py` — Tesseract OCR + preprocessing + CER/WER scoring
- `eval_harness.py` — zero-shot / fine-tuned extraction scoring (per-field P/R, bootstrap CIs)
