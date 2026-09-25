"""
OCR pipeline: preprocessing (deskew, denoise, binarize) + Tesseract, plus a
harness to measure raw OCR error rate against ground-truth text.

This measures the OCR step in isolation, BEFORE any fine-tuned extraction
model touches the text -- i.e. "how much noise is the extraction model
going to have to deal with", per the project brief's step 2.

Two Tesseract language configs are tried per image and the result is
compared: 'fra' (French only), 'ara' (Arabic only), and 'fra+ara'
(both loaded together) -- mixed FR/AR documents need the combined config,
but it's worth measuring whether combining language models hurts accuracy
on single-language documents (a real trade-off worth reporting).
"""

import os
import json
import glob
import numpy as np
import cv2
import pytesseract


# --------------------------------------------------------------------------
# Preprocessing: deskew, denoise, binarize
# --------------------------------------------------------------------------

def deskew(img_gray):
    """Estimate and correct small rotation via minAreaRect on thresholded
    text pixels. Falls back to no-op if too few pixels are found (blank
    or near-blank image)."""
    inv = cv2.bitwise_not(img_gray)
    thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return img_gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # Ignore near-zero corrections (avoid nudging already-straight images)
    if abs(angle) < 0.1:
        return img_gray
    (h, w) = img_gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)


def denoise(img_gray):
    return cv2.fastNlMeansDenoising(img_gray, h=10)


def binarize(img_gray):
    # Otsu's global threshold works well for our fairly uniform synthetic
    # backgrounds; adaptive thresholding is the fallback to try if real
    # scanned invoices have uneven lighting.
    return cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def preprocess_image(path, out_path=None):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = deskew(img)
    img = denoise(img)
    img = binarize(img)
    if out_path:
        cv2.imwrite(out_path, img)
    return img


# --------------------------------------------------------------------------
# Tesseract runner
# --------------------------------------------------------------------------

def run_tesseract(img, lang="fra+ara", psm=6):
    config = f"--psm {psm}"
    return pytesseract.image_to_string(img, lang=lang, config=config)


_BIDI_CONTROL_CHARS = "\u200e\u200f\u202a\u202b\u202c\u202d\u202e"


def strip_bidi_controls(text):
    """Tesseract emits invisible Unicode bidi control marks (LRM/RLM/
    embedding marks) as part of its right-to-left reconstruction attempt.
    They carry no visible character-error information and would inflate
    CER/WER with invisible 'errors' the reader never sees, so they are
    stripped before scoring (kept in the raw hypothesis for inspection)."""
    return "".join(ch for ch in text if ch not in _BIDI_CONTROL_CHARS)


# --------------------------------------------------------------------------
# Error metrics: character error rate (CER) and word error rate (WER),
# via Levenshtein edit distance -- standard OCR quality metrics.
# --------------------------------------------------------------------------

def _levenshtein(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def cer(hypothesis, reference):
    ref = reference.replace("\n", " ")
    hyp = hypothesis.replace("\n", " ")
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _levenshtein(hyp, ref) / len(ref)


def wer(hypothesis, reference):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return _levenshtein(hyp_words, ref_words) / len(ref_words)


# --------------------------------------------------------------------------
# Batch evaluation harness
# --------------------------------------------------------------------------

def evaluate_manifest(manifest_records, preprocessed_dir=None, langs=("fra+ara",), psm=6):
    """
    manifest_records: list of dicts with image_path, ground_truth_text, language
    Returns per-record results + summary stats per language config.
    """
    results = []
    for rec in manifest_records:
        entry = {"record_id": rec["record_id"], "language": rec["language"]}
        pre_path = None
        if preprocessed_dir:
            os.makedirs(preprocessed_dir, exist_ok=True)
            pre_path = os.path.join(preprocessed_dir, os.path.basename(rec["image_path"]))
        img = preprocess_image(rec["image_path"], out_path=pre_path)

        for lang_cfg in langs:
            hyp_raw = run_tesseract(img, lang=lang_cfg, psm=psm)
            hyp = strip_bidi_controls(hyp_raw)
            entry[f"cer_{lang_cfg}"] = cer(hyp, rec["ground_truth_text"])
            entry[f"wer_{lang_cfg}"] = wer(hyp, rec["ground_truth_text"])
            entry[f"hyp_{lang_cfg}"] = hyp
        results.append(entry)
    return results


def summarize(results, langs=("fra+ara",)):
    from collections import defaultdict
    by_lang_cfg = {lc: defaultdict(list) for lc in langs}
    for r in results:
        for lc in langs:
            by_lang_cfg[lc][r["language"]].append(r[f"cer_{lc}"])
            by_lang_cfg[lc]["ALL"].append(r[f"cer_{lc}"])

    print("\n=== OCR baseline error rates (mean CER, lower is better) ===")
    for lc in langs:
        print(f"\nTesseract lang config: {lc}")
        for doc_lang, values in by_lang_cfg[lc].items():
            mean_cer = sum(values) / len(values)
            print(f"  {doc_lang:8s} n={len(values):3d}  mean CER={mean_cer:.3f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True, help="Dir of rendered PNGs from render_images.py")
    p.add_argument("--jsonl", required=True, help="Original dataset JSONL (for ground truth text)")
    p.add_argument("--preprocessed-dir", default=None)
    p.add_argument("--langs", nargs="+", default=["fra+ara"])
    p.add_argument("--psm-sweep", nargs="+", type=int, default=None,
                    help="If given, try multiple PSM modes and report CER for each instead of a single run")
    p.add_argument("--psm", type=int, default=6)
    args = p.parse_args()

    # Rebuild the manifest by matching image files back to their labels
    gt_by_id = {}
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt_by_id[rec["labels"]["record_id"]] = rec

    manifest = []
    for img_path in sorted(glob.glob(os.path.join(args.images_dir, "*.png"))):
        rid = os.path.splitext(os.path.basename(img_path))[0]
        if rid not in gt_by_id:
            continue
        rec = gt_by_id[rid]
        manifest.append({
            "record_id": rid,
            "image_path": img_path,
            "language": rec["labels"]["language"],
            "ground_truth_text": rec["ocr_text"],
        })

    if args.psm_sweep:
        print(f"PSM sweep over {args.psm_sweep} on {len(manifest)} images, lang={args.langs[0]}")
        for psm in args.psm_sweep:
            results = evaluate_manifest(manifest, preprocessed_dir=None, langs=args.langs, psm=psm)
            print(f"\n--- PSM {psm} ---")
            summarize(results, langs=args.langs)
    else:
        print(f"Evaluating {len(manifest)} images with Tesseract configs: {args.langs}, psm={args.psm}")
        results = evaluate_manifest(manifest, preprocessed_dir=args.preprocessed_dir, langs=args.langs, psm=args.psm)
        summarize(results, langs=args.langs)

        with open("ocr_eval_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\nFull per-record results -> ocr_eval_results.json")
