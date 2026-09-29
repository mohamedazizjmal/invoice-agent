"""
Zero-shot (pre-fine-tuning) baseline evaluation harness for the invoice
extraction agent.

This is the SAME harness you'll reuse to score the fine-tuned model later
-- the whole point of step 4 in the build order is to lock in exactly this
scoring methodology now, on the un-tuned model, so the "baseline vs
fine-tuned" comparison in the final report is apples-to-apples.

IMPORTANT CAVEAT (read before trusting any numbers from --mode mock):
This sandbox has no GPU and no access to huggingface.co, so Qwen3-8B
cannot actually run here. `--mode mock` exists ONLY to prove the scoring
logic itself is correct (parsing, field matching, CI computation) against
a fake model that deliberately makes realistic mistakes (missing fields,
type errors, line-item reordering, minor date-format drift). Numbers from
mock mode are NOT a real baseline -- do not put them in your report.
`--mode qwen` is the real path, meant to run on Kaggle/Colab with
transformers + Unsloth installed (see qwen_predict_fn below).
"""

import argparse
import json
import os
import random
import re
import difflib
from datetime import datetime

# --------------------------------------------------------------------------
# 1. Zero-shot prompt
# --------------------------------------------------------------------------

FIELD_SCHEMA = {
    "invoice_number": "string",
    "issue_date": "string, format DD/MM/YYYY",
    "vendor_name": "string",
    "vendor_tax_id": "string (Tunisian Matricule Fiscal)",
    "line_items": "array of {description, quantity, unit_price, line_total}",
    "subtotal": "number",
    "vat_rate": "number, as a decimal e.g. 0.19 for 19%",
    "vat_amount": "number",
    "grand_total": "number",
    "currency": "string, e.g. TND",
}

PROMPT_TEMPLATE = """Tu es un système d'extraction de données pour des factures tunisiennes, rédigées en français, en arabe, ou dans un mélange des deux.

Voici le texte brut d'une facture (issu d'un OCR) :
---
{ocr_text}
---

Extrais les champs suivants et retourne UNIQUEMENT un objet JSON valide, sans aucun texte avant ou après, sans balises markdown, correspondant exactement à ce schéma :

{schema}

Règles :
- Si un champ est absent ou illisible, mets la valeur null.
- Les nombres doivent être des nombres JSON (pas de chaînes, pas de séparateurs de milliers).
- vat_rate est un taux décimal (19% -> 0.19).
- Les nombres du texte source utilisent l'espace comme séparateur de milliers et le
  point comme séparateur décimal (exemple : 19 286.595 signifie 19286.595, PAS
  19286595). Conserve toujours le point décimal, ne le supprime jamais.
- N'invente aucune valeur qui n'apparaît pas dans le texte.

JSON:"""


def build_prompt(ocr_text):
    schema_str = json.dumps(FIELD_SCHEMA, ensure_ascii=False, indent=2)
    return PROMPT_TEMPLATE.format(ocr_text=ocr_text, schema=schema_str)


# --------------------------------------------------------------------------
# 2. JSON parsing / repair
# --------------------------------------------------------------------------

def extract_json(raw_text):
    """LLMs frequently wrap JSON in markdown fences or add stray prose.
    Strip fences, find the first balanced {...} block, and attempt
    progressively more forgiving parses before giving up."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Find first balanced brace block (handles trailing commentary after the JSON)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    end = None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    candidate = text[start:end]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Common repair: trailing commas before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------
# 3. Field-level comparison
# --------------------------------------------------------------------------

SCALAR_FIELDS = ["invoice_number", "issue_date", "vendor_name", "vendor_tax_id",
                  "subtotal", "vat_rate", "vat_amount", "grand_total", "currency"]
LINE_ITEM_SUBFIELDS = ["description", "quantity", "unit_price", "line_total"]

NUMERIC_FIELDS = {"subtotal", "vat_rate", "vat_amount", "grand_total", "quantity", "unit_price", "line_total"}


def _norm_str(s):
    if s is None:
        return None
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _norm_number(x, tol_field=None):
    if x is None:
        return None
    try:
        if isinstance(x, str):
            x = x.replace(" ", "").replace(",", ".")
        return round(float(x), 2)
    except (ValueError, TypeError):
        return None


def _values_match(field, gold_val, pred_val, numeric_tol=0.02):
    if gold_val is None and pred_val is None:
        return True
    if gold_val is None or pred_val is None:
        return False
    if field in NUMERIC_FIELDS:
        g, p = _norm_number(gold_val), _norm_number(pred_val)
        if g is None or p is None:
            return False
        return abs(g - p) <= max(numeric_tol, abs(g) * 0.005)  # 0.5% relative or 0.02 abs
    return _norm_str(gold_val) == _norm_str(pred_val)


def match_line_items(gold_items, pred_items):
    """Greedy best-match line items by description similarity (order in
    the document isn't guaranteed to be preserved by the model), then
    compare subfields on matched pairs. Unmatched gold items are misses
    (false negatives); unmatched predicted items are hallucinated line
    items (false positives)."""
    gold_items = gold_items or []
    pred_items = pred_items or []
    remaining_pred = list(enumerate(pred_items))
    matches = []  # (gold_item, pred_item)
    unmatched_gold = []

    for g in gold_items:
        if not remaining_pred:
            unmatched_gold.append(g)
            continue
        gdesc = _norm_str(g.get("description", "")) or ""
        best_idx, best_score = None, -1
        for idx, (_, p) in enumerate(remaining_pred):
            pdesc = _norm_str(p.get("description", "")) or ""
            score = difflib.SequenceMatcher(None, gdesc, pdesc).ratio()
            if score > best_score:
                best_score, best_idx = score, idx
        if best_score is not None and best_score >= 0.5:
            _, matched_pred = remaining_pred.pop(best_idx)
            matches.append((g, matched_pred))
        else:
            unmatched_gold.append(g)

    unmatched_pred = [p for _, p in remaining_pred]
    return matches, unmatched_gold, unmatched_pred


def score_record(gold, pred):
    """Returns per-field tp/fp/fn counts for one record.
    tp = correct non-null value, fp = predicted a (wrong or hallucinated)
    value where gold differs, fn = missed a value gold had."""
    counts = {f: {"tp": 0, "fp": 0, "fn": 0} for f in SCALAR_FIELDS + LINE_ITEM_SUBFIELDS}

    if pred is None:
        # Total parse failure: every gold non-null field is a miss
        for f in SCALAR_FIELDS:
            if gold.get(f) is not None:
                counts[f]["fn"] += 1
        for item in gold.get("line_items", []) or []:
            for sf in LINE_ITEM_SUBFIELDS:
                if item.get(sf) is not None:
                    counts[sf]["fn"] += 1
        return counts

    for f in SCALAR_FIELDS:
        gv, pv = gold.get(f), pred.get(f)
        if _values_match(f, gv, pv):
            if gv is not None:
                counts[f]["tp"] += 1
        else:
            if gv is not None:
                counts[f]["fn"] += 1
            if pv is not None:
                counts[f]["fp"] += 1

    matches, unmatched_gold, unmatched_pred = match_line_items(
        gold.get("line_items", []), pred.get("line_items", []) if isinstance(pred.get("line_items"), list) else []
    )
    for g_item, p_item in matches:
        for sf in LINE_ITEM_SUBFIELDS:
            gv, pv = g_item.get(sf), p_item.get(sf)
            if _values_match(sf, gv, pv):
                if gv is not None:
                    counts[sf]["tp"] += 1
            else:
                if gv is not None:
                    counts[sf]["fn"] += 1
                if pv is not None:
                    counts[sf]["fp"] += 1
    for g_item in unmatched_gold:
        for sf in LINE_ITEM_SUBFIELDS:
            if g_item.get(sf) is not None:
                counts[sf]["fn"] += 1
    for p_item in unmatched_pred:
        for sf in LINE_ITEM_SUBFIELDS:
            if p_item.get(sf) is not None:
                counts[sf]["fp"] += 1

    return counts


def aggregate_counts(all_counts):
    fields = all_counts[0].keys()
    agg = {f: {"tp": 0, "fp": 0, "fn": 0} for f in fields}
    for rec_counts in all_counts:
        for f in fields:
            for k in ("tp", "fp", "fn"):
                agg[f][k] += rec_counts[f][k]
    summary = {}
    for f, c in agg.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 and not (precision != precision) and not (recall != recall)
              else float("nan"))
        summary[f] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return summary


# --------------------------------------------------------------------------
# 4. Bootstrap confidence intervals (per-field recall, record-level resampling)
# --------------------------------------------------------------------------

def bootstrap_field_recall_ci(all_counts, field, n_boot=2000, seed=42, alpha=0.05):
    rng = random.Random(seed)
    n = len(all_counts)
    if n == 0:
        return None
    boot_recalls = []
    for _ in range(n_boot):
        sample = [all_counts[rng.randrange(n)] for _ in range(n)]
        tp = sum(c[field]["tp"] for c in sample)
        fn = sum(c[field]["fn"] for c in sample)
        if tp + fn > 0:
            boot_recalls.append(tp / (tp + fn))
    if not boot_recalls:
        return None
    boot_recalls.sort()
    lo_idx = int((alpha / 2) * len(boot_recalls))
    hi_idx = int((1 - alpha / 2) * len(boot_recalls)) - 1
    return {
        "mean": sum(boot_recalls) / len(boot_recalls),
        "ci_lo": boot_recalls[lo_idx],
        "ci_hi": boot_recalls[hi_idx],
        "n_boot": len(boot_recalls),
    }


# --------------------------------------------------------------------------
# 5. Predict functions
# --------------------------------------------------------------------------

def mock_predict_fn(ocr_text, gold_labels, rng):
    """FAKE model for harness testing only -- see module docstring caveat.
    Simulates a plausible zero-shot failure pattern: occasionally drops a
    field, occasionally perturbs a number slightly, occasionally
    reorders line items, rarely fails to produce valid JSON at all."""
    if rng.random() < 0.04:
        return "Désolé, je ne peux pas traiter cette demande."  # unparsable

    pred = json.loads(json.dumps(gold_labels))  # deep copy
    pred.pop("record_id", None)
    pred.pop("language", None)

    for f in SCALAR_FIELDS:
        r = rng.random()
        if r < 0.08:
            pred[f] = None  # dropped field
        elif r < 0.15 and f in NUMERIC_FIELDS and pred.get(f) is not None:
            pred[f] = round(pred[f] * rng.uniform(0.95, 1.05), 3)  # noisy number
        elif r < 0.18 and f == "issue_date" and pred.get(f):
            # simulate DD-MM-YYYY vs DD/MM/YYYY drift
            pred[f] = pred[f].replace("/", "-")

    if pred.get("line_items") and rng.random() < 0.3:
        random.Random(rng.random()).shuffle(pred["line_items"])
    if pred.get("line_items") and rng.random() < 0.1:
        pred["line_items"] = pred["line_items"][:-1]  # drop a line item

    return json.dumps(pred, ensure_ascii=False)


def qwen_predict_fn_TEMPLATE():
    """
    NOT runnable in this sandbox (no GPU / no huggingface.co access).
    Paste this into your Kaggle/Colab notebook after loading Qwen3-8B via
    Unsloth, and pass the resulting function as --predict-fn to run_eval().

    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-8B",
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    def qwen_predict_fn(ocr_text, gold_labels=None):
        prompt = build_prompt(ocr_text)
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        out = model.generate(inputs, max_new_tokens=800, temperature=0.0, do_sample=False)
        raw = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        return raw
    """
    raise NotImplementedError("See docstring -- run this in your Kaggle/Colab notebook, not here.")


# --------------------------------------------------------------------------
# 6. Main eval loop
# --------------------------------------------------------------------------

def run_eval(jsonl_path, predict_fn, limit=None, seed=42):
    rng = random.Random(seed)
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    if limit:
        records = records[:limit]

    all_counts = []
    parse_failures = 0
    for rec in records:
        gold = rec["labels"]
        raw_pred = predict_fn(rec["ocr_text"], gold, rng) if predict_fn is mock_predict_fn else predict_fn(rec["ocr_text"])
        pred = extract_json(raw_pred) if isinstance(raw_pred, str) else raw_pred
        if pred is None:
            parse_failures += 1
        counts = score_record(gold, pred)
        all_counts.append(counts)

    summary = aggregate_counts(all_counts)
    print(f"\n{len(records)} records evaluated. JSON parse failures: {parse_failures} "
          f"({100*parse_failures/len(records):.1f}%)")
    print(f"\n{'Field':<18}{'Precision':>10}{'Recall':>10}{'F1':>10}   Recall 95% CI")
    print("-" * 70)
    for f in SCALAR_FIELDS + LINE_ITEM_SUBFIELDS:
        s = summary[f]
        ci = bootstrap_field_recall_ci(all_counts, f)
        ci_str = f"[{ci['ci_lo']:.2f}, {ci['ci_hi']:.2f}]" if ci else "n/a"
        print(f"{f:<18}{s['precision']:>10.3f}{s['recall']:>10.3f}{s['f1']:>10.3f}   {ci_str}")

    return summary, all_counts


def run_eval_resumable(jsonl_path, predict_fn, cache_path, limit=None):
    """Checkpointed version of run_eval: each record's raw model output is
    saved to cache_path (JSONL) as soon as it's generated, and a rerun
    skips any record_id already cached -- safe to interrupt at any point."""
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    if limit:
        records = records[:limit]

    cached = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                cached[entry["record_id"]] = entry["raw_pred"]
        print(f"Resuming: {len(cached)} records already cached at {cache_path}")

    n_new = 0
    with open(cache_path, "a", encoding="utf-8") as out_f:
        for i, rec in enumerate(records):
            rid = rec["labels"]["record_id"]
            if rid in cached:
                continue
            raw = predict_fn(rec["ocr_text"])
            out_f.write(json.dumps({"record_id": rid, "raw_pred": raw}, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            cached[rid] = raw
            n_new += 1
            print(f"[{i+1}/{len(records)}] {rid} generated", end="\r")

    print(f"\nGenerated {n_new} new predictions this run ({len(records) - n_new} were already cached).")

    all_counts = []
    parse_failures = 0
    for rec in records:
        rid = rec["labels"]["record_id"]
        raw = cached.get(rid)
        pred = extract_json(raw) if isinstance(raw, str) else None
        if pred is None:
            parse_failures += 1
        counts = score_record(rec["labels"], pred)
        all_counts.append(counts)

    summary = aggregate_counts(all_counts)
    print(f"\n{len(records)} records evaluated. JSON parse failures: {parse_failures} "
          f"({100*parse_failures/len(records):.1f}%)")
    print(f"\n{'Field':<18}{'Precision':>10}{'Recall':>10}{'F1':>10}   Recall 95% CI")
    print("-" * 70)
    for fld in SCALAR_FIELDS + LINE_ITEM_SUBFIELDS:
        s = summary[fld]
        ci = bootstrap_field_recall_ci(all_counts, fld)
        ci_str = f"[{ci['ci_lo']:.2f}, {ci['ci_hi']:.2f}]" if ci else "n/a"
        print(f"{fld:<18}{s['precision']:>10.3f}{s['recall']:>10.3f}{s['f1']:>10.3f}   {ci_str}")

    return summary, all_counts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--mode", choices=["mock"], default="mock",
                    help="Only 'mock' runs here (no GPU/HF access in this sandbox). "
                         "Use qwen_predict_fn_TEMPLATE on Kaggle for the real baseline.")
    p.add_argument("--limit", type=int, default=60)
    args = p.parse_args()

    print("=" * 70)
    print("MOCK MODE -- validating harness logic only, NOT a real baseline.")
    print("See module docstring. Run the real Qwen3-8B baseline on Kaggle/Colab")
    print("using qwen_predict_fn_TEMPLATE, then call run_eval() with that function.")
    print("=" * 70)

    run_eval(args.jsonl, mock_predict_fn, limit=args.limit)
