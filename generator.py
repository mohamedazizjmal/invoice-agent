"""
Synthetic French / Tunisian-Arabic invoice generator.

Produces paired samples of:
  - "ocr_text": a plain-text rendering of an invoice, meant to *simulate*
    what an OCR engine would output (line breaks, spacing quirks, no rich
    layout) — this is what a real OCR pipeline would hand to the extraction
    model.
  - "labels": the ground-truth structured JSON for that invoice.

Design goals (per project brief):
  - French and Tunisian-Arabic examples are generated from the SAME pools
    of vendors/products where possible, and mixed documents (French
    boilerplate + Arabic product lines, a very common real-world Tunisian
    pattern) are supported explicitly — nothing is siloed by language.
  - Multiple layout templates per language to create structural variety,
    since open English invoice datasets already give you structure
    diversity but not language coverage — this generator's job is the
    reverse: language/format coverage for FR + TN-Arabic specifically.
  - Realistic Tunisian specifics: TND currency, Matricule Fiscal (MF) tax
    ID format, real Tunisian VAT rates (19%, 13%, 7%, 0%).
"""

import argparse
import json
import random
import re
from datetime import date, timedelta

# --------------------------------------------------------------------------
# Data pools
# --------------------------------------------------------------------------

FR_COMPANY_NAMES = [
    "Société Générale de Distribution", "Atelier Mécanique El Amen",
    "Tunisie Fournitures Bureau", "Groupe Sahel Industrie",
    "Ets Ben Salah & Fils", "Nord Africa Logistique",
    "Compagnie Tunisienne d'Électricité", "Menuiserie Moderne du Cap Bon",
    "Import Export Zitouna", "Cabinet Conseil Carthage",
]

AR_COMPANY_NAMES = [
    "شركة النور للتوزيع", "مؤسسة الأمل للتجارة",
    "الشركة التونسية للمواد الغذائية", "مصنع الزيتون الذهبي",
    "مؤسسة بن علي وأبنائه", "شركة الوفاء للنقل",
    "مكتب الدراسات الهندسية الحديث", "شركة الرابطة للبناء",
    "مؤسسة السلام للإلكترونيات", "الشركة الجهوية للتجهيزات",
]

CITIES = [
    ("Tunis", "1000"), ("Ariana", "2080"), ("Sfax", "3000"),
    ("Sousse", "4000"), ("Nabeul", "8000"), ("Bizerte", "7000"),
    ("Gabès", "6000"), ("Monastir", "5000"),
]
CITIES_AR = [
    ("تونس", "1000"), ("أريانة", "2080"), ("صفاقس", "3000"),
    ("سوسة", "4000"), ("نابل", "8000"), ("بنزرت", "7000"),
]

FR_PRODUCTS = [
    "Ramette papier A4 80g", "Cartouche encre HP 305", "Chaise de bureau ergonomique",
    "Câble réseau CAT6 5m", "Prestation de maintenance informatique",
    "Service de transport de marchandises", "Panneau isolant 10mm",
    "Licence logicielle annuelle", "Prestation de conseil (jour/homme)",
    "Climatiseur split 12000 BTU", "Ordinateur portable 15 pouces",
    "Pack de vis inox 6x40",
]
AR_PRODUCTS = [
    "علبة ورق A4", "حبر طابعة HP 305", "كرسي مكتب",
    "كابل شبكة 5 أمتار", "خدمة صيانة معلوماتية",
    "خدمة نقل بضائع", "لوح عازل 10 ملم",
    "رخصة برمجية سنوية", "استشارة (يوم/شخص)",
    "مكيف هواء 12000 وحدة", "حاسوب محمول 15 بوصة",
    "علبة براغي ستانلس 6x40",
]

VAT_RATES = [0.19, 0.13, 0.07, 0.0]
VAT_WEIGHTS = [0.55, 0.20, 0.20, 0.05]  # 19% is the standard/most common rate

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _rand_date(start_year=2023, end_year=2026):
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def _fmt_date_fr(d):
    return d.strftime("%d/%m/%Y")


def _fmt_date_ar(d):
    # Tunisian Arabic invoices commonly still use Gregorian dates,
    # sometimes with Arabic-Indic digits — we keep Western digits here
    # since that's what's most common in real Tunisian business docs.
    return d.strftime("%d/%m/%Y")


def _rand_mf():
    """Generate a plausible Tunisian Matricule Fiscal (tax ID)."""
    digits = "".join(str(random.randint(0, 9)) for _ in range(7))
    letter = random.choice("ABCDEFGHIJKLMNPQRSTVWXYZ")
    suffix = random.choice(["A/M/000", "P/M/000", "N/A/000", "A/B/000"])
    return f"{digits}{letter}/{suffix}"


def _rand_invoice_number(year):
    return f"FAC-{year}-{random.randint(1000, 9999)}"


def _money(x):
    return f"{x:,.3f}".replace(",", " ")  # TND is quoted to 3 decimals


def _reshape_note():
    # Placeholder kept for future integration with arabic-reshaper +
    # python-bidi if/when this generator is extended to render actual
    # PDF/image invoices for the OCR pipeline (see OCR pipeline step).
    pass


# --------------------------------------------------------------------------
# Core generation
# --------------------------------------------------------------------------

def _gen_line_items(lang, n_items):
    pool = AR_PRODUCTS if lang == "ar" else FR_PRODUCTS
    items = []
    for _ in range(n_items):
        desc = random.choice(pool)
        qty = random.randint(1, 20)
        unit_price = round(random.uniform(5, 800), 3)
        line_total = round(qty * unit_price, 3)
        items.append({
            "description": desc,
            "quantity": qty,
            "unit_price": unit_price,
            "line_total": line_total,
        })
    return items


def generate_record(record_id, lang):
    """
    lang: "fr" | "ar" | "mixed"
      - "fr": French vendor, French products, French boilerplate
      - "ar": Arabic vendor, Arabic products, Arabic boilerplate
      - "mixed": French boilerplate/labels (very common in Tunisia: legal
        text and field labels in French even when product lines are in
        Arabic) with Arabic vendor + Arabic product descriptions
    """
    is_ar_vendor = lang in ("ar", "mixed")
    vendor = random.choice(AR_COMPANY_NAMES if is_ar_vendor else FR_COMPANY_NAMES)
    city, postal = random.choice(CITIES_AR if is_ar_vendor else CITIES)

    invoice_date = _rand_date()
    year = invoice_date.year
    invoice_number = _rand_invoice_number(year)
    mf = _rand_mf()

    items_lang = "ar" if lang in ("ar", "mixed") else "fr"
    n_items = random.randint(1, 5)
    items = _gen_line_items(items_lang, n_items)

    subtotal = round(sum(i["line_total"] for i in items), 3)
    vat_rate = random.choices(VAT_RATES, weights=VAT_WEIGHTS, k=1)[0]
    vat_amount = round(subtotal * vat_rate, 3)
    grand_total = round(subtotal + vat_amount, 3)

    labels = {
        "record_id": record_id,
        "language": lang,
        "invoice_number": invoice_number,
        "issue_date": _fmt_date_fr(invoice_date),
        "vendor_name": vendor,
        "vendor_tax_id": mf,
        "line_items": items,
        "subtotal": subtotal,
        "vat_rate": vat_rate,
        "vat_amount": vat_amount,
        "grand_total": grand_total,
        "currency": "TND",
    }

    ocr_text = render_ocr_text(labels, lang, city, postal)
    return {"ocr_text": ocr_text, "labels": labels}


# --------------------------------------------------------------------------
# OCR-text rendering (layout templates)
# --------------------------------------------------------------------------

FR_LABELS = {
    "invoice": "FACTURE", "number": "N° Facture", "date": "Date",
    "vendor": "Fournisseur", "tax_id": "Matricule Fiscal",
    "description": "Désignation", "qty": "Qté", "unit_price": "Prix Unit.",
    "total": "Total", "subtotal": "Sous-total HT", "vat": "TVA",
    "grand_total": "Total TTC", "address": "Adresse",
}
AR_LABELS = {
    "invoice": "فاتورة", "number": "رقم الفاتورة", "date": "التاريخ",
    "vendor": "المورد", "tax_id": "المعرف الجبائي",
    "description": "البيان", "qty": "الكمية", "unit_price": "سعر الوحدة",
    "total": "المجموع", "subtotal": "المجموع دون احتساب الأداء", "vat": "الأداء",
    "grand_total": "المجموع الجملي", "address": "العنوان",
}


def _template_fr_a(f, city, postal):
    lines = []
    lines.append(f"{f['vendor_name']}")
    lines.append(f"{city} {postal}, Tunisie")
    lines.append(f"{FR_LABELS['tax_id']}: {f['vendor_tax_id']}")
    lines.append("-" * 40)
    lines.append(f"{FR_LABELS['invoice']} N° {f['invoice_number']}")
    lines.append(f"{FR_LABELS['date']}: {f['issue_date']}")
    lines.append("")
    lines.append(f"{FR_LABELS['description']:<28}{FR_LABELS['qty']:>5}{FR_LABELS['unit_price']:>12}{FR_LABELS['total']:>12}")
    for it in f["line_items"]:
        lines.append(f"{it['description'][:28]:<28}{it['quantity']:>5}{_money(it['unit_price']):>12}{_money(it['line_total']):>12}")
    lines.append("-" * 40)
    lines.append(f"{FR_LABELS['subtotal']}: {_money(f['subtotal'])} TND")
    lines.append(f"{FR_LABELS['vat']} ({int(f['vat_rate']*100)}%): {_money(f['vat_amount'])} TND")
    lines.append(f"{FR_LABELS['grand_total']}: {_money(f['grand_total'])} TND")
    return "\n".join(lines)


def _template_fr_b(f, city, postal):
    # A denser, less tabular layout — simulates a second real-world format
    lines = []
    lines.append(f"=== {FR_LABELS['invoice']} ===")
    lines.append(f"{f['vendor_name']} - {FR_LABELS['tax_id']} {f['vendor_tax_id']}")
    lines.append(f"{FR_LABELS['address']}: {city}, {postal}")
    lines.append(f"{FR_LABELS['number']}: {f['invoice_number']}    {FR_LABELS['date']}: {f['issue_date']}")
    lines.append("")
    for it in f["line_items"]:
        lines.append(f"* {it['description']} x{it['quantity']} @ {_money(it['unit_price'])} TND = {_money(it['line_total'])} TND")
    lines.append("")
    lines.append(f"{FR_LABELS['subtotal']} : {_money(f['subtotal'])}")
    lines.append(f"{FR_LABELS['vat']} {int(f['vat_rate']*100)}% : {_money(f['vat_amount'])}")
    lines.append(f"{FR_LABELS['grand_total']} : {_money(f['grand_total'])} TND")
    return "\n".join(lines)


def _template_ar_a(f, city, postal):
    lines = []
    lines.append(f"{f['vendor_name']}")
    lines.append(f"{city} {postal} - تونس")
    lines.append(f"{AR_LABELS['tax_id']}: {f['vendor_tax_id']}")
    lines.append("-" * 40)
    lines.append(f"{AR_LABELS['invoice']} {AR_LABELS['number']}: {f['invoice_number']}")
    lines.append(f"{AR_LABELS['date']}: {f['issue_date']}")
    lines.append("")
    lines.append(f"{AR_LABELS['description']} | {AR_LABELS['qty']} | {AR_LABELS['unit_price']} | {AR_LABELS['total']}")
    for it in f["line_items"]:
        lines.append(f"{it['description']} | {it['quantity']} | {_money(it['unit_price'])} | {_money(it['line_total'])}")
    lines.append("-" * 40)
    lines.append(f"{AR_LABELS['subtotal']}: {_money(f['subtotal'])} د.ت")
    lines.append(f"{AR_LABELS['vat']} ({int(f['vat_rate']*100)}%): {_money(f['vat_amount'])} د.ت")
    lines.append(f"{AR_LABELS['grand_total']}: {_money(f['grand_total'])} د.ت")
    return "\n".join(lines)


def _template_mixed_a(f, city, postal):
    # French field labels / boilerplate, Arabic vendor + product lines —
    # a very common real pattern in Tunisian small-business invoicing.
    lines = []
    lines.append(f"{f['vendor_name']}")
    lines.append(f"{city} {postal}, Tunisie")
    lines.append(f"{FR_LABELS['tax_id']}: {f['vendor_tax_id']}")
    lines.append("-" * 40)
    lines.append(f"{FR_LABELS['invoice']} N° {f['invoice_number']}")
    lines.append(f"{FR_LABELS['date']}: {f['issue_date']}")
    lines.append("")
    lines.append(f"{FR_LABELS['description']} / {AR_LABELS['description']}")
    for it in f["line_items"]:
        lines.append(f"{it['description']}  x{it['quantity']}  {_money(it['unit_price'])}  {_money(it['line_total'])}")
    lines.append("-" * 40)
    lines.append(f"{FR_LABELS['subtotal']}: {_money(f['subtotal'])} TND")
    lines.append(f"{FR_LABELS['vat']} ({int(f['vat_rate']*100)}%): {_money(f['vat_amount'])} TND")
    lines.append(f"{FR_LABELS['grand_total']}: {_money(f['grand_total'])} TND")
    return "\n".join(lines)


TEMPLATES = {
    "fr": [_template_fr_a, _template_fr_b],
    "ar": [_template_ar_a],
    "mixed": [_template_mixed_a],
}


def inject_ocr_noise(text, noise_level=0.0):
    """
    Optionally corrupt text slightly to simulate realistic OCR errors
    (character substitution/drops). noise_level=0.0 means no corruption
    (use this for building the clean baseline dataset). Turn this on later
    to stress-test the model against noisier real-world OCR output.
    """
    if noise_level <= 0:
        return text
    chars = list(text)
    n_edits = int(len(chars) * noise_level)
    confusable = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5",
                  "B": "8", "8": "B"}
    for _ in range(n_edits):
        idx = random.randint(0, len(chars) - 1)
        c = chars[idx]
        if c in confusable and random.random() < 0.7:
            chars[idx] = confusable[c]
        elif c == " " and random.random() < 0.1:
            chars[idx] = ""
    return "".join(chars)


def render_ocr_text(labels, lang, city, postal):
    template_fn = random.choice(TEMPLATES[lang])
    return template_fn(labels, city, postal)


# --------------------------------------------------------------------------
# Dataset assembly + CLI
# --------------------------------------------------------------------------

def build_dataset(n, fr_ratio, ar_ratio, mixed_ratio, noise_level, seed):
    random.seed(seed)
    total = fr_ratio + ar_ratio + mixed_ratio
    fr_ratio, ar_ratio, mixed_ratio = (fr_ratio / total, ar_ratio / total, mixed_ratio / total)

    langs = random.choices(
        ["fr", "ar", "mixed"], weights=[fr_ratio, ar_ratio, mixed_ratio], k=n
    )
    records = []
    for i, lang in enumerate(langs):
        rec = generate_record(record_id=f"inv_{i:05d}", lang=lang)
        if noise_level > 0:
            rec["ocr_text"] = inject_ocr_noise(rec["ocr_text"], noise_level)
        records.append(rec)
    return records


def split_dataset(records, train_frac=0.7, val_frac=0.15, seed=42):
    """
    Language-stratified split: shuffles within each language group
    separately so French/Arabic/mixed stay mixed across all three splits
    rather than siloed (per the project brief's data plan).
    """
    rng = random.Random(seed)
    by_lang = {"fr": [], "ar": [], "mixed": []}
    for r in records:
        by_lang[r["labels"]["language"]].append(r)

    train, val, test = [], [], []
    for lang, group in by_lang.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_stats(name, records):
    from collections import Counter
    langs = Counter(r["labels"]["language"] for r in records)
    print(f"  {name}: {len(records)} records — {dict(langs)}")


def main():
    p = argparse.ArgumentParser(description="Generate synthetic FR/Tunisian-Arabic invoice dataset")
    p.add_argument("--n", type=int, default=500, help="Total number of invoices to generate")
    p.add_argument("--out", type=str, default="dataset", help="Output directory")
    p.add_argument("--fr-ratio", type=float, default=0.4)
    p.add_argument("--ar-ratio", type=float, default=0.35)
    p.add_argument("--mixed-ratio", type=float, default=0.25)
    p.add_argument("--noise", type=float, default=0.0,
                    help="OCR noise level 0.0-1.0 (0 = clean text, for the base labeled set)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)

    records = build_dataset(
        n=args.n, fr_ratio=args.fr_ratio, ar_ratio=args.ar_ratio,
        mixed_ratio=args.mixed_ratio, noise_level=args.noise, seed=args.seed,
    )
    train, val, test = split_dataset(records, seed=args.seed)

    write_jsonl(train, os.path.join(args.out, "train.jsonl"))
    write_jsonl(val, os.path.join(args.out, "val.jsonl"))
    write_jsonl(test, os.path.join(args.out, "test.jsonl"))

    print(f"Generated {len(records)} synthetic invoices -> {args.out}/")
    print_stats("train", train)
    print_stats("val", val)
    print_stats("test", test)
    print("\nSample record:")
    print(json.dumps(records[0], ensure_ascii=False, indent=2)[:800])


if __name__ == "__main__":
    main()
