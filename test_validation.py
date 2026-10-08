import validation as v

def test(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    assert cond, name

# --------------------------------------------------------------------
# A fully valid, internally consistent extraction -> no flags at all
# --------------------------------------------------------------------
good = {
    "invoice_number": "FAC-2025-1234",
    "issue_date": "17/10/2025",
    "vendor_name": "Société Générale de Distribution",
    "vendor_tax_id": "8377701J/P/M/000",
    "line_items": [
        {"description": "Chaise de bureau", "quantity": 5, "unit_price": 100.0, "line_total": 500.0},
        {"description": "Ramette papier", "quantity": 2, "unit_price": 25.0, "line_total": 50.0},
    ],
    "subtotal": 550.0,
    "vat_rate": 0.19,
    "vat_amount": 104.5,
    "grand_total": 654.5,
    "currency": "TND",
}
r = v.validate_extraction(good)
test("valid extraction -> needs_review False", r.needs_review is False)
test("valid extraction -> no high priority issues", len(r.high_priority_issues) == 0)
test("valid extraction -> no issues at all", len(r.issues) == 0)

# --------------------------------------------------------------------
# Malformed invoice number -> high priority flag
# --------------------------------------------------------------------
bad_invnum = dict(good, invoice_number="FAC2025999")  # missing dashes
r = v.validate_extraction(bad_invnum)
test("malformed invoice_number -> needs_review True", r.needs_review is True)
test("malformed invoice_number -> flagged high priority", any("invoice_number" in i for i in r.high_priority_issues))

# --------------------------------------------------------------------
# Malformed tax ID -> high priority flag
# --------------------------------------------------------------------
bad_taxid = dict(good, vendor_tax_id="837770 1J PM000")  # OCR-mangled spacing
r = v.validate_extraction(bad_taxid)
test("malformed vendor_tax_id -> needs_review True", r.needs_review is True)
test("malformed vendor_tax_id -> flagged high priority", any("vendor_tax_id" in i for i in r.high_priority_issues))

# --------------------------------------------------------------------
# Arithmetic mismatch (grand_total wrong) -> high priority flag
# --------------------------------------------------------------------
bad_total = dict(good, grand_total=999.99)
r = v.validate_extraction(bad_total)
test("bad grand_total -> needs_review True", r.needs_review is True)
test("bad grand_total -> flagged via vat consistency", any("grand_total" in i for i in r.high_priority_issues))

# --------------------------------------------------------------------
# Line item inconsistency (qty*price != line_total) -> high priority flag
# --------------------------------------------------------------------
bad_lineitem = dict(good)
bad_lineitem["line_items"] = [
    {"description": "Chaise de bureau", "quantity": 5, "unit_price": 100.0, "line_total": 999.0},  # wrong
    {"description": "Ramette papier", "quantity": 2, "unit_price": 25.0, "line_total": 50.0},
]
r = v.validate_extraction(bad_lineitem)
test("bad line_total -> needs_review True", r.needs_review is True)
test("bad line_total -> flagged", any("line_items[0]" in i for i in r.high_priority_issues))

# --------------------------------------------------------------------
# Missing optional-ish field (currency) -> low priority only, not needs_review
# --------------------------------------------------------------------
missing_currency = dict(good)
del missing_currency["currency"]
r = v.validate_extraction(missing_currency)
test("missing currency alone -> needs_review False (low priority only)", r.needs_review is False)
test("missing currency -> appears in issues (not high priority)", any("currency" in i for i in r.issues))

# --------------------------------------------------------------------
# Total extraction failure (pred is None, e.g. JSON parse failure) -> flagged
# --------------------------------------------------------------------
r = v.validate_extraction(None)
test("None extraction -> needs_review True", r.needs_review is True)

# --------------------------------------------------------------------
# Duplicate detection across a batch
# --------------------------------------------------------------------
seen = set()
r1 = v.validate_extraction(good, seen_keys=seen)
test("first occurrence -> not flagged as duplicate", not any("duplicate" in i for i in r1.high_priority_issues))
r2 = v.validate_extraction(good, seen_keys=seen)  # same invoice again
test("second identical occurrence -> flagged as duplicate", any("duplicate" in i for i in r2.high_priority_issues))
test("duplicate -> needs_review True", r2.needs_review is True)

# Different invoice (different invoice_number) -> not a duplicate
different = dict(good, invoice_number="FAC-2025-5678")
r3 = v.validate_extraction(different, seen_keys=seen)
test("different invoice_number -> not flagged as duplicate", not any("duplicate" in i for i in r3.high_priority_issues))

print("\nAll validation.py unit tests passed.")
