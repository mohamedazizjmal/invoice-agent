"""
validation.py -- rule-based validation / confidence-routing layer for the
invoice extraction agent.

Designed directly from the end-to-end evaluation findings: invoice_number
and vendor_tax_id are the fields real OCR damages most (exact-match
alphanumeric codes are fragile to single-character OCR confusions like
O/0, 1/l, S/5), so those get strict format checks. Monetary fields are
checked via arithmetic consistency instead -- a stronger, assumption-free
signal than any format rule, and one the fine-tuned model rarely gets
wrong on its own (see eval results), so a mismatch here usually points at
an upstream OCR/extraction error worth a human looking at.

No ML in this file by design. This is the deterministic, auditable layer
between the model's raw JSON output and a human reviewer -- every flag it
raises should be explainable to a non-technical reviewer without any
reference to the model at all.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------
# Format validators
# --------------------------------------------------------------------------

# Matches this project's synthetic invoice-number format: FAC-YYYY-NNNN.
# Real-world invoice numbers vary a lot between vendors -- once labeling
# real invoices (Label Studio step), loosen or replace this with whatever
# patterns actually show up rather than assuming this one generalizes.
INVOICE_NUMBER_RE = re.compile(r"^[A-Z]{2,5}-\d{4}-\d{3,6}$")

# Matches this project's synthetic Tunisian Matricule Fiscal format:
# 7 digits + check letter + "/" + category + "/" + type + "/" + 3-digit
# establishment code, e.g. "8377701J/P/M/000". Real MF numbers follow this
# general shape but with more category-letter variants in practice --
# validate against real examples once available rather than trusting this
# blindly.
TAX_ID_RE = re.compile(r"^\d{7}[A-Z]/[A-Z]/[A-Z]/\d{3}$")

DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")


def validate_invoice_number(value):
    if not value:
        return False, "invoice_number is missing"
    if not INVOICE_NUMBER_RE.match(value):
        return False, f"invoice_number '{value}' doesn't match the expected format (e.g. FAC-2025-1234)"
    return True, None


def validate_tax_id(value):
    if not value:
        return False, "vendor_tax_id is missing"
    if not TAX_ID_RE.match(value):
        return False, f"vendor_tax_id '{value}' doesn't match the expected Matricule Fiscal format"
    return True, None


def validate_date(value):
    if not value:
        return False, "issue_date is missing"
    if not DATE_RE.match(value):
        return False, f"issue_date '{value}' doesn't match DD/MM/YYYY"
    return True, None


# --------------------------------------------------------------------------
# Arithmetic consistency checks -- free signal, no format assumptions needed
# --------------------------------------------------------------------------

def _is_close(a, b, rel_tol=0.01, abs_tol=0.05):
    if a is None or b is None:
        return False
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def check_vat_consistency(extracted):
    issues = []
    subtotal = extracted.get("subtotal")
    vat_rate = extracted.get("vat_rate")
    vat_amount = extracted.get("vat_amount")
    grand_total = extracted.get("grand_total")

    if subtotal is not None and vat_rate is not None and vat_amount is not None:
        try:
            expected_vat = float(subtotal) * float(vat_rate)
            if not _is_close(expected_vat, vat_amount):
                issues.append(
                    f"vat_amount ({vat_amount}) doesn't match subtotal*vat_rate "
                    f"({subtotal}*{vat_rate}={expected_vat:.3f})"
                )
        except (TypeError, ValueError):
            pass

    if subtotal is not None and vat_amount is not None and grand_total is not None:
        try:
            expected_total = float(subtotal) + float(vat_amount)
            if not _is_close(expected_total, grand_total):
                issues.append(
                    f"grand_total ({grand_total}) doesn't match subtotal+vat_amount "
                    f"({subtotal}+{vat_amount}={expected_total:.3f})"
                )
        except (TypeError, ValueError):
            pass
    return issues


def check_line_items_consistency(extracted):
    issues = []
    items = extracted.get("line_items") or []
    if not isinstance(items, list):
        return issues

    computed_subtotal = 0.0
    any_items_valid = False

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        qty, unit_price, line_total = item.get("quantity"), item.get("unit_price"), item.get("line_total")
        if qty is not None and unit_price is not None and line_total is not None:
            try:
                expected_line_total = float(qty) * float(unit_price)
                if not _is_close(expected_line_total, line_total):
                    issues.append(
                        f"line_items[{i}]: line_total ({line_total}) doesn't match "
                        f"quantity*unit_price ({qty}*{unit_price}={expected_line_total:.3f})"
                    )
                computed_subtotal += float(line_total)
                any_items_valid = True
            except (TypeError, ValueError):
                pass

    if any_items_valid and extracted.get("subtotal") is not None:
        if not _is_close(computed_subtotal, extracted["subtotal"]):
            issues.append(
                f"sum of line_items' line_total ({computed_subtotal:.3f}) doesn't match "
                f"subtotal ({extracted['subtotal']})"
            )
    return issues


# --------------------------------------------------------------------------
# Duplicate detection
# --------------------------------------------------------------------------

def make_dedup_key(extracted):
    """Key used to detect likely-duplicate invoices. Vendor + invoice
    number is the strongest signal; vendor + date + total is a softer
    fallback for when invoice_number itself was OCR-garbled -- which we
    now know is this system's single most fragile field, so relying on
    it alone for dedup would miss exactly the cases most likely to need
    a fallback."""
    vendor = (extracted.get("vendor_tax_id") or extracted.get("vendor_name") or "")
    vendor = vendor.strip().lower() if isinstance(vendor, str) else ""
    inv_num = extracted.get("invoice_number")
    inv_num = inv_num.strip().lower() if isinstance(inv_num, str) else ""
    if vendor and inv_num:
        return ("strict", vendor, inv_num)
    date = extracted.get("issue_date")
    total = extracted.get("grand_total")
    if vendor and date and total is not None:
        try:
            return ("soft", vendor, date, round(float(total), 2))
        except (TypeError, ValueError):
            pass
    return None


def check_duplicate(extracted, seen_keys):
    key = make_dedup_key(extracted)
    if key is None:
        return [], key
    if key in seen_keys:
        kind = "exact (vendor + invoice number)" if key[0] == "strict" else "likely (vendor + date + total)"
        return [f"possible duplicate invoice -- {kind} match against a previously processed invoice"], key
    return [], key


# --------------------------------------------------------------------------
# Missing-field check
# --------------------------------------------------------------------------

REQUIRED_FIELDS = ["invoice_number", "issue_date", "vendor_name", "vendor_tax_id",
                    "subtotal", "vat_rate", "vat_amount", "grand_total", "currency"]


def check_missing_fields(extracted):
    return [f"{f} is missing" for f in REQUIRED_FIELDS if extracted.get(f) is None]


# --------------------------------------------------------------------------
# Top-level validation + confidence routing
# --------------------------------------------------------------------------

@dataclass
class ValidationResult:
    issues: list = field(default_factory=list)
    high_priority_issues: list = field(default_factory=list)
    needs_review: bool = False
    dedup_key: Optional[tuple] = None

    def to_dict(self):
        return {
            "needs_review": self.needs_review,
            "high_priority_issues": self.high_priority_issues,
            "issues": self.issues,
        }


def validate_extraction(extracted, seen_keys=None):
    """
    extracted: the model's extracted JSON dict for one invoice (already
      parsed -- pass None-safe dicts, not raw model text).
    seen_keys: a set of dedup keys accumulated across a batch run -- pass
      the same set across multiple calls to detect duplicates within a
      processing run; mutated in place.

    Routing policy, built directly from the end-to-end eval findings:
    - Format failures on invoice_number/vendor_tax_id -> HIGH priority.
      These are exactly the fields OCR damages most (-36 points recall
      in the end-to-end eval), so a format failure here is the single
      strongest signal something needs a human look.
    - Arithmetic inconsistency -> HIGH priority. The fine-tuned model
      rarely gets this wrong on its own, so a mismatch usually means an
      upstream OCR or extraction error in the numbers.
    - Duplicate match -> HIGH priority (fraud / double-payment risk).
    - Missing fields, malformed date -> lower priority (worth a glance,
      less urgent, and much more common as an honest "field not found"
      than as a sign of a wrong extraction).
    """
    if extracted is None:
        result = ValidationResult()
        result.high_priority_issues.append("extraction failed entirely -- no valid JSON was produced")
        result.needs_review = True
        return result

    result = ValidationResult()

    ok, msg = validate_invoice_number(extracted.get("invoice_number"))
    if not ok:
        result.high_priority_issues.append(msg)

    ok, msg = validate_tax_id(extracted.get("vendor_tax_id"))
    if not ok:
        result.high_priority_issues.append(msg)

    ok, msg = validate_date(extracted.get("issue_date"))
    if not ok:
        result.issues.append(msg)

    result.high_priority_issues.extend(check_vat_consistency(extracted))
    result.high_priority_issues.extend(check_line_items_consistency(extracted))
    result.issues.extend(check_missing_fields(extracted))

    if seen_keys is not None:
        dup_issues, key = check_duplicate(extracted, seen_keys)
        result.high_priority_issues.extend(dup_issues)
        result.dedup_key = key
        if key is not None:
            seen_keys.add(key)

    result.needs_review = len(result.high_priority_issues) > 0
    return result
