"""Generate the fictional 'Meridian Labs' sample corpus as real text PDFs.

Each document is defined page-by-page and rendered with an explicit ``add_page()``
per page (auto page break off), so the fact -> page mapping is deterministic —
that mapping is the ground truth for eval/golden.yaml citations.

Run:  python scripts/make_sample_pdfs.py
Self-verifies with pypdf that every key fact lands on its expected page.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from pypdf import PdfReader

OUT_DIR = Path(__file__).resolve().parent.parent / "sample_pdfs"

# ────────────────────────────── corpus content ──────────────────────────────
DOCS: dict[str, dict] = {
    "meridian_employee_handbook.pdf": {
        "title": "Meridian Labs - Employee Handbook",
        "pages": [
            {
                "heading": "1. Working Hours & Time Off",
                "items": [
                    "Meridian Labs observes core working hours of 10:00 AM to 4:00 PM local time; "
                    "employees may arrange the remaining hours flexibly around them.",
                    "Full-time employees accrue 20 days of paid vacation per calendar year, in "
                    "addition to 10 company-recognized public holidays.",
                    "Unused vacation of up to 5 days may be carried over into the following year; "
                    "any amount beyond 5 days is forfeited.",
                    "New employees serve a 90-day probationary period before becoming eligible "
                    "for benefits.",
                ],
            },
            {
                "heading": "2. Remote Work & Conduct",
                "items": [
                    "Employees may work remotely up to 3 days per week; fully remote arrangements "
                    "require written approval from a director.",
                    "The company observes a quiet-hours policy: no internal meetings should be "
                    "scheduled before 9:00 AM or after 5:00 PM.",
                    "All employees must complete the annual Code of Conduct acknowledgment by "
                    "January 31 each year.",
                    "The dress code is smart-casual; client-facing meetings require business attire.",
                ],
            },
        ],
    },
    "meridian_it_security_policy.pdf": {
        "title": "Meridian Labs - IT Security Policy",
        "pages": [
            {
                "heading": "1. Access & Authentication",
                "items": [
                    "All account passwords must be at least 14 characters long and include a mix "
                    "of letters, numbers, and symbols.",
                    "Multi-factor authentication (MFA) is mandatory for every internal and cloud "
                    "system, with no exceptions.",
                    "Passwords must be rotated every 180 days; reuse of any of the previous 5 "
                    "passwords is prohibited.",
                    "Workstations must automatically lock after 10 minutes of inactivity.",
                ],
            },
            {
                "heading": "2. Data Classification & Incident Response",
                "items": [
                    "Company data is classified into four tiers: Public, Internal, Confidential, "
                    "and Restricted.",
                    "Restricted data, such as customer financial records, must never be copied to "
                    "personal devices or removable media.",
                    "Suspected security incidents must be reported to the security team within "
                    "1 hour of discovery via the incident portal.",
                    "All laptops issued to staff are protected with full-disk encryption by default.",
                ],
            },
        ],
    },
    "meridian_expense_travel_policy.pdf": {
        "title": "Meridian Labs - Expense & Travel Policy",
        "pages": [
            {
                "heading": "1. Reimbursement Rules",
                "items": [
                    "Employees are reimbursed for business meals up to a limit of $75 per day "
                    "for domestic travel.",
                    "Expense reports must be submitted within 30 days of the expense being "
                    "incurred; late reports may be denied.",
                    "Any single expense greater than $1,000 requires prior written approval from "
                    "a Vice President.",
                    "Personal entertainment, alcohol, and traffic fines are never reimbursable.",
                ],
            },
            {
                "heading": "2. Travel Booking",
                "items": [
                    "Flights with a scheduled duration over 6 hours may be booked in premium "
                    "economy; business class requires C-level approval.",
                    "Hotel reimbursement is capped at $250 per night in domestic cities and "
                    "$350 per night internationally.",
                    "Employees should book travel at least 14 days in advance to obtain "
                    "preferred rates.",
                    "Mileage for business use of a personal vehicle is reimbursed at $0.67 "
                    "per mile.",
                ],
            },
        ],
    },
    "meridian_benefits_guide.pdf": {
        "title": "Meridian Labs - Employee Benefits Guide",
        "pages": [
            {
                "heading": "1. Health & Retirement",
                "items": [
                    "Meridian Labs covers 100% of health insurance premiums for employees and "
                    "75% for dependents.",
                    "The company matches employee 401(k) contributions dollar-for-dollar up to "
                    "6% of salary.",
                    "Employer 401(k) matching contributions vest fully after 2 years of "
                    "continuous service.",
                    "Employees are automatically enrolled in the dental and vision plans at no cost.",
                ],
            },
            {
                "heading": "2. Leave & Wellness",
                "items": [
                    "Meridian Labs provides 16 weeks of fully paid parental leave for all new "
                    "parents, regardless of gender.",
                    "Each employee receives an annual wellness stipend of $500, usable for gym "
                    "memberships or fitness equipment.",
                    "The company offers 5 paid volunteer days per year for community service.",
                    "An Employee Assistance Program (EAP) provides free, confidential counseling "
                    "24/7.",
                ],
            },
        ],
    },
}

# fact -> (filename, expected 1-based page); used for self-verification below and
# mirrored by eval/golden.yaml as citation ground truth.
PROBES: list[tuple[str, str, int]] = [
    ("20 days of paid vacation", "meridian_employee_handbook.pdf", 1),
    ("up to 3 days per week", "meridian_employee_handbook.pdf", 2),
    ("at least 14 characters", "meridian_it_security_policy.pdf", 1),
    ("within 1 hour", "meridian_it_security_policy.pdf", 2),
    ("$75 per day", "meridian_expense_travel_policy.pdf", 1),
    ("premium economy", "meridian_expense_travel_policy.pdf", 2),
    ("6% of salary", "meridian_benefits_guide.pdf", 1),
    ("16 weeks", "meridian_benefits_guide.pdf", 2),
]


# ─────────────────────────────── rendering ──────────────────────────────────
class _DocPDF(FPDF):
    def __init__(self, title: str):
        super().__init__(format="letter")
        self.doc_title = title
        self.set_auto_page_break(False)  # page layout is explicit and deterministic

    def header(self):  # rendered on every page; extracted text includes it (realistic)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(130)
        self.cell(0, 6, self.doc_title, align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(130)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def render_doc(filename: str, spec: dict) -> Path:
    pdf = _DocPDF(spec["title"])

    def mc(h: float, text: str) -> None:
        # full-width cell that returns the cursor to the left margin, next line
        pdf.multi_cell(0, h, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    for idx, page in enumerate(spec["pages"]):
        pdf.add_page()
        pdf.set_text_color(0)
        if idx == 0:
            pdf.set_font("Helvetica", "B", 16)
            mc(9, spec["title"])
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(90)
            mc(6, "Policy effective January 1, 2026")
            pdf.ln(4)
            pdf.set_text_color(0)
        pdf.set_font("Helvetica", "B", 13)
        mc(8, page["heading"])
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 11)
        for item in page["items"]:
            mc(7, f"- {item}")
            pdf.ln(1)
    out = OUT_DIR / filename
    pdf.output(str(out))
    return out


# ────────────────────────────── verification ────────────────────────────────
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def verify() -> bool:
    ok = True
    for filename, spec in DOCS.items():
        reader = PdfReader(OUT_DIR / filename)
        n_expected = len(spec["pages"])
        status = "OK" if len(reader.pages) == n_expected else "FAIL"
        ok &= status == "OK"
        print(f"  {filename}: {len(reader.pages)} page(s) [{status}]")
    print("  fact -> page probes:")
    for fact, filename, page_no in PROBES:
        reader = PdfReader(OUT_DIR / filename)
        text = _norm(reader.pages[page_no - 1].extract_text() or "")
        hit = fact in text
        ok &= hit
        print(f"    [{'OK' if hit else 'FAIL'}] {fact!r} on {filename} p.{page_no}")
    return ok


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, spec in DOCS.items():
        path = render_doc(filename, spec)
        print(f"wrote {path.name} ({path.stat().st_size} bytes)")
    print("verifying with pypdf:")
    sys.exit(0 if verify() else 1)
