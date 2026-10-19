"""Clinical-report de-identification of label-leaking fields.

Implements the protocol described in Sec. 3.2 (Clinical Report Encoding):
before tokenization, every report is stripped of all assessment and
impression fields (BI-RADS category, management recommendation, diagnostic
conclusion); only descriptive findings are retained. Studies without a
report are processed with an empty text sequence.
"""
import re

LEAK_HEADERS = [
    r"assessment", r"impression", r"bi-?rads", r"category\s*:",
    r"recommendation", r"management", r"conclusion", r"diagnosis",
]
LEAK_RE = re.compile(
    r"^\s*(?:" + "|".join(LEAK_HEADERS) + r")[^\n]*(?:\n|$)",
    re.IGNORECASE | re.MULTILINE,
)
BIRADS_INLINE_RE = re.compile(r"bi-?rads\s*(category\s*)?[0-6]", re.IGNORECASE)


def strip_leaking_fields(report_text: str) -> str:
    """Remove assessment/impression/BI-RADS lines and inline BI-RADS mentions."""
    cleaned = LEAK_RE.sub("", report_text)
    cleaned = BIRADS_INLINE_RE.sub("[removed]", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def load_study_text(report_path):
    """Return cleaned report text, or '' when the study has no report."""
    if report_path is None:
        return ""
    with open(report_path, "r", encoding="utf-8") as f:
        return strip_leaking_fields(f.read())
