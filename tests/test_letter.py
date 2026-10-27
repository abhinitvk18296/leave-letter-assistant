import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from letter_generator import CATEGORY_LABELS, classify_reason, draft_letter, load_model  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.mark.parametrize("text,expected", [
    ("I have a high fever and cough", "illness"),
    ("I have a dentist appointment tomorrow", "medical_appointment"),
    ("My cousin's wedding is this weekend", "family_event"),
    ("My grandfather passed away", "bereavement"),
])
def test_classifier(model, text, expected):
    label, _ = classify_reason(model, text)
    assert label == expected


def test_letter_has_blank_signatures():
    letter = draft_letter("illness", "Asha", "10-B", "Class Teacher", date(2026, 10, 27), date(2026, 10, 28))
    assert "Parent/Guardian's signature: ____" in letter
    assert "2 days" in letter


def test_all_categories_have_templates():
    for cat in CATEGORY_LABELS:
        draft_letter(cat, "A", "B", "C", date.today(), date.today(), tone="formal")
        draft_letter(cat, "A", "B", "C", date.today(), date.today(), tone="polite_casual")


def test_bad_dates():
    with pytest.raises(ValueError):
        draft_letter("illness", "A", "B", "C", date(2026, 10, 28), date(2026, 10, 27))
