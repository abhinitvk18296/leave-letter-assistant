"""Predict the category of an absence reason and draft a leave letter.

The generated letter always has BLANK signature lines. The parent/guardian (and
doctor, where relevant) must sign it themselves, and any medical certificate
must be issued by a real doctor. This tool only drafts wording.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib

MODEL_PATH = Path("models/reason_classifier.joblib")

CATEGORY_LABELS = {
    "illness": "Illness",
    "medical_appointment": "Medical appointment",
    "family_event": "Family function",
    "family_emergency": "Family emergency",
    "bereavement": "Bereavement",
    "personal_work": "Personal / official work",
    "exam_competition": "Exam / competition",
    "travel_transport": "Travel / transport disruption",
}

# Reasons where a supporting document is normally expected.
SUPPORTING_DOC = {
    "illness": "a medical certificate issued by my doctor",
    "medical_appointment": "the appointment slip or prescription from the clinic",
    "family_emergency": "any relevant hospital or official document, if available",
    "exam_competition": "the hall ticket, invitation or participation letter",
    "bereavement": None,
    "family_event": None,
    "personal_work": "the appointment letter or receipt from the office",
    "travel_transport": None,
}

BODY = {
    "formal": {
        "illness": "I am writing to inform you that I am unwell and have been advised rest. I will therefore be unable to attend classes",
        "medical_appointment": "I am writing to inform you that I have a medical appointment that I am unable to reschedule. I will therefore be unable to attend classes",
        "family_event": "I am writing to inform you that I need to attend an important family function. I will therefore be unable to attend classes",
        "family_emergency": "I am writing to inform you that a family emergency requires my presence at home. I will therefore be unable to attend classes",
        "bereavement": "I am writing with regret to inform you of a bereavement in my family. I will therefore be unable to attend classes",
        "personal_work": "I am writing to inform you that I must attend to an important official matter in person. I will therefore be unable to attend classes",
        "exam_competition": "I am writing to inform you that I will be taking part in an exam/competition on the dates below. I will therefore be unable to attend classes",
        "travel_transport": "I am writing to inform you that a travel disruption has prevented me from reaching college. I am therefore unable to attend classes",
    },
    "polite_casual": {
        "illness": "I'm not feeling well and my doctor has suggested rest, so I won't be able to come to class",
        "medical_appointment": "I have a medical appointment I couldn't move, so I won't be able to come to class",
        "family_event": "There's an important family function I need to attend, so I won't be able to come to class",
        "family_emergency": "There's an urgent situation in my family that needs my attention, so I won't be able to come to class",
        "bereavement": "Sadly, there has been a loss in my family, so I won't be able to come to class",
        "personal_work": "I have some important official work to finish in person, so I won't be able to come to class",
        "exam_competition": "I'm taking part in an exam/competition, so I won't be able to come to class",
        "travel_transport": "My travel got disrupted and I couldn't reach on time, so I won't be able to come to class",
    },
}


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python generate_data.py` then `python train_classifier.py` first."
        )
    return joblib.load(path)


def classify_reason(model, reason: str) -> tuple[str, float]:
    """Return (label, confidence)."""
    probs = model.predict_proba([reason])[0]
    idx = probs.argmax()
    return model.classes_[idx], float(probs[idx])


def draft_letter(
    category: str,
    student_name: str,
    class_name: str,
    recipient: str,
    from_date: date,
    to_date: date,
    reason_detail: str = "",
    tone: str = "formal",
    sender_role: str = "student",
) -> str:
    tone = tone if tone in BODY else "formal"
    days = (to_date - from_date).days + 1
    if days < 1:
        raise ValueError("to_date must not be before from_date")

    when = (
        f"on {from_date:%d %B %Y}" if days == 1
        else f"from {from_date:%d %B %Y} to {to_date:%d %B %Y} ({days} days)"
    )
    detail = f" ({reason_detail.strip().rstrip('.')})" if reason_detail.strip() else ""
    body = BODY[tone][category]

    doc = SUPPORTING_DOC.get(category)
    doc_line = f"\nI will submit {doc} on my return.\n" if doc else ""

    closing = "Yours sincerely," if tone == "formal" else "Thank you for your understanding,"

    return f"""Date: {date.today():%d %B %Y}

To,
{recipient}

Subject: Application for leave {when}

Respected Sir/Madam,

{body}{detail} {when}.
{doc_line}
I request you to kindly grant me leave for the above period. I will make up for the
missed work and notes at the earliest.

{closing}

Name: {student_name}
Class / Section: {class_name}

Student's signature: ____________________

Parent/Guardian's name: ____________________
Parent/Guardian's signature: ____________________
(To be signed by the parent or guardian personally.)
"""
