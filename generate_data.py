"""Generate a synthetic training dataset of absence reasons.

Each row is (text, label). Labels:
    illness, medical_appointment, family_event, family_emergency,
    bereavement, personal_work, exam_competition, travel_transport

Run:  python generate_data.py
"""
import csv
import random
from pathlib import Path

random.seed(42)

SEEDS = {
    "illness": [
        "I have a high fever", "I am down with the flu", "I caught a bad cold and cough",
        "I have food poisoning", "I have a severe headache and cannot concentrate",
        "I am suffering from viral fever", "I have stomach infection and vomiting",
        "I am sick and the doctor advised rest", "I have typhoid and need bed rest",
        "I have chickenpox", "I am unwell with body pain and weakness",
        "I injured my ankle while playing and cannot walk properly",
    ],
    "medical_appointment": [
        "I have a dentist appointment", "I have a scheduled eye check-up",
        "I need to visit the hospital for a follow-up", "I have a physiotherapy session",
        "I have a doctor consultation booked", "I need to get my vaccination done",
        "I have a pre-booked health check-up", "I have to go for blood tests and reports",
    ],
    "family_event": [
        "I have to attend my cousin's wedding", "there is a family function at home",
        "my sister's engagement ceremony is today", "we are celebrating a religious festival at home",
        "I must attend a family puja", "my brother's graduation ceremony is today",
        "I have to attend my relative's naming ceremony", "there is a housewarming ceremony in the family",
    ],
    "family_emergency": [
        "my mother is admitted to the hospital", "my father had an accident and I need to be with him",
        "there is a medical emergency in my family", "my grandmother fell ill suddenly",
        "I need to take my younger brother to the hospital", "a pipe burst and flooded our house",
        "there was a fire in our building and we are dealing with it",
    ],
    "bereavement": [
        "my grandfather passed away", "there was a death in the family",
        "I have to attend the funeral of my uncle", "we are observing the mourning rituals for my grandmother",
        "my relative passed away and we are travelling for the last rites",
    ],
    "personal_work": [
        "I have to renew my passport at the office", "I need to complete my Aadhaar update formalities",
        "I have a bank formality that needs me in person", "I have to appear for a visa interview",
        "I need to submit documents at a government office", "I have a driving licence test today",
    ],
    "exam_competition": [
        "I am participating in an inter-college debate", "I have a coaching entrance exam today",
        "I am representing the college in a sports tournament", "I have an external certification exam",
        "I am attending a hackathon", "I have a music competition performance",
        "I am appearing for a scholarship exam",
    ],
    "travel_transport": [
        "my train was cancelled and I cannot reach in time", "I am stuck due to a transport strike",
        "there is heavy flooding on the route to college", "my bus broke down on the highway",
        "my flight got delayed and I am still travelling", "roads are blocked due to heavy rain",
    ],
}

PREFIXES = ["", "Sir, ", "Respected madam, ", "Unfortunately, ", "Kindly note that ", "Today ", "Since yesterday "]
SUFFIXES = ["", " so I cannot attend classes", " and will be absent", " hence I request leave",
            " therefore I will miss lectures", ", please grant me leave"]


def main() -> None:
    rows = []
    for label, seeds in SEEDS.items():
        for seed in seeds:
            for _ in range(12):
                text = random.choice(PREFIXES) + seed + random.choice(SUFFIXES)
                text = text[0].upper() + text[1:]
                rows.append((text, label))
    random.shuffle(rows)

    out = Path("data")
    out.mkdir(exist_ok=True)
    with open(out / "reasons.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to data/reasons.csv")


if __name__ == "__main__":
    main()
