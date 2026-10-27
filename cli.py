"""Command-line version.  Run:  python cli.py"""
from datetime import date, datetime

from letter_generator import CATEGORY_LABELS, classify_reason, draft_letter, load_model


def ask_date(prompt: str) -> date:
    while True:
        raw = input(f"{prompt} (YYYY-MM-DD, blank = today): ").strip()
        if not raw:
            return date.today()
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            print("Please use the format YYYY-MM-DD.")


def main() -> None:
    model = load_model()
    reason = input("Why will you be absent? ").strip()
    label, conf = classify_reason(model, reason)
    print(f"-> Detected: {CATEGORY_LABELS[label]} ({conf:.0%} confidence)")

    name = input("Your name: ").strip()
    cls = input("Class/section: ").strip()
    recipient = input("Addressed to [The Class Teacher]: ").strip() or "The Class Teacher"
    start = ask_date("From")
    end = ask_date("To")
    tone = input("Tone: formal / polite_casual [formal]: ").strip() or "formal"

    print("\n" + draft_letter(label, name, cls, recipient, start, end, reason, tone))


if __name__ == "__main__":
    main()
