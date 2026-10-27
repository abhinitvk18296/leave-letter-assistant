# 📝 Leave Letter Assistant

A small machine-learning project that helps students write **honest** leave applications.

1. A **text classifier** (TF-IDF + Logistic Regression, scikit-learn) reads your reason for being absent and
   labels it: illness, medical appointment, family function, family emergency, bereavement,
   official work, exam/competition, or travel disruption.
2. A **letter generator** picks the right wording and tone (formal or polite-casual), fills in your
   details and dates, and tells you which supporting document to bring.
3. A **Streamlit web app** and a **command-line tool** wrap it all.

> ⚠️ **What this project deliberately does NOT do:** it never fabricates medical certificates, doctor
> signatures, or parent signatures. Every letter ends with **blank signature lines**. A parent/guardian must
> sign personally, and any medical certificate must be issued by a real doctor. This tool only drafts wording
> for genuine absences.

## Project structure

```
leave-letter-assistant/
├── generate_data.py      # builds a synthetic dataset -> data/reasons.csv
├── train_classifier.py   # trains + evaluates the model -> models/reason_classifier.joblib
├── letter_generator.py   # classify_reason() and draft_letter()
├── app.py                # Streamlit web app
├── cli.py                # command-line version
├── tests/test_letter.py  # pytest tests
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python generate_data.py          # 1. create the dataset
python train_classifier.py       # 2. train the model and print the evaluation report
streamlit run app.py             # 3a. web app
python cli.py                    # 3b. or the command-line version
pytest                           # run tests
```

## How it works

- **Data:** `generate_data.py` combines seed sentences for each category with random prefixes and suffixes
  (e.g. "Sir, I have a high fever so I cannot attend classes"). The data is synthetic, so the metrics will look
  very high. Treat them as a sanity check, not a real-world benchmark.
- **Model:** `TfidfVectorizer` (1-2 grams) feeding a `LogisticRegression`, wrapped in a scikit-learn `Pipeline`.
- **Low confidence:** if the model is under 40% sure, the app asks the user to double-check the category.

## Ideas to extend it

- Replace the synthetic data with real, anonymised phrases and measure honest accuracy.
- Fine-tune a small seq2seq model (e.g. T5) to rephrase letters in more natural language.
- Add Hindi or other-language support.
- Export letters as PDF (with blank signature lines).
- Deploy on Streamlit Community Cloud.

## License

MIT
