"""Train and evaluate a reason-for-absence classifier (TF-IDF + Logistic Regression).

Run:  python train_classifier.py
Outputs: models/reason_classifier.joblib and a printed evaluation report.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=5.0)),
    ])


def main() -> None:
    df = pd.read_csv("data/reasons.csv")

    # Split on the unique seed-ish text so near-duplicates don't leak across splits.
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/reason_classifier.joblib")
    print("Saved model to models/reason_classifier.joblib")


if __name__ == "__main__":
    main()
