#!/usr/bin/env python3
"""
Spam Detector - SMS/Email spam classification using Naive Bayes + TF-IDF
Closes: #1
"""

import re
import string
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


# ── Data ──────────────────────────────────────────────────────────────────────

DATASET_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
    "master/data/sms.tsv"
)


def load_dataset(path: str | None = None) -> pd.DataFrame:
    """Load SMS Spam Collection dataset (TSV). Downloads if path is None."""
    if path:
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    else:
        print("Downloading SMS Spam Collection dataset...")
        with urllib.request.urlopen(DATASET_URL) as r:
            lines = r.read().decode("utf-8").splitlines()
        rows = [line.split("\t", 1) for line in lines if "\t" in line]
        df = pd.DataFrame(rows, columns=["label", "text"])
    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_clean"] = df["text"].apply(clean_text)
    df["label_bin"] = (df["label"] == "spam").astype(int)   # spam=1, ham=0
    return df


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    """Return a TF-IDF vectorizer and a Multinomial Naive Bayes classifier."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        max_features=10_000,
        sublinear_tf=True,
    )
    classifier = MultinomialNB(alpha=0.1)
    return vectorizer, classifier


def train(df: pd.DataFrame):
    """Train and return (vectorizer, classifier, X_test, y_test)."""
    X = df["text_clean"]
    y = df["label_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer, clf = build_model()
    X_train_vec = vectorizer.fit_transform(X_train)
    clf.fit(X_train_vec, y_train)

    return vectorizer, clf, X_test, y_test


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(vectorizer, clf, X_test, y_test):
    """Print classification metrics."""
    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)

    print("\n── Evaluation Results ──────────────────────────────")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-Score : {f1_score(y_test, y_pred):.4f}")
    print("\n── Classification Report ───────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("── Confusion Matrix ────────────────────────────────")
    cm = confusion_matrix(y_test, y_pred)
    print(f"           Predicted ham  Predicted spam")
    print(f"Actual ham       {cm[0][0]:>5}           {cm[0][1]:>5}")
    print(f"Actual spam      {cm[1][0]:>5}           {cm[1][1]:>5}")
    print()


# ── Prediction helper ─────────────────────────────────────────────────────────

def predict(vectorizer, clf, messages: list[str]) -> list[dict]:
    """Predict spam/ham for a list of raw message strings."""
    cleaned = [clean_text(m) for m in messages]
    vecs = vectorizer.transform(cleaned)
    labels = clf.predict(vecs)
    probs = clf.predict_proba(vecs)
    results = []
    for msg, label, prob in zip(messages, labels, probs):
        results.append({
            "message": msg,
            "prediction": "spam" if label == 1 else "ham",
            "confidence": f"{max(prob):.2%}",
        })
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load
    df = load_dataset()
    print(f"Dataset: {len(df)} messages — {df['label'].value_counts().to_dict()}")

    # 2. Preprocess
    df = preprocess(df)

    # 3. Train
    print("Training model...")
    vectorizer, clf, X_test, y_test = train(df)

    # 4. Evaluate
    evaluate(vectorizer, clf, X_test, y_test)

    # 5. Demo predictions
    samples = [
        "Congratulations! You've won a free iPhone. Click here to claim now!",
        "Hey, are we still on for lunch tomorrow?",
        "URGENT: Your bank account has been suspended. Call 0800-FREE now.",
        "Can you pick up some milk on your way home?",
        "You have been selected for a $1000 gift card. Reply WIN to claim.",
    ]
    print("── Sample Predictions ──────────────────────────────")
    for r in predict(vectorizer, clf, samples):
        print(f"  [{r['prediction'].upper():4s}] ({r['confidence']})  {r['message'][:70]}")


if __name__ == "__main__":
    main()
