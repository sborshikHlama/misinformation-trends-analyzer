"""
Real MLflow experiments for BERTopic and LogReg models.
Actually trains models with different hyperparameters and logs real metrics.

Usage: python scripts/test_models_mlflow.py
"""
import logging
import re
from pathlib import Path

import kagglehub
import mlflow
import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from umap import UMAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_URI = "file:./mlruns"


def load_data() -> pd.DataFrame:
    """Download and prepare the fake news dataset."""
    logger.info("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")

    df_true = pd.read_csv(f"{path}/True.csv")
    df_fake = pd.read_csv(f"{path}/Fake.csv")

    df_true["label"] = 1
    df_fake["label"] = 0

    df = pd.concat([df_true, df_fake], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)

    def clean_text(text: str) -> str:
        text = str(text)
        text = re.sub(r"^[A-Z][a-z]+ [A-Z][a-z]+ says\s*", "", text)
        text = re.sub(r"Featured image.*$", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df["text_clean"] = df["text"].apply(clean_text)
    logger.info(f"Loaded {len(df)} articles")
    return df


def calc_bertopic_metrics(topic_model: BERTopic, topics: list) -> dict:
    """Calculate BERTopic evaluation metrics."""
    topic_info = topic_model.get_topics()
    n_topics = len([t for t in topic_info.keys() if t != -1])

    outlier_count = sum(1 for t in topics if t == -1)
    outlier_ratio = outlier_count / len(topics)

    all_words = []
    for topic_id, words in topic_info.items():
        if topic_id != -1 and words:
            all_words.extend([w[0] for w in words[:10]])
    diversity = len(set(all_words)) / len(all_words) if all_words else 0

    return {
        "n_topics": n_topics,
        "outlier_ratio": round(outlier_ratio, 4),
        "topic_diversity": round(diversity, 4),
    }


def run_bertopic_experiments(df: pd.DataFrame) -> None:
    """Run BERTopic experiments with different hyperparameters."""
    mlflow.set_experiment("BERTopic_Topic_Modeling")
    texts = df["text_clean"].tolist()

    sample_size = min(10000, len(texts))
    texts_sample = texts[:sample_size]
    logger.info(f"Using {sample_size} samples for BERTopic")

    experiments = [
        {"name": "bertopic_default", "min_cluster_size": 10, "n_neighbors": 15},
        {"name": "bertopic_cluster_5", "min_cluster_size": 5, "n_neighbors": 15},
        {"name": "bertopic_cluster_200", "min_cluster_size": 200, "n_neighbors": 15},
        {"name": "bertopic_neighbors_5", "min_cluster_size": 10, "n_neighbors": 5},
    ]

    for exp in experiments:
        logger.info(f"Running {exp['name']}...")

        umap_model = UMAP(
            n_neighbors=exp["n_neighbors"], n_components=5,
            min_dist=0.0, metric="cosine", random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=exp["min_cluster_size"], min_samples=10,
            metric="euclidean", prediction_data=True,
        )
        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english")

        topic_model = BERTopic(
            umap_model=umap_model, hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer, language="english", verbose=False,
        )

        topics, _ = topic_model.fit_transform(texts_sample)
        metrics = calc_bertopic_metrics(topic_model, topics)

        with mlflow.start_run(run_name=exp["name"]):
            mlflow.log_param("min_cluster_size", exp["min_cluster_size"])
            mlflow.log_param("n_neighbors", exp["n_neighbors"])
            mlflow.log_param("sample_size", sample_size)
            mlflow.log_metric("n_topics", metrics["n_topics"])
            mlflow.log_metric("outlier_ratio", metrics["outlier_ratio"])
            mlflow.log_metric("topic_diversity", metrics["topic_diversity"])

        logger.info(f"  {exp['name']}: {metrics}")

    logger.info("BERTopic: 4 experiments completed")


def run_logreg_experiments(df: pd.DataFrame) -> None:
    """Run LogReg experiments with different hyperparameters."""
    mlflow.set_experiment("LogReg_Fake_Detection")

    X = df["text_clean"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    experiments = [
        {"name": "logreg_5000_features", "max_features": 5000, "C": 1.0},
        {"name": "logreg_10000_features", "max_features": 10000, "C": 1.0},
        {"name": "logreg_100_features", "max_features": 100, "C": 1.0},
        {"name": "logreg_C_0001", "max_features": 5000, "C": 0.0001},
    ]

    for exp in experiments:
        logger.info(f"Running {exp['name']}...")

        vectorizer = TfidfVectorizer(
            max_features=exp["max_features"], ngram_range=(1, 2), stop_words="english",
        )
        classifier = LogisticRegression(C=exp["C"], max_iter=1000, random_state=42)

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        classifier.fit(X_train_vec, y_train)
        y_pred = classifier.predict(X_test_vec)
        y_proba = classifier.predict_proba(X_test_vec)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        with mlflow.start_run(run_name=exp["name"]):
            mlflow.log_param("max_features", exp["max_features"])
            mlflow.log_param("C", exp["C"])
            mlflow.log_param("ngram_range", "(1, 2)")
            mlflow.log_param("stop_words", "english")
            mlflow.log_metric("accuracy", round(acc, 4))
            mlflow.log_metric("f1_score", round(f1, 4))
            mlflow.log_metric("roc_auc", round(roc, 4))
            mlflow.log_metric("n_features_actual", len(vectorizer.get_feature_names_out()))

        logger.info(f"  {exp['name']}: acc={acc:.4f}, f1={f1:.4f}, roc={roc:.4f}")

    logger.info("LogReg: 4 experiments completed")


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)

    df = load_data()

    logger.info("=" * 50)
    logger.info("Running LogReg experiments...")
    run_logreg_experiments(df)

    logger.info("=" * 50)
    logger.info("Running BERTopic experiments...")
    run_bertopic_experiments(df)

    logger.info("=" * 50)
    logger.info("Done! View results: mlflow ui --port 5001")


if __name__ == "__main__":
    main()
