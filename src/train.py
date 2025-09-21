import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from src.data import load_and_preprocess
from sklearn.pipeline import Pipeline

import os
import joblib
import pandas as pd

DATA_PATH = "data/ai_vs_human_cleaned.csv"
ARTIFACT_DIR = "artifacts"

def train_svm(config, X_train, y_train, X_test, y_test):
    """Train and evaluate a single SVM config (for RayTune)."""
    
    # Create pipeline: TF-IDF + SVM
    pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
            ("svm", SVC(C=config["C"], kernel=config["kernel"], gamma=config["gamma"], probability=True))
        ])

    X_train = X_train.copy()
    X_test = X_test.copy()
    
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    ray.train.report({"f1_score": f1, "roc_auc": roc})


def main():
    X_train, X_test, y_train, y_test = load_and_preprocess(DATA_PATH)

    search_space = {
        "C": tune.loguniform(1e-2, 10),
        "kernel": tune.choice(["linear", "rbf"]),
        "gamma": tune.loguniform(1e-4, 1)
    }

    tuner = tune.Tuner(
        tune.with_parameters(train_svm, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test),
        tune_config=tune.TuneConfig(
            search_alg=HyperOptSearch(metric="f1_score", mode="max"),
            num_samples=10
        ),
        param_space=search_space
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="f1_score", mode="max")
    best_config = best_result.config

    print("\n[INFO] Best config:", best_result.config)
    print("[INFO] Best F1:", best_result.metrics["f1_score"])
    print("[INFO] Best ROC-AUC:", best_result.metrics["roc_auc"])

    # Retrain best model on full train set
    best_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
        ("svm", SVC(C=best_config["C"], kernel=best_config["kernel"], gamma=best_config["gamma"], probability=True))
    ])

    best_pipeline.fit(X_train, y_train)

    # example_input = pd.DataFrame({"clean_text": X_test[0]})
    # signature = infer_signature(X_test[:5], best_pipeline.predict(X_test[:5]))
    
    # Final evaluation
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # MLflow logging
    with mlflow.start_run():
        mlflow.log_params(best_result.config)
        mlflow.log_metrics({
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        })
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model",
            registered_model_name="SVMTextClassifier",
            pip_requirements=[
                "scikit-learn==1.1.2",
                "cloudpickle==3.1.1",
                "pandas",
                "numpy"
            ],
            # signature=signature,
            # input_example=example_input
        )
    # Save artifacts
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(best_pipeline, os.path.join(ARTIFACT_DIR, "svm_model.pkl"))

if __name__ == "__main__":
    main()
