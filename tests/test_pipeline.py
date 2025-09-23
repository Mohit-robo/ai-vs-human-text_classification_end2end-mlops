import pytest
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10)),
        ("clf", SVC())
    ])

def test_pipeline_fit_predict():
    X = ["hello machine learning", "world of ai"]
    y = [1, 0]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    preds = pipeline.predict(["ai is cool"])
    assert preds.shape == (1,)
