import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

def test_tfidf_vectorizer_fit_transform():
    docs = ["hello world", "machine learning"]
    vectorizer = TfidfVectorizer(max_features=5)
    X = vectorizer.fit_transform(docs)

    assert X.shape[0] == 2   # 2 docs
    assert X.shape[1] <= 5   # at most 5 features
    assert isinstance(vectorizer.vocabulary_, dict)
