import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from src.data import load_and_preprocess

def evaluate(model_path, vectorizer_path, data_path):
    # Load artifacts
    model = joblib.load(model_path)

    # Reload data
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Reports
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate(
        model_path="artifacts/svm_model.pkl",
        vectorizer_path="artifacts/vectorizer.pkl",
        data_path="data/ai_vs_human_cleaned.csv"
    )
