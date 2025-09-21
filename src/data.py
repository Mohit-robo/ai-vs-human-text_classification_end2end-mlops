import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_and_preprocess(path, test_size=0.2, random_state=42):
    df = pd.read_csv(path)

    # Prepare X and y
    X = df['clean_text'].astype(str)
    y = df['label'].map({'Human': 0, 'AI': 1})


    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test
