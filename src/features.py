# src/features.py
from sklearn.feature_extraction.text import TfidfVectorizer
from config import TFIDF_PARAMS


def get_tfidf_vectorizer():
    """
    Create and return a TfidfVectorizer instance.
    """
    return TfidfVectorizer(**TFIDF_PARAMS)

def fit_transform_tfidf(vectorizer, X_train, X_test):
    """
    Fit the TfidfVectorizer on the training data and transform both training and test data.
    
    Parameters:
    - vectorizer: An instance of TfidfVectorizer.
    - X_train: List of training documents.
    - X_test: List of test documents.
    
    Returns:
    - X_train_tfidf: TF-IDF features for the training data.
    - X_test_tfidf: TF-IDF features for the test data.
    """
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf

    