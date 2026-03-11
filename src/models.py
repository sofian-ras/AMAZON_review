from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score,
                            precision_score, 
                            recall_score, 
                            f1_score, 
                            roc_auc_score
                            )

def get_lr():
    """
    Create and return a Logistic Regression model instance.
    """
    return LogisticRegression(random_state=42, max_iter=1000)

def get_nb():
    """
    Create and return a Multinomial Naive Bayes model instance.
    """
    return MultinomialNB()

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Train the model and evaluate its performance on the test set.
    
    Parameters:
    - model: The machine learning model to train.
    - X_train: Training features.
    - y_train: Training labels.
    - X_test: Test features.
    - y_test: Test labels.
    
    Returns:
    A dictionary containing evaluation metrics.
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    # Calculate evaluation metrics
    metrics  = {
        "model_name"   : type(model).__name__,
        "accuracy"     : accuracy_score(y_test, y_pred),
        "f1"           : f1_score(y_test, y_pred),
        "precision"    : precision_score(y_test, y_pred),
        "recall"       : recall_score(y_test, y_pred),
        "roc_auc"      : roc_auc_score(y_test, y_pred_proba),
        "y_pred"       : y_pred,
        "y_pred_proba" : y_pred_proba,
    }
    
    return metrics