from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score, roc_curve
from dataclasses import dataclass


@dataclass
class Params:
    params  = {

                "LogisticRegression": {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear'],
                    'max_iter': [1000]
                },

                "RandomForestClassifier": {
                    'n_estimators': [100, 200],
                    'max_depth': [20, 30, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },

                "SVC": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                },

                "MultinomialNB": {
                    'alpha': [0.5, 1.0]
                }
            }

class Models:
    def __init__(self):
        self.models = {
        "LogisticRegression" : LogisticRegression(),
        "RandomForestClassifier" : RandomForestClassifier(),
        "MultinomialNB" : MultinomialNB(),
        "SVC" : SVC()
        }
        
        
if __name__ == "__main__":
    models = Models().models

    for key, model in models.items():
        grid = GridSearchCV(
        model, 
        Params.params[key],
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

        # grid.fit(X_train_tfidf, y_train)

        # print(f"\nMeilleurs paramètres : {grid.best_params_}")
        # print(f"Meilleur score CV : {grid.best_score_:.4f}")


        # y_pred = grid.predict(X_test_tfidf)
        # y_proba = grid.predict_proba(X_test_tfidf)[:, 1]

        # # Métriques
        # classification = classification_report(y_test, y_pred)
        # print(classification)

        # # Matrice de confusion
        # cm = confusion_matrix(y_test, y_pred_best)
        # print(cm)
    
            
    