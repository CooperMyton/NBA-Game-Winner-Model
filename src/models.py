#build out different models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_logistic():
    return LogisticRegression(max_iter=1000)

def get_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
