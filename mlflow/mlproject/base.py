import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow

data = pd.read_csv("../data/titanic.csv")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"

X = data[features]
y = data[target]

X = pd.get_dummies(X, columns=["Sex", "Embarked"])

X["Age"].fillna(X["Age"].median(), inplace=True)
X["Fare"].fillna(X["Fare"].median(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Titanic_Survival_Prediction")

with mlflow.start_run(run_name="Baseline_RF"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    mlflow.log_params({"model": "RandomForest", "features": features})
    mlflow.log_metrics({"accuracy": acc, "f1": f1, "roc_auc": roc_auc})
    mlflow.sklearn.log_model(model, "model")