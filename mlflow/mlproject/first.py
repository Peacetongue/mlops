import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow

data = pd.read_csv("../data/titanic.csv")

data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

title_counts = data['Title'].value_counts()
rare_titles = title_counts[title_counts < 10].index
data['Title'] = data['Title'].replace(rare_titles, 'Other')

data['FamilySize'] = data['SibSp'] + data['Parch']
data['IsAlone'] = (data['FamilySize'] == 0).astype(int)

features = ["Pclass", "Sex", "Age", "Fare", "FamilySize", "IsAlone", "Title", "Embarked"]
target = "Survived"

X = data[features]
y = data[target]

X = pd.get_dummies(X, columns=["Sex", "Embarked", "Title"])

X["Age"].fillna(X["Age"].median(), inplace=True)
X["Fare"].fillna(X["Fare"].median(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Titanic_Survival_Prediction")

with mlflow.start_run(run_name="Feature_Engineering"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mlflow.log_params({
        "features": list(X.columns),
        "added_features": ["FamilySize", "IsAlone", "Title"]
    })

    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    })

    mlflow.sklearn.log_model(model, "model_with_features")
