import argparse
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=True)
args = parser.parse_args()

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = Ridge(alpha=args.alpha)
model.fit(X_train, y_train)
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)

mlflow.log_param("alpha", args.alpha)
mlflow.log_metric("rmse", rmse)
mlflow.sklearn.log_model(model, "model")