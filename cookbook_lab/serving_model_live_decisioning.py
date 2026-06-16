
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = load_wine()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

mlflow.set_experiment("Wine Model Experiment")

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

for n_estimators in [50, 100]:
    for max_depth in [3, 5, 10]:

        with mlflow.start_run():

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            rmse, mae, r2 = eval_metrics(y_test, predictions)

            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            mlflow.sklearn.log_model(model, "model")

            print("Model completed")
            print("n_estimators:", n_estimators)
            print("max_depth:", max_depth)
            print("RMSE:", rmse)
            print("MAE:", mae)
            print("R2:", r2)
            print("-------------------------")
