import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import mean_squared_error


def test_model(test_folder, model_path):
    model = joblib.load(model_path)

    mse_scores = []

    for file in os.listdir(test_folder):
        df = pd.read_csv(os.path.join(test_folder, file))
        X_test = df['x'].values.reshape(-1, 1)
        y_test = df['y'].values

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

    print(f"Average MSE: {np.mean(mse_scores)}")


if __name__ == "__main__":
    test_model("test_processed", "models/model.pkl")