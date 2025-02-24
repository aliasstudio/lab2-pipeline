import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os


def train_model(train_folder):
    X_train = []
    y_train = []

    for file in os.listdir(train_folder):
        df = pd.read_csv(os.path.join(train_folder, file))
        X_train.append(df['x'].values.reshape(-1, 1))
        y_train.append(df['y'].values)

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Сохраняем модель
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(model, "models/model.pkl")


if __name__ == "__main__":
    train_model("train_processed")