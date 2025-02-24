import numpy as np
import os
import pandas as pd


def generate_data(n_samples=100, noise=False, anomaly=False):
    # Генерация данных (например, синусоидальная функция)
    x = np.linspace(0, 10, n_samples)
    y = np.sin(x)

    if noise:
        y += np.random.normal(0, 0.1, size=y.shape)

    if anomaly:
        y[50] += 5  # Добавляем аномалию

    return x, y


def save_data(x, y, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    df = pd.DataFrame({'x': x, 'y': y})
    df.to_csv(os.path.join(folder, filename), index=False)


if __name__ == "__main__":
    # Генерация обучающих данных
    x_train, y_train = generate_data(noise=True)
    save_data(x_train, y_train, "train", "train_data.csv")

    # Генерация тестовых данных
    x_test, y_test = generate_data(anomaly=True)
    save_data(x_test, y_test, "test", "test_data.csv")