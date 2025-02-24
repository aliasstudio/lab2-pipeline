import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


def preprocess_data(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        df = pd.read_csv(os.path.join(input_folder, file))
        scaler = StandardScaler()
        df['x'] = scaler.fit_transform(df[['x']])
        df['y'] = scaler.fit_transform(df[['y']])
        df.to_csv(os.path.join(output_folder, file), index=False)


if __name__ == "__main__":
    preprocess_data("train", "train_processed")
    preprocess_data("test", "test_processed")