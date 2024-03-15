"""In this file we try out a k-NN regessor to predict the pressure trace from experiment, as a baseline
"""

import json
import logging

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.paths import project_dir


def main():
    X_train, y_train, X_val, y_val = load_data()

    print(X_train.shape, y_train.shape)

    # Train k-NN regressor
    for k in [1, 2, 5, 10]:
        knn = train_knn(X_train, y_train, k)
        score = evaluate_knn(knn, X_val, y_val)

        print(f"Mean squared error of k-NN with k={k}: {score}")

        plot_knn_predictions(knn, X_val, y_val, k)


def train_knn(X_train, y_train, k):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn


def evaluate_knn(knn, X_val, y_val):
    y_pred = knn.predict(X_val)
    mse = np.mean((y_pred - y_val)**2)
    return mse


def plot_knn_predictions(knn, X_val, y_val, k):
    y_pred = knn.predict(X_val)

    n_rows = len(y_val) // 8 + 1
    n_cols = 8

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), sharex=True)
    fig.suptitle(f"Validation set predictions of k-NN for k={k}")

    for k, (y_true, y_p) in enumerate(zip(y_val, y_pred)):
        i = k // n_cols
        j = k % n_cols

        ax = axs[i, j]
        ax.plot(y_true, label='Ground truth')
        ax.plot(y_p, label='Prediction')
        ax.set_ylim(-0.2, 2.5)

    for ax in axs.flatten():
        ax.axis('off')

    plt.show()


def load_data():

    train_df = pd.read_csv(project_dir() / "data/exp/train.csv")
    val_df = pd.read_csv(project_dir() / "data/exp/val.csv")

    X_train = train_df.iloc[:, 1:4].values
    y_train = train_df.iloc[:, 4:].values

    X_val = val_df.iloc[:, 1:4].values
    y_val = val_df.iloc[:, 4:].values

    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    main()