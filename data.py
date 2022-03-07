import numpy as np
import pandas as pd


def get_data():
    np.random.seed(0)
    df = np.array(pd.read_csv("fetal_health.csv"), dtype=np.float32)
    np.random.shuffle(df)

    X = np.array(df[:, :-1])
    # We subtract 1 to make classes 0,1,2 instead of 1,2,3
    y = np.array(df[:, -1], dtype=np.int64) - 1

    num_data = len(y)
    num_train = int(np.floor(num_data * 0.6))
    num_val = int(np.floor(num_data * 0.2))

    train_X = X[:num_train]
    train_y = y[:num_train]

    val_X = X[num_train:num_train + num_val]
    val_y = y[num_train:num_train + num_val]

    test_X = X[num_train + num_val:]
    test_y = y[num_train + num_val:]

    return train_X, train_y, val_X, val_y, test_X, test_y
