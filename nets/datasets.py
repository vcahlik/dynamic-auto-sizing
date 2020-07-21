import pandas as pd
from sklearn.model_selection import train_test_split


DATA_DIR_PATH = '../datasets/'


def mnist_3():
    data = pd.read_csv(DATA_DIR_PATH + 'mnist.csv')
    X = data.iloc[:, 1:]
    y = (data['label'] == 3).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.to_numpy().T / 255
    y_train = y_train.to_numpy().reshape((1, -1))
    X_test = X_test.to_numpy().T / 255
    y_test = y_test.to_numpy().reshape((1, -1))

    return X_train, X_test, y_train, y_test


def fashion_mnist_tshirt():
    data = pd.read_csv(DATA_DIR_PATH + 'fashion-mnist.csv')
    X = data.iloc[:, 1:]
    y = (data['label'] == 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.to_numpy().T / 255
    y_train = y_train.to_numpy().reshape((1, -1))
    X_test = X_test.to_numpy().T / 255
    y_test = y_test.to_numpy().reshape((1, -1))

    return X_train, X_test, y_train, y_test
