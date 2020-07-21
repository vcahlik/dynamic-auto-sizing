from .core import forward

from sklearn.metrics import accuracy_score


def measure_accuracy(parameters, X, y, threshold=0.5):
    p, _ = forward(X, parameters)
    y_pred = (p >= threshold).astype(int)
    return accuracy_score(y.reshape((-1), ), y_pred.reshape((-1), ))
