import numpy as np


def cmapss_score(predict: np.ndarray, label: np.ndarray) -> float:
    a1 = 13
    a2 = 10
    error = predict - label
    pos_e = np.exp(-error[error < 0] / a1) - 1
    neg_e = np.exp(error[error >= 0] / a2) - 1
    return sum(pos_e) + sum(neg_e)