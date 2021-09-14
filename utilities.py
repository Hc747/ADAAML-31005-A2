import pandas as pd
import numpy as np
from typing import Optional


def require(x: Optional[any], field: str):
    if x is None:
        raise ValueError(f'Missing required value: "{field}".')
    return x


# def value_counts(y, normalise: bool = True):
#     return pd.Series(y).value_counts(normalize=normalise)
#
#
# def compute_entropy(y, eps=1e-9):
#     frequencies = np.array(value_counts(y, normalise=True))
#     return -(frequencies * np.log2(frequencies + eps)).sum()


def value_counts(y, normalise: bool = True):
    classes, counts = np.unique(y, return_counts=True)
    # print(f'input: {y}\nclasses: {classes}\ncounts: {counts}')
    if normalise:
        return classes, counts / (y if np.isscalar(y) else len(y))
    return classes, counts
    # if normalise:
    #     counts /= len(y)
    # return classes, counts


# def __compute_entropy(y, eps) -> float:
#     classes, frequency = value_counts(y, normalise=True)
#     # classes, counts = np.unique(y, return_counts=True)
#     # frequency = counts / len(y)
#     # frequency = value_counts(y, normalise=True)
#     return -(frequency * np.log2(frequency + eps)).sum()
#
# # def entropy(y, eps) -> float
# # def value_counts(x, normalise=True):
# #     unique, counts = np.unique(x, return_counts=True)
#
#
#
# def compute_entropy(y, eps=1e-9) -> float:
#     return 0 if len(y) < 2 else __compute_entropy(y, eps)
