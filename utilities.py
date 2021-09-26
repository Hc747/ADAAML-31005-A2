# import numpy as np
# from typing import Optional
#
#
# def require(x: Optional[any], field: str):
#     if x is None:
#         raise ValueError(f'Missing required value: "{field}".')
#     return x
#
#
# def default(x: Optional[any], otherwise: any) -> any:
#     return otherwise if x is None else x
#
#
# def is_numeric(x):
#     return x.dtype == int or x.dtype == float or x.dtype == bool
#
#
# def value_counts(y, normalise: bool = True):
#     classes, counts = np.unique(y, return_counts=True)
#     if normalise:
#         return classes, counts / (y if np.isscalar(y) else len(y))
#     return classes, counts
