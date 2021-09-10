from typing import Optional


def require(x: Optional[any], field: str):
    if x is None:
        raise ValueError(f'Missing required value: "{field}".')
    return x
