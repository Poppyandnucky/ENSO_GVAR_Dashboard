# analysis/var_spec.py
from dataclasses import dataclass
from typing import List

@dataclass
class VARSpec:
    endog_vars: List[str]
    exog_vars: List[str]
    include_const: bool = True
    demean: bool = True
    standardize: bool = True
    p: int = 1

def prepare_varx_data(Y, X, add_const=True):
    ...

def fit_varx(Y, X, p):
    ...
