# tvp/__init__.py
#
# Core numerical routines for time-varying parameter (TVP) models.
# This package contains ONLY math:
#   - Kalman / TVP-VAR updates
#   - IRF computation utilities
#
# No pandas, no statsmodels, no country logic, no Streamlit.
# This package must be importable in isolation and safe for unit testing.

from models.kalman_var import tvp_var_with_exog
from models.irf import compute_irf_varp

__all__ = [
    "tvp_var_with_exog",
    "compute_irf_varp",
]