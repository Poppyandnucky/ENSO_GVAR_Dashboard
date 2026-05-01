# analysis/__init__.py
#
# Model orchestration and analysis logic.
#
# This package coordinates:
#   - Data preparation
#   - Fixed VAR and TVP-VAR estimation
#   - Scenario construction and forecasting
#
# Modules here may depend on tvp/, pandas, and statsmodels,
# but tvp/ must never import from analysis/.