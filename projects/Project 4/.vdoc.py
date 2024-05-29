# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import pandas as pd
import numpy as np
import pyrsm as rsm 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.stats import pearsonr
import shap
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

# Load the data 
data = pd.read_csv('/home/jovyan/Desktop/MGTA495-2/projects/Project 4/data_for_drivers_analysis.csv')

data.head




#
#
#
#
#
# Define the independent variables (perceptions) and the dependent variable (satisfaction)
X = data[['trust', 'build', 'differs', 'easy', 'appealing', 'rewarding', 'popular', 'service', 'impact']]
y = data['satisfaction']

# Standardize the independent variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit a linear regression model
reg = LinearRegression().fit(X_scaled, y)

# Standardized regression coefficients
standardized_coefficients = reg.coef_

# Calculate permutation importance (as an approximation for Shapley values)
perm_importance = permutation_importance(reg, X_scaled, y, n_repeats=30, random_state=42)
perm_importance_mean = perm_importance.importances_mean

# Fit a Random Forest model for Mean Decrease in Gini Coefficient
rf = RandomForestRegressor()
rf.fit(X_scaled, y)
gini_importance = rf.feature_importances_

# Fit an XGBoost model and get feature importance
xgb_model = XGBRegressor()
xgb_model.fit(X_scaled, y)
xgb_importance = xgb_model.feature_importances_

# Calculate Johnson's Relative Weights (using a suitable approximation)
johnson_weights = np.abs(standardized_coefficients) / np.sum(np.abs(standardized_coefficients))

# Calculate Pearson Correlations
pearson_correlations = {col: pearsonr(X[col], y)[0] for col in X.columns}

# Calculate "Usefulness" as an average of standardized coefficients, Shapley values, and Johnson's epsilon
usefulness = (np.abs(standardized_coefficients) + perm_importance_mean + johnson_weights) / 3

# Create the table
results = pd.DataFrame({
    'Perception': X.columns,
    'Pearson Correlations (%)': [round(pearson_correlations[col] * 100, 1) for col in X.columns],
    'Standardized Multiple Regression Coefficients (%)': [round(coef * 100, 1) for coef in standardized_coefficients],
    'Shapley Values (%)': [round(value * 100, 1) for value in perm_importance_mean],
    'Johnson\'s Epsilon (%)': [round(weight * 100, 1) for weight in johnson_weights],
    'Mean Decrease in RF Gini Coefficient (%)': [round(value * 100, 1) for value in gini_importance],
    'XGBoost Importance (%)': [round(value * 100, 1) for value in xgb_importance],
    'Usefulness (%)': [round(value * 100, 1) for value in usefulness]
})

# Display the final results
results = results[['Perception', 'Pearson Correlations (%)', 'Standardized Multiple Regression Coefficients (%)',
                   'Shapley Values (%)', 'Johnson\'s Epsilon (%)', 'Mean Decrease in RF Gini Coefficient (%)',
                   'XGBoost Importance (%)', 'Usefulness (%)']]
results.index = results.index + 1

styled_df = results.style.background_gradient(cmap='Greens', axis=None, vmin=0, vmax=29)
def format_func(val):
    if isinstance(val, (int, float)):
        return f"{val:.1f}"
    return val

styled_df = styled_df.format(format_func)


styled_df






#
#
#
