import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("C:/Users/susai/OneDrive/Desktop/College/PROJECTS/Codealpha Tasks/car data.csv")

# Data preprocessing
data = data.drop(['Car_Name'], axis=1)
data['current_year'] = 2020
data['no_year'] = data['current_year'] - data['Year']
data = data.drop(['Year', 'current_year'], axis=1)

# One-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Debugging: Print columns after preprocessing
print("Columns after preprocessing:", data.columns)

# Map required column names to actual column names
column_mapping = {
    'Kms_Driven': 'Driven_kms',
    'Seller_Type_Individual': 'Selling_type_Individual'
}

# Replace column names in the required list
required_columns = [
    'Selling_Price', 'Present_Price', 'Driven_kms', 'no_year', 'Owner',
    'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Selling_type_Individual', 'Transmission_Manual'
]

# Check if all required columns are present
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    print("Missing columns:", missing_columns)
    raise KeyError(f"Columns missing after preprocessing: {missing_columns}")

data = data[required_columns]

# Feature-target split
x = data.iloc[:, 1:]
y = data.iloc[:, 0]

# Feature importance using ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(x, y)
print("Feature Importances:", model.feature_importances_)

# Define parameter grid
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf
}
print("Hyperparameter Grid:", grid)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

# RandomizedSearchCV for RandomForestRegressor
model = RandomForestRegressor(random_state=42)
hyp = RandomizedSearchCV(estimator=model, param_distributions=grid, n_iter=10,
                         scoring='neg_mean_squared_error', cv=5, verbose=2,
                         random_state=42, n_jobs=1)
hyp.fit(x_train, y_train)

# Predict and evaluate
y_pred = hyp.predict(x_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))
