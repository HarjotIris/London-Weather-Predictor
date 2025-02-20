import pandas as pd
import numpy as np

path = r"C:\Desktop\PRACTICE\LONDON WEATHER\london_weather.csv"

df = pd.read_csv(path)

print(df.info())
print(df.head())

# convert 'date' to a proper datetime format
df['date'] = pd.to_datetime(df['date'], format = '%Y%m%d')

# check missing values in each column
missing_values = df.isnull().sum()
print(missing_values)

# Fill missing values:
# - 'snow_depth' has too many missing values, so we drop it.
# - Other missing values are filled with column medians.

df.drop(columns=['snow_depth'], inplace=True)
df.fillna(df.median(), inplace=True)

# verify missing values are handles
print(df.isnull().sum(), df.head())


from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df.drop(columns=['mean_temp', 'date'])
y = df['mean_temp']

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# check dataset sizes
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# train a decision tree regressor
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# predict on test data
y_pred_tree = tree_model.predict(X_test)

# evaluate performance
# Mean Absolute Error (MAE) → Measures how far predictions are from actual values.
# R² Score → Measures how well the model explains target (temperature in our case) variations.
mae_tree = mean_absolute_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print(mae_tree, r2_tree)
# Results:
# MAE: 0.96°C → On average, predictions are off by 0.96°C.
# R² Score: 0.95 → The model explains 95% of temperature variations.


from sklearn.ensemble import RandomForestRegressor

# train a random forest regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42) # Creates a Random Forest model with 100 trees.
rf_model.fit(X_train, y_train)

# make predictions on test data
y_pred_rf = rf_model.predict(X_test)

# evaluate performance
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(mae_rf, r2_rf)
# Results:
# MAE: 0.69°C (better than Decision Tree's 0.96°C).
# R² Score: 0.97 (explains 97% of temperature variation, better than 0.95).

# Why is Random Forest Better?
# It reduces overfitting by using multiple trees.
# It averages predictions, making it more stable.
# It learns better patterns from data.
'''
Using GridSearchCV
# finetuning hyperparameters
from sklearn.model_selection import GridSearchCV

# define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200], # number of trees
    'max_depth': [10, 20, None],  # depth of each tree
    'min_samples_split': [2, 5, 10], # minimum samples needed to split
    'min_samples_leaf': [1, 2, 4], # minimum samples per leaf
    'max_features': ['sqrt', 'log2'] # number of features to consider per split
}

# initialize random forest model
rf = RandomForestRegressor(random_state=42)

# perform grid search (tries all combinations)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# get best hyperparameters
best_params = grid_search.best_params_

print(best_params)
# What This Does:
# Defines a range of values for each hyperparameter.
# Uses GridSearchCV to test all possible combinations.
# Uses cv=5 (5-fold cross-validation) for better accuracy.
# Selects the best hyperparameters based on the highest R² Score.
'''
'''
Using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Define hyperparameter ranges
param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Initialize model
rf = RandomForestRegressor(random_state=42)

# Randomized Search (tries 15 random combinations)
random_search = RandomizedSearchCV(rf, param_dist, n_iter=15, cv=5, scoring='r2', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Get best params
print(random_search.best_params_)
# {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 30}
'''

# now train the optimized model based on the best hyperparameters 
best_rf = RandomForestRegressor(
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=200,
    random_state=42)
best_rf.fit(X_train, y_train)

# predict on test set
y_pred_best_rf = best_rf.predict(X_test)

# evaluate performance
mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)
r2_best_rf = r2_score(y_test, y_pred_best_rf)

print(mae_best_rf, r2_best_rf)


# Feature Importance Analysis
import pandas as pd
import matplotlib.pyplot as plt

# get feature importances from the trained model , note that it does not get the important features rather the feature importances
importances = best_rf.feature_importances_

# Create a DataFrame for visualization
feature_names = X.columns
feat_importances = pd.DataFrame({'Feature': feature_names,
                                 'Importance': importances})
# Sort by importance
feat_importances = feat_importances.sort_values(by='Importance', ascending=False)

# plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feat_importances['Feature'], feat_importances['Importance'], color = 'skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.title('Feature Importance in Predicting Temperature')
plt.gca().invert_yaxis()
plt.show()

# What This Does:
# Extracts feature importances from the Random Forest model.
# Sorts them in descending order.
# Uses matplotlib to create a bar chart.
# Helps us understand which factors influence temperature the most.
'''
Why Are Some Features More Important?
Now, let's analyze why the model ranked features in this order.

Most Important Features:
1. min_temp -

If today's minimum temperature is high, the mean temperature will likely be high too.
Strong direct correlation.

2. max_temp - 

Similar reasoning: If the maximum temperature is high, mean temp will be high.

3. global_radiation - 

Measures sunlight energy hitting the ground (W/m²).
More radiation -> More heat -> Higher temperatures.
sunshine

More sunshine hours mean higher daytime temperatures.


Less Important Features:
4. pressure - 

Affects weather, but not directly related to temperature on a daily basis.

5. cloud_cover - 

Affects sunshine but has less direct impact on the daily mean temperature.

6. precipitation - 
Rain cools the atmosphere, but temperature isn't always directly linked to rainfall amount.

Conclusion:

1) Min and Max temperature are the strongest predictors because temperature is continuous and doesn't change drastically overnight.

2) Global radiation and sunshine affect heating, but not as much as min/max temp.

3) Cloud cover, precipitation, and pressure affect weather but not temperature directly.

'''

import joblib

# # save the model
joblib.dump(best_rf, "C:/Desktop/PRACTICE/LONDON WEATHER/temperature_predictor.pkl")

# load the model later
loaded_model = joblib.load("temperature_predictor.pkl")

# Define the feature names (same as the ones used in training)
feature_names = ['cloud_cover', 'sunshine', 'global_radiation', 
                 'max_temp', 'min_temp', 'precipitation', 'pressure']


# Example input data (modify values as needed)
sample_data = np.array([[3, 5, 120, 15, 7, 0.5, 1010]])

# Convert to DataFrame with column names
sample_df = pd.DataFrame(sample_data, columns=feature_names)

# Make prediction
prediction = loaded_model.predict(sample_df)

print("Predicted Mean Temperature:", prediction[0])




