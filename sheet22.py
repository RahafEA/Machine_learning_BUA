from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso


df=pd.read_csv(r"C:\Users\rahaf\OneDrive\سطح المكتب\تالتة\ML\employee_salary_data.csv")

X = df[['YearsExperience', 'Age', 'WorkingHoursPerWeek']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = linear_model.predict(X_train_scaled)
y_test_pred = linear_model.predict(X_test_scaled)

# MSE
linear_train_mse = mean_squared_error(y_train, y_train_pred)
linear_test_mse = mean_squared_error(y_test, y_test_pred)

print("Linear Regression Train MSE:", linear_train_mse)
print("Linear Regression Test MSE:", linear_test_mse)


##################################################
##################################################

ridge = Ridge(alpha=0.01)
ridge.fit(X_train_scaled, y_train)
y_train_pred = ridge.predict(X_train_scaled)
y_test_pred = ridge.predict(X_test_scaled)
ridge_train_mse = mean_squared_error(y_train, y_train_pred)
ridge_test_mse = mean_squared_error(y_test, y_test_pred)
print("Ridge Regression (alpha=0.01)")
print("Train MSE:", ridge_train_mse)
print("Test MSE:", ridge_test_mse)



ridge = Ridge(alpha=0.1)
ridge.fit(X_train_scaled, y_train)
y_train_pred = ridge.predict(X_train_scaled)
y_test_pred = ridge.predict(X_test_scaled)
ridge_train_mse = mean_squared_error(y_train, y_train_pred)
ridge_test_mse = mean_squared_error(y_test, y_test_pred)
print("Ridge Regression (alpha=0.1)")
print("Train MSE:", ridge_train_mse)
print("Test MSE:", ridge_test_mse)


ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_train_pred = ridge.predict(X_train_scaled)
y_test_pred = ridge.predict(X_test_scaled)
ridge_train_mse = mean_squared_error(y_train, y_train_pred)
ridge_test_mse = mean_squared_error(y_test, y_test_pred)
print("Ridge Regression (alpha=1)")
print("Train MSE:", ridge_train_mse)
print("Test MSE:", ridge_test_mse)

ridge = Ridge(alpha=10)
ridge.fit(X_train_scaled, y_train)
y_train_pred = ridge.predict(X_train_scaled)
y_test_pred = ridge.predict(X_test_scaled)
ridge_train_mse = mean_squared_error(y_train, y_train_pred)
ridge_test_mse = mean_squared_error(y_test, y_test_pred)
print("Ridge Regression (alpha=10)")
print("Train MSE:", ridge_train_mse)
print("Test MSE:", ridge_test_mse)


###############################
###############################

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Lasso Regression (alpha=0.01)")
print("Train MSE:", train_mse)
print("Test MSE:", test_mse)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Lasso Regression (alpha=0.1)")
print("Train MSE:", train_mse)
print("Test MSE:", test_mse)

lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Lasso Regression (alpha=1)")
print("Train MSE:", train_mse)
print("Test MSE:", test_mse)

lasso = Lasso(alpha=10)
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Lasso Regression (alpha=10)")
print("Train MSE:", train_mse)
print("Test MSE:", test_mse)

