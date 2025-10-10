from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

cal_housing =fetch_california_housing()

X= cal_housing.data
y= cal_housing.target

X_train,X_test, y_train ,y_test=train_test_split(X,y,test_size=100,random_state=42)
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train,y_train)

y_pred = rf_reg.predict(X_test)
rmse = root_mean_squared_error(y_test,y_pred)

print("rmse:"  , rmse)
