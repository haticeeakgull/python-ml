from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import mean_squared_error

diabetes= load_diabetes()

X=diabetes.data
y=diabetes.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

ridge = Ridge()
ridge_param_grid= {"alpha":[0.1,1,10,100]}
ridge_grid_search= GridSearchCV(ridge,ridge_param_grid,cv=5)
ridge_grid_search.fit(X_train,y_train)
print("ridge en iyi parametreler:" , ridge_grid_search.best_params_)
print("ridge en iyi skor: " , ridge_grid_search.best_score_)

best_ridge_model=ridge_grid_search.best_estimator_
y_pred_ridge=best_ridge_model.predict(X_test)
ridge_mse= mean_squared_error(y_test,y_pred_ridge)
print("ridge mse: " ,ridge_mse)
print(" ")


lasso= Lasso()
lasso_param_grid={"alpha":[0.1,1,10,100]}
lasso_grid_search=GridSearchCV(lasso,lasso_param_grid ,cv=5)
lasso_grid_search.fit(X_train,y_train)
print("lasso en iyi parametreler:" , lasso_grid_search.best_params_)
print("lasso en iyi skor: " , lasso_grid_search.best_score_)
best_lasso_model=lasso_grid_search.best_estimator_
y_pred_lasso=best_lasso_model.predict(X_test)
lasso_mse= mean_squared_error(y_test,y_pred_lasso)
print("lasso mse:",lasso_mse)
