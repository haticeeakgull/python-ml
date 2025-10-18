from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

iris=load_iris()
X=iris.data
y=iris.target 

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

knn = KNeighborsClassifier()
knn_param_grid={"n_neighbors": np.arange(2,31)}

knn_grid_search = GridSearchCV(knn,knn_param_grid)
knn_grid_search.fit(X_train,y_train)
print("knn grid search best parameters: " , knn_grid_search.best_params_)
print("knn grid search best accuracy: " , knn_grid_search.best_score_)

knn_random_search = RandomizedSearchCV(knn,knn_param_grid,n_iter=10)
knn_random_search.fit(X_train,y_train)
print("knn random search best parameters: " , knn_random_search.best_params_)
print("knn random search best accuracy: " , knn_random_search.best_score_)

