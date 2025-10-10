from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

iris = load_iris()
X=iris.data
y=iris.target

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=42)
nb_claf= GaussianNB()
nb_claf.fit(X_train,y_train)
y_pred = nb_claf.predict(X_test)
print(classification_report(y_test,y_pred))