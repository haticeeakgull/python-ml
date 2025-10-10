from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score 

oli= fetch_olivetti_faces()

plt.figure()
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(oli.images[i],cmap ="gray")
    plt.axis("off")

plt.show()

X= oli.data
y=oli.target

X_train , X_test , y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

rf_claf = RandomForestClassifier(n_estimators=120,random_state=42)

rf_claf.fit(X_train,y_train)
y_pred = rf_claf.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print("accuracy: " , accuracy)