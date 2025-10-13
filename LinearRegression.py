from sklearn.linear_model import LinearRegression

import numpy as np
import matplotlib.pyplot as plt

X= np.random.rand(100,1)
y=3+4*X+np.random.rand(100,1)


lin_reg=LinearRegression()
lin_reg.fit(X,y)

plt.scatter(X,y)
plt.plot(X,lin_reg.predict(X),color="red",alpha=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.show()