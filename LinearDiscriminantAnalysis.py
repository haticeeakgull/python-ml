"""
LDA, farklı sınıfların kümelerini birbirinden mümkün olduğunca uzağa iterken, her bir küme içindeki noktaları da birbirine mümkün olduğunca yaklaştıracak en iyi bakış açısını (yeni ekseni) bulur.


LDA'nın bulduğu yeni eksen, bir sınıflandırma algoritmasının (Lojistik Regresyon gibi) sınıfları ayırmak için kullanabileceği en kolay ve en temiz boyut olacaktır.
"""

from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

mnist=fetch_openml("mnist_784",version=1)

X=mnist.data
y=mnist.target.astype(int)
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda=lda.fit_transform(X,y)

plt.figure()
plt.scatter(X_lda[:,0],X_lda[:,1],c=y , cmap="tab10",alpha=0.6)
plt.title("LDA of MNIST Dataset")
plt.xlabel("lda1")
plt.ylabel("lda2")
plt.colorbar(label="Digits")
plt.show()

