"""
t-SNE'nin ana işlevi, bir veri setindeki yüksek boyutlu (çok fazla özellikli) karmaşık ilişkileri alıp, bunları koruyarak 2 veya 3 boyutlu bir grafiğe indirgemektir.

Ne Yapar?
Yerel Yapıyı Korur: Birbirine yüksek boyutta çok yakın olan veri noktalarını, düşük boyuttaki grafikte de yakın tutmaya odaklanır.

Kümelemeyi Ortaya Çıkarır: Veri setinde doğal olarak var olan küme (cluster) yapılarını, grafik üzerinde belirgin ve net bir şekilde ayrılmış adalar veya gruplar halinde gösterir.
"""


from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

mnist = fetch_openml("mnist_784",version=1)

X=mnist.data
y=mnist.target.astype(int)

tsne= TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

plt.figure()
plt.scatter(X_tsne[:,0],X_tsne[:,1],c=y , cmap="tab10",alpha=0.6)
plt.title("TSNE of Mnist dataset")
plt.xlabel("T-SNE Dimension 1")
plt.ylabel("T-SNE Dimension 2")
plt.show()

