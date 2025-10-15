"""
PCA (Temel Bileşen Analizi), bir veri setindeki boyut sayısını (özellik sayısını) azaltmak için kullanılan istatistiksel bir tekniktir.

Ne Yapar?

Veri setindeki en fazla varyansı (bilgiyi) taşıyan yeni, bağımsız eksenler (bileşenler) bulur.

İlk bileşen, en çok bilgiyi içerir. İkinci bileşen, kalan bilgiden en çoğunu içerir, vb.

Modelinizi eğitmek için orijinal özelliklerin tamamını değil, bu en önemli bileşenlerin sadece ilk birkaçını seçerek boyutu büyük ölçüde düşürürsünüz.

Temel Amaç: Bilgi kaybını minimuma indirerek, hesaplama maliyetini düşürmek ve veriyi görselleştirmeyi kolaylaştırmaktır.
"""

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

iris = load_iris()
X= iris.data
y = iris.target

pca = PCA(n_components=2)

X_pca=pca.fit_transform(X)

plt.figure()
for i in range(len(iris.target_names)):
    plt.scatter(X_pca[y==i,0],X_pca[y==i,1],label=iris.target_names[i])

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Iris Dataset")
plt.show()