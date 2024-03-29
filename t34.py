import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])

clf = KMeans(n_clusters=2)
clf.fit(X)

colors = ["c.","m.","y.","k.","r.","b."]

labels = clf.labels_
centroids = clf.cluster_centers_

for i in range(len(X)):
    plt.plot(X[i][0],X[i][1], colors[labels[i]], markersize=25)
plt.scatter(centroids[:,0],centroids[:,1], marker="x", linewidths=5, s=150)
plt.show()