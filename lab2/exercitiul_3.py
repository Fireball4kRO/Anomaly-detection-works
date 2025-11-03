import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF

X, _ = make_blobs(n_samples=[200, 100], centers=[(-10, -10), (10, 10)], cluster_std=[2, 6], random_state=42)

contamination_rate = 0.07
n_neighbors_list = [5, 10, 20]  

knn_model = KNN(contamination=contamination_rate, n_neighbors=10)
lof_model = LOF(contamination=contamination_rate, n_neighbors=10)

knn_model.fit(X)
lof_model.fit(X)

knn_labels = knn_model.labels_
lof_labels = lof_model.labels_

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Subplot pentru KNN
axs[0].scatter(X[:, 0], X[:, 1], c=knn_labels, cmap='coolwarm', marker='o')
axs[0].set_title("KNN Detection")

# Subplot pentru LOF
axs[1].scatter(X[:, 0], X[:, 1], c=lof_labels, cmap='coolwarm', marker='o')
axs[1].set_title("LOF Detection")

plt.show()