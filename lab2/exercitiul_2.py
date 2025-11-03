import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score

X_train, y_train, X_test, y_test = generate_data_clusters(
    n_train=400, n_test=200, n_clusters=2, n_features=2, contamination=0.1, random_state=42
)

print("Dimensiune X_train:", X_train.shape)
print("Dimensiune y_train:", y_train.shape)
print("Dimensiune X_test:", X_test.shape)
print("Dimensiune y_test:", y_test.shape)

n_neighbors_values = [5, 10, 20, 30]

fig, axs = plt.subplots(len(n_neighbors_values), 4, figsize=(20, len(n_neighbors_values) * 5))

for idx, n_neighbors in enumerate(n_neighbors_values):
    knn_model = KNN(n_neighbors=n_neighbors, contamination=0.1)
    knn_model.fit(X_train)

    y_train_pred = knn_model.labels_  # 0 = inlier, 1 = outlier
    y_test_pred = knn_model.predict(X_test)

    ba_train = balanced_accuracy_score(y_train, y_train_pred)
    ba_test = balanced_accuracy_score(y_test, y_test_pred)
    print(f"n_neighbors: {n_neighbors} - Acuratețe balansată train: {ba_train:.2f}, test: {ba_test:.2f}")

    axs[idx, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', marker='o')
    axs[idx, 0].set_title(f"Real Train Labels (n_neighbors={n_neighbors})")

    axs[idx, 1].scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred, cmap='coolwarm', marker='o')
    axs[idx, 1].set_title(f"Predicted Train Labels (n_neighbors={n_neighbors})\nBA: {ba_train:.2f}")

    axs[idx, 2].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='o')
    axs[idx, 2].set_title("Real Test Labels")

    axs[idx, 3].scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred, cmap='coolwarm', marker='o')
    axs[idx, 3].set_title(f"Predicted Test Labels\nBA: {ba_test:.2f}")

plt.tight_layout()
plt.show()