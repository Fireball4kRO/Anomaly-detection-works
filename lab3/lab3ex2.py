import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA


X_train, _ = make_blobs(n_samples=1000, centers=[[10, 0], [0, 10]], cluster_std=1, n_features=2, random_state=42)

n_outliers = 20
extreme_points = np.random.uniform(-30, 40, (n_outliers, 2)) 
X_train = np.vstack([X_train, extreme_points])  

X_test = np.random.uniform(-10, 20, (1000, 2))

contamination_rate = 0.02
iforest_model = IForest(contamination=contamination_rate)
iforest_model.fit(X_train)
iforest_scores = iforest_model.decision_function(X_test)

dif_model = DIF(contamination=contamination_rate, hidden_neurons=[64, 32])
dif_model.fit(X_train)
dif_scores = dif_model.decision_function(X_test)

loda_model = LODA(contamination=contamination_rate, n_bins=50)  
loda_model.fit(X_train)
loda_scores = loda_model.decision_function(X_test)

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sc = axs[0].scatter(X_test[:, 0], X_test[:, 1], c=iforest_scores, cmap='viridis')
axs[0].set_title("Scoruri pentru Isolation Forest Anomaly")
fig.colorbar(sc, ax=axs[0])

sc = axs[1].scatter(X_test[:, 0], X_test[:, 1], c=dif_scores, cmap='viridis')
axs[1].set_title("Scoruri pentru Deep Isolation Forest Anomaly")
fig.colorbar(sc, ax=axs[1])

sc = axs[2].scatter(X_test[:, 0], X_test[:, 1], c=loda_scores, cmap='viridis')
axs[2].set_title("Scoruri pentru LODA Anomaly (n_bins=50)")
fig.colorbar(sc, ax=axs[2])

plt.show()

X_train_3d, _ = make_blobs(n_samples=1000, centers=[[0, 10, 0], [10, 0, 10]], cluster_std=1, n_features=3, random_state=42)

extreme_points_3d = np.random.uniform(-30, 40, (n_outliers, 3))
X_train_3d = np.vstack([X_train_3d, extreme_points_3d])

X_test_3d = np.random.uniform(-10, 20, (1000, 3))

iforest_model.fit(X_train_3d)
iforest_scores_3d = iforest_model.decision_function(X_test_3d)

dif_model.fit(X_train_3d)
dif_scores_3d = dif_model.decision_function(X_test_3d)

loda_model.fit(X_train_3d)
loda_scores_3d = loda_model.decision_function(X_test_3d)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131, projection='3d')
sc = ax1.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], c=iforest_scores_3d, cmap='viridis')
ax1.set_title("Scoruri pentru Isolation Forest Anomaly")
fig.colorbar(sc, ax=ax1)

ax2 = fig.add_subplot(132, projection='3d')
sc = ax2.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], c=dif_scores_3d, cmap='viridis')
ax2.set_title("Scoruri pentru Deep Isolation Forest Anomaly")
fig.colorbar(sc, ax=ax2)

ax3 = fig.add_subplot(133, projection='3d')
sc = ax3.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], c=loda_scores_3d, cmap='viridis')
ax3.set_title("Scoruri pentru LODA Anomaly (n_bins=50)")
fig.colorbar(sc, ax=ax3)

plt.show()