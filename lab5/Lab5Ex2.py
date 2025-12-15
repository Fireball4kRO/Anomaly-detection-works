import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
import os

if not os.path.exists('shuttle.mat'):
    print("EROARE: Nu gasesc shuttle.mat in folderul curent!")
    exit()

data = loadmat("shuttle.mat")
X = data["X"]
y = data["y"].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

contamination_rate = np.mean(y_train == 1)

print("Fitting PyOD PCA...")
pca = PCA(contamination=contamination_rate)
pca.fit(X_train_norm)

y_train_pred_pca = pca.labels_
y_test_pred_pca = pca.predict(X_test_norm)

ba_train_pca = balanced_accuracy_score(y_train, y_train_pred_pca)
ba_test_pca = balanced_accuracy_score(y_test, y_test_pred_pca)

print(f"PCA Train Balanced Accuracy: {ba_train_pca:.4f}")
print(f"PCA Test Balanced Accuracy:  {ba_test_pca:.4f}")

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, label="Individual Variance")
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label="Cumulative Variance")
plt.xlabel("Componente")
plt.ylabel("Varianta")
plt.title("PCA Explained Variance")
plt.legend()
plt.show()

print("Fitting PyOD KPCA (Kernel PCA)...")
kpca = KPCA(contamination=contamination_rate)
kpca.fit(X_train_norm)

y_train_pred_kpca = kpca.labels_
y_test_pred_kpca = kpca.predict(X_test_norm)

ba_train_kpca = balanced_accuracy_score(y_train, y_train_pred_kpca)
ba_test_kpca = balanced_accuracy_score(y_test, y_test_pred_kpca)

print(f"KPCA Train Balanced Accuracy: {ba_train_kpca:.4f}")
print(f"KPCA Test Balanced Accuracy:  {ba_test_kpca:.4f}")