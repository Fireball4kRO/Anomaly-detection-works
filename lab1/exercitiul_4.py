import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

os.makedirs("results", exist_ok=True)

n = 1000
cont = 0.1
mu = np.array([2, -1])
Sigma = np.array([[2, 0.6], [0.6, 1]])
L = np.linalg.cholesky(Sigma)

n_out = int(n * cont)
n_in = n - n_out

X_in = np.random.randn(n_in, 2) @ L.T + mu
X_out = np.random.uniform(low=-6, high=6, size=(n_out, 2))
X = np.vstack([X_in, X_out])
y = np.array([0] * n_in + [1] * n_out)

z = (X - X.mean(axis=0)) / X.std(axis=0)
score = np.max(abs(z), axis=1)
thr = np.quantile(score, 1 - cont)
y_pred = (score > thr).astype(int)

tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
ba = balanced_accuracy_score(y, y_pred)
print(f"Ex4 -> Thr={thr:.3f}, TN={tn}, FP={fp}, FN={fn}, TP={tp}, BA={ba:.3f}")

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', s=15)
plt.title("Ex4 - MultiD Z-Score")
plt.xlabel("X1")
plt.ylabel("X2")
plt.savefig("results/ex4.png")
plt.close()

print("Ex4 gata -> results/ex4.png")
