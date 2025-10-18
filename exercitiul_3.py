import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

os.makedirs("results", exist_ok=True)

n = 1000
cont = 0.1
n_out = int(n * cont)
n_in = n - n_out

X_in = np.random.normal(0, 1, n_in)
X_out = np.random.normal(6, 1, n_out)
X = np.concatenate([X_in, X_out])
y = np.array([0] * n_in + [1] * n_out)

z = (X - np.mean(X)) / np.std(X)
thr = np.quantile(abs(z), 1 - cont)
y_pred = (abs(z) > thr).astype(int)

tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
ba = balanced_accuracy_score(y, y_pred)
print(f"Ex3 -> thr={thr:.3f}, TN={tn}, FP={fp}, FN={fn}, TP={tp}, BA={ba:.3f}")

plt.hist(z[y == 0], bins=30, alpha=0.7, label="normale")
plt.hist(z[y == 1], bins=30, alpha=0.7, label="anomalii")
plt.axvline(thr, color='red', linestyle='--', label='prag')
plt.axvline(-thr, color='red', linestyle='--')
plt.title("Ex3 - Distributia Z-score")
plt.xlabel("Z-score")
plt.ylabel("Frecventa")
plt.legend()
plt.savefig("results/ex3.png")
plt.close()

print("Ex3 gata/ex3.png")
