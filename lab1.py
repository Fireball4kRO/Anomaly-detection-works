# ===== Anomaly Detection - Lab 1 =====
# Autor: <numele tău>
# Exercițiile 1-4

import numpy as np
from matplotlib import pyplot as plt
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, auc

RNG = np.random.default_rng(42)

# ---------- Ex. 1 ----------
def ex1_generate_and_plot(contamination=0.1):
    X_train, X_test, y_train, y_test = generate_data(
        n_train=400, n_test=100, n_features=2,
        contamination=contamination, random_state=42
    )

    plt.figure()
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=20)
    plt.title("Ex.1 — Training data with outliers")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    return X_train, X_test, y_train, y_test

# ---------- Ex. 2 ----------
def ex2_knn_metrics(X_train, X_test, y_train, y_test, contamination=0.1):
    clf = KNN(contamination=contamination)
    clf.fit(X_train)

    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    ba = balanced_accuracy_score(y_test, y_pred)

    print(f"Ex.2 — TN={tn} FP={fp} FN={fn} TP={tp} | BA={ba:.3f}")

    scores = clf.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.title("Ex.2 — ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()

# ---------- Ex. 3 ----------
def ex3_univariate_zscore(contamination=0.1, n_train=1000):
    X_in = np.random.normal(0, 1, int(n_train * (1 - contamination)))
    X_out = np.random.normal(6, 1, int(n_train * contamination))
    X = np.concatenate([X_in, X_out])
    y = np.array([0] * len(X_in) + [1] * len(X_out))

    z = (X - np.mean(X)) / np.std(X)
    thr = np.quantile(np.abs(z), 1 - contamination)
    y_pred = (np.abs(z) > thr).astype(int)

    ba = balanced_accuracy_score(y, y_pred)
    print(f"Ex.3 — Z-score threshold={thr:.3f} | BA={ba:.3f}")

# ---------- Ex. 4 ----------
def ex4_multidim_zscore(contamination=0.1, n_train=1000):
    mu = np.array([2, -1])
    Sigma = np.array([[2, 0.6], [0.6, 1]])
    L = np.linalg.cholesky(Sigma)
    n_in = int(n_train * (1 - contamination))
    n_out = n_train - n_in
    X_in = np.random.randn(n_in, 2) @ L.T + mu
    X_out = np.random.uniform(low=-6, high=6, size=(n_out, 2))
    X = np.vstack((X_in, X_out))
    y = np.array([0] * n_in + [1] * n_out)
    z = (X - X.mean(axis=0)) / X.std(axis=0)
    score = np.max(np.abs(z), axis=1)
    thr = np.quantile(score, 1 - contamination)
    y_pred = (score > thr).astype(int)
    ba = balanced_accuracy_score(y, y_pred)
    print(f"Ex.4 — Multidim Z-score | Thr={thr:.3f} | BA={ba:.3f}")

if __name__ == "__main__":
    Xtr, Xte, ytr, yte = ex1_generate_and_plot()
    ex2_knn_metrics(Xtr, Xte, ytr, yte)
    ex3_univariate_zscore()
    ex4_multidim_zscore()
