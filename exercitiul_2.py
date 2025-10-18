import os
import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, auc

os.makedirs("results", exist_ok=True)

X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, n_features=2,contamination=0.1, random_state=42)

model = KNN(contamination=0.1)
model.fit(X_train)

y_pred = model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
ba = balanced_accuracy_score(y_test, y_pred)
print(f"Ex2 -> TN={tn}, FP={fp}, FN={fn}, TP={tp}, BA={ba:.3f}")

scores = model.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title("Ex2 - ROC Curve (KNN)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.savefig("results/ex2.png")
plt.close()

print("Ex2 gata -> results/ex2.png")
