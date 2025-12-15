import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from pyod.utils.utility import standardizer
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# 1. Generare date
X_train, X_test, y_train, y_test = generate_data(
    n_train=300,
    n_test=200,
    n_features=3,
    contamination=0.15,
    random_state=42
)

X_train, X_test = standardizer(X_train, X_test)

def plot_3d_data(ax, X, y, title):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], c='blue', label='Inliers')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], c='red', label='Outliers')
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.legend()

def train_and_evaluate_ocsvm(kernel, X_train, X_test, y_train, y_test):
    model = OCSVM(kernel=kernel, contamination=0.15)
    model.fit(X_train)
    y_test_pred = model.predict(X_test)
    ba = balanced_accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, model.decision_function(X_test))
    return model, y_test_pred, ba, roc_auc

fig = plt.figure(figsize=(14, 10))

# --- 1. Ground Truth ---
ax1 = fig.add_subplot(221, projection='3d')
plot_3d_data(ax1, X_test, y_test, 'Ground Truth (Test Data)')

# --- 2. OCSVM Linear Kernel ---
ocsvm_linear, y_test_pred_linear, ba_linear, roc_auc_linear = train_and_evaluate_ocsvm(
    kernel='linear', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)
ax2 = fig.add_subplot(222, projection='3d')
plot_3d_data(ax2, X_test, y_test_pred_linear, 'Predicted (OCSVM Linear)')

# --- 3. OCSVM RBF Kernel ---
ocsvm_rbf, y_test_pred_rbf, ba_rbf, roc_auc_rbf = train_and_evaluate_ocsvm(
    kernel='rbf', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)
ax3 = fig.add_subplot(223, projection='3d')
plot_3d_data(ax3, X_test, y_test_pred_rbf, 'Predicted (OCSVM RBF)')

print(f"Linear Kernel: BA = {ba_linear:.2f}, ROC AUC = {roc_auc_linear:.2f}")
print(f"RBF Kernel:    BA = {ba_rbf:.2f}, ROC AUC = {roc_auc_rbf:.2f}")

# --- 4. DeepSVDD ---
deep_svdd_model = DeepSVDD(
    contamination=0.15, 
    hidden_neurons=[16, 8], 
    epochs=50, 
    verbose=1, 
    n_features=3 
)
deep_svdd_model.fit(X_train)
y_test_pred_deep_svdd = deep_svdd_model.predict(X_test)
ba_deep_svdd = balanced_accuracy_score(y_test, y_test_pred_deep_svdd)
roc_auc_deep_svdd = roc_auc_score(y_test, deep_svdd_model.decision_function(X_test))

ax4 = fig.add_subplot(224, projection='3d')
plot_3d_data(ax4, X_test, y_test_pred_deep_svdd, 'Predicted (DeepSVDD)')

print(f"DeepSVDD:      BA = {ba_deep_svdd:.2f}, ROC AUC = {roc_auc_deep_svdd:.2f}")

plt.tight_layout()
plt.show()