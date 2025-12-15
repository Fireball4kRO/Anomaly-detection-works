import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import os

if not os.path.exists('shuttle.mat'):
    print("EROARE: Fisierul 'shuttle.mat' nu este in acest folder!")
    exit()

data = loadmat('shuttle.mat')
X = data['X']
y = data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

print("-" * 30)
print("1. Fitting OCSVM...")
ocsvm_model = OCSVM(kernel='rbf', gamma='scale', nu=0.1)
ocsvm_model.fit(X_train_norm)

y_test_pred_ocsvm = ocsvm_model.predict(X_test_norm)
y_test_scores_ocsvm = ocsvm_model.decision_function(X_test_norm)

roc_auc_ocsvm = roc_auc_score(y_test, y_test_scores_ocsvm)
balanced_acc_ocsvm = balanced_accuracy_score(y_test, y_test_pred_ocsvm)

print(f"OCSVM Result -> BA: {balanced_acc_ocsvm:.4f}, ROC AUC: {roc_auc_ocsvm:.4f}")

print("-" * 30)
print("2. Fitting DeepSVDD (Testing Architectures)...")

architectures = [[16, 8], [32, 16, 8]]

for arch in architectures:
    print(f"\nTesting DeepSVDD architecture {arch}...")
    
    n_features = X_train_norm.shape[1]
    deep_svdd_model = DeepSVDD(
        contamination=0.1,
        hidden_neurons=arch,
        batch_size=64,
        epochs=20, 
        verbose=1,
        n_features=n_features
    )
    
    deep_svdd_model.fit(X_train_norm)

    y_test_pred_deepsvdd = deep_svdd_model.predict(X_test_norm)
    y_test_scores_deepsvdd = deep_svdd_model.decision_function(X_test_norm)
    
    roc_auc_deepsvdd = roc_auc_score(y_test, y_test_scores_deepsvdd)
    balanced_acc_deepsvdd = balanced_accuracy_score(y_test, y_test_pred_deepsvdd)

    print(f"Arch {arch} -> BA: {balanced_acc_deepsvdd:.4f}, ROC AUC: {roc_auc_deepsvdd:.4f}")