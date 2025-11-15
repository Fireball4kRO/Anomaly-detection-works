import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
from pyod.utils.utility import standardizer

data = loadmat('shuttle.mat')

X = data['X']
y = data['y'].ravel()

n_splits = 10
ba_results = {'IForest': [], 'DIF': [], 'LODA': []}
roc_auc_results = {'IForest': [], 'DIF': [], 'LODA': []}

for _ in range(n_splits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=None, stratify=y)

    X_train, X_test = standardizer(X_train, X_test)

    iforest_model = IForest(contamination=0.02)
    iforest_model.fit(X_train)
    y_test_scores_if = iforest_model.decision_function(X_test)
    ba_if = balanced_accuracy_score(y_test, y_test_scores_if > np.median(y_test_scores_if))
    roc_auc_if = roc_auc_score(y_test, y_test_scores_if)

    dif_model = DIF(contamination=0.02, hidden_neurons=[32, 16], n_estimators=10)
    dif_model.fit(X_train)
    y_test_scores_dif = dif_model.decision_function(X_test)
    ba_dif = balanced_accuracy_score(y_test, y_test_scores_dif > np.median(y_test_scores_dif))
    roc_auc_dif = roc_auc_score(y_test, y_test_scores_dif)

    loda_model = LODA(contamination=0.02, n_bins=10)
    loda_model.fit(X_train)
    y_test_scores_loda = loda_model.decision_function(X_test)
    ba_loda = balanced_accuracy_score(y_test, y_test_scores_loda > np.median(y_test_scores_loda))
    roc_auc_loda = roc_auc_score(y_test, y_test_scores_loda)

    ba_results['IForest'].append(ba_if)
    roc_auc_results['IForest'].append(roc_auc_if)

    ba_results['DIF'].append(ba_dif)
    roc_auc_results['DIF'].append(roc_auc_dif)

    ba_results['LODA'].append(ba_loda)
    roc_auc_results['LODA'].append(roc_auc_loda)

for model in ['IForest', 'DIF', 'LODA']:
    print(f"Model: {model}")
    print(f"Mean Balanced Accuracy: {np.mean(ba_results[model]):.4f}")
    print(f"Mean ROC AUC: {np.mean(roc_auc_results[model]):.4f}")
    print("-" * 30)