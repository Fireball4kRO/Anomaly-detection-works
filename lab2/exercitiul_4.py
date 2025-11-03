import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from pyod.utils.utility import standardizer
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.combination import average, maximization
from sklearn.metrics import balanced_accuracy_score

data = loadmat("lab2/cardio.mat") 
X = data['X']
y = data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

X_train_norm, X_test_norm = standardizer(X_train, X_test)

contamination_rate = 0.1

n_neighbors_values = range(30, 130, 10)  #  n_neighbors de la 30 la 120
train_scores = []
test_scores = []

for n_neighbors in n_neighbors_values:
    model = KNN(contamination=contamination_rate, n_neighbors=n_neighbors)
    
    model.fit(X_train_norm)
    
    train_scores.append(model.decision_scores_)
    test_scores.append(model.decision_function(X_test_norm))
    
    train_pred = model.labels_
    test_pred = model.predict(X_test_norm)
    
    ba_train = balanced_accuracy_score(y_train, train_pred)
    ba_test = balanced_accuracy_score(y_test, test_pred)
    
    print(f"n_neighbors: {n_neighbors} - BA train: {ba_train:.2f}, BA test: {ba_test:.2f}")


train_scores_norm, test_scores_norm = standardizer(np.array(train_scores).T, np.array(test_scores).T)

average_scores = average(test_scores_norm)
max_scores = maximization(test_scores_norm)

threshold_avg = np.quantile(average_scores, 1 - contamination_rate)
threshold_max = np.quantile(max_scores, 1 - contamination_rate)

final_pred_avg = (average_scores > threshold_avg).astype(int)
final_pred_max = (max_scores > threshold_max).astype(int)


ba_avg = balanced_accuracy_score(y_test, final_pred_avg)
ba_max = balanced_accuracy_score(y_test, final_pred_max)

print(f"Balanced accuracy - Average: {ba_avg:.2f}")
print(f"Balanced accuracy - Maximization: {ba_max:.2f}")