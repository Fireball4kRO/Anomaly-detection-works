import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from numpy.random import multivariate_normal, uniform

X_train, _ = make_blobs(n_samples=500, centers=1, n_features=2, random_state=42)

projections = [multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]]).reshape(-1, 1)
               for _ in range(5)]
projections = [v / np.linalg.norm(v) for v in projections]  # Normalizare la lungime unitară

def compute_anomaly_score(data, projections, bins=10, range_limit=(-5, 5)):
    scores = []
    for v in projections:
        projected_data = data @ v
        
        hist, bin_edges = np.histogram(projected_data, bins=bins, range=range_limit, density=True)
        bin_probs = hist * np.diff(bin_edges)  # Probabilitățile per bin
        
        bin_indices = np.digitize(projected_data, bins=bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_probs) - 1)  # Corectare pentru margini
        scores.append(bin_probs[bin_indices])
    
    scores = np.mean(np.column_stack(scores), axis=1)
    return scores

anomaly_scores_train = compute_anomaly_score(X_train, projections)

X_test = uniform(-3, 3, (500, 2))
anomaly_scores_test = compute_anomaly_score(X_test, projections)

plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=anomaly_scores_test, cmap='viridis')
plt.colorbar(label="Anomaly Score")
plt.title("Anomaly Score Map (Test Data)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

for bins in [5, 10, 20]:
    anomaly_scores_test = compute_anomaly_score(X_test, projections, bins=bins)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=anomaly_scores_test, cmap='viridis')
    plt.colorbar(label="Anomaly Score")
    plt.title(f"Anomaly Score Map (Test Data) with {bins} bins")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()