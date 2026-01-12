import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans

file_path = 'ca-AstroPh.txt'
G = nx.Graph()

try:
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('#'):
                continue
            if i >= 1500: 
                break
            
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                if u != v:
                    G.add_edge(u, v)
except FileNotFoundError:
    print(f"Eroare: Nu gasesc fisierul {file_path}")
    exit()

adj_matrix = nx.to_numpy_array(G)

se = SpectralEmbedding(n_components=2)
embedding = se.fit_transform(adj_matrix)

kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(embedding)
centroids = kmeans.cluster_centers_

distances = []
for i, label in enumerate(labels):
    center = centroids[label]
    point = embedding[i]
    dist = np.linalg.norm(point - center)
    distances.append(dist)

distances = np.array(distances)
threshold = np.percentile(distances, 95)
outliers_idx = np.where(distances > threshold)[0]

plt.figure(figsize=(10, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c='gray', alpha=0.5, label='Noduri normale')
plt.title("Spectral Embedding")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', alpha=0.5, label='Clustere')
plt.scatter(embedding[outliers_idx, 0], embedding[outliers_idx, 1], c='red', s=50, label='Anomalii (Distanta mare)')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroizi')
plt.title("Spectral Embedding + K-Means + Anomalii")
plt.legend()
plt.show()