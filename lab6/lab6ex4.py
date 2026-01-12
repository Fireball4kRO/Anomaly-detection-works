import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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

features = []
nodes = []

for node in G.nodes():
    ego = nx.ego_graph(G, node)
    Ni = len(ego.nodes()) - 1 
    Ei = ego.size()
    
    features.append([Ni, Ei])
    nodes.append(node)

features = np.array(features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

iso_forest = IsolationForest(contamination=0.05, random_state=42)
preds = iso_forest.fit_predict(X_scaled)
scores = iso_forest.decision_function(X_scaled)

anomalies_idx = np.where(preds == -1)[0]
normal_idx = np.where(preds == 1)[0]

print(f"Numar anomalii detectate: {len(anomalies_idx)}")

plt.figure(figsize=(10, 6))
plt.scatter(features[normal_idx, 0], features[normal_idx, 1], c='blue', alpha=0.6, label='Normal')
plt.scatter(features[anomalies_idx, 0], features[anomalies_idx, 1], c='red', s=50, label='Anomalii (Isolation Forest)')
plt.xlabel('Ni (Numar Noduri Vecine)')
plt.ylabel('Ei (Numar Muchii in Ego-Net)')
plt.title('Isolation Forest pe Structura Grafului')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(scores, bins=50, color='purple', alpha=0.7)
plt.title("Distributia Scorurilor de Anomalie")
plt.xlabel("Scor (mai mic = mai anormal)")
plt.ylabel("Numar Noduri")
plt.show()