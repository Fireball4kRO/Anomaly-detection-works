import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor

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
                    if G.has_edge(u, v):
                        G[u][v]['weight'] += 1
                    else:
                        G.add_edge(u, v, weight=1)

except FileNotFoundError:
    print(f"Eroare: Nu gasesc fisierul {file_path}")
    exit()

print(f"Graf incarcat: {len(G.nodes())} noduri, {len(G.edges())} muchii.")

features = []
nodes = []

for node in G.nodes():
    ego = nx.ego_graph(G, node)
    
    Ni = len(ego.nodes()) - 1 
    Ei = ego.size(weight=None)
    Wi = ego.size(weight='weight')
    
    lambda_1 = 0
    if Ni > 0:
        adj = nx.to_numpy_array(ego, weight='weight')
        if adj.shape[0] > 0:
            vals = np.linalg.eigvals(adj)
            lambda_1 = np.max(np.abs(vals))
        
    features.append([Ni, Ei, Wi, lambda_1])
    nodes.append(node)

features = np.array(features)
nodes = np.array(nodes)

log_Ni = np.log(features[:, 0] + 1).reshape(-1, 1)
log_Ei = np.log(features[:, 1] + 1).reshape(-1, 1)

reg = LinearRegression()
reg.fit(log_Ni, log_Ei)
pred_log_Ei = reg.predict(log_Ni)

scores = []
for k in range(len(features)):
    y_real = log_Ei[k][0]
    y_pred = pred_log_Ei[k][0]
    
    score = (max(y_real, y_pred) / (min(y_real, y_pred) + 1e-5)) * np.log(abs(y_real - y_pred) + 1)
    scores.append(score)

scores = np.array(scores)
top_idx = np.argsort(scores)[-10:] 
top_nodes = nodes[top_idx]

plt.figure(figsize=(10, 6))
plt.scatter(log_Ni, log_Ei, alpha=0.5, label='Noduri normale')
plt.scatter(log_Ni[top_idx], log_Ei[top_idx], c='r', label='Top 10 Anomalii')
plt.plot(log_Ni, pred_log_Ei, c='orange', linewidth=2, label='Regresie')
plt.xlabel('log(Ni)')
plt.ylabel('log(Ei)')
plt.title('OddBall')
plt.legend()
plt.show()

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=42)
node_colors = ['red' if n in top_nodes else 'blue' for n in G.nodes()]
node_sizes = [50 if n in top_nodes else 20 for n in G.nodes()]
nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, with_labels=False)
plt.title("Graf cu Top 10 Anomalii")
plt.show()

X_lof = features[:, :2] 
lof = LocalOutlierFactor(n_neighbors=10)
lof_preds = lof.fit_predict(X_lof)
lof_scores = -lof.negative_outlier_factor_

plt.figure(figsize=(10, 6))
plt.scatter(X_lof[:, 0], X_lof[:, 1], c=lof_scores, cmap='coolwarm', s=50)
plt.colorbar(label='Scor LOF')
plt.xlabel('Ni')
plt.ylabel('Ei')
plt.title('LOF (Ni, Ei)')
plt.show()