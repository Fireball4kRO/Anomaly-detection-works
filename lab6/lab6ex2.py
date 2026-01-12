import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

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

cliques = list(nx.find_cliques(G))
clique_sizes = [len(c) for c in cliques]

plt.figure(figsize=(10, 6))
plt.hist(clique_sizes, bins=20, color='skyblue', edgecolor='black')
plt.title("Distributia marimii clicilor")
plt.xlabel("Marime Clica")
plt.ylabel("Numar de Clici")
plt.show()

max_clique_nodes = max(cliques, key=len)
print(f"Marimea clicei maximale: {len(max_clique_nodes)}")
print(f"Nodurile clicei maximale: {max_clique_nodes}")

try:
    conductance = nx.conductance(G, max_clique_nodes)
    print(f"Conductanta clicei maximale: {conductance:.4f}")
except Exception as e:
    print("Nu s-a putut calcula conductanta (posibil graf deconectat sau nod izolat).")

clique_subgraph = G.subgraph(max_clique_nodes)
clust_clique = nx.average_clustering(clique_subgraph)
degrees_clique = [d for n, d in clique_subgraph.degree()]
avg_deg_clique = np.mean(degrees_clique) if degrees_clique else 0

non_clique_nodes = [n for n in G.nodes() if n not in max_clique_nodes]
if len(non_clique_nodes) >= 10:
    random_nodes = random.sample(non_clique_nodes, 10)
    random_subgraph = G.subgraph(random_nodes)
    clust_random = nx.average_clustering(random_subgraph)
    degrees_random = [d for n, d in random_subgraph.degree()]
    avg_deg_random = np.mean(degrees_random) if degrees_random else 0
else:
    clust_random = 0
    avg_deg_random = 0

print("-" * 40)
print(f"{'Metrica':<25} | {'Clica Maximala':<15} | {'Random 10 Noduri':<15}")
print("-" * 40)
print(f"{'Avg Clustering Coeff':<25} | {clust_clique:<15.4f} | {clust_random:<15.4f}")
print(f"{'Avg Degree':<25} | {avg_deg_clique:<15.4f} | {avg_deg_random:<15.4f}")
print("-" * 40)