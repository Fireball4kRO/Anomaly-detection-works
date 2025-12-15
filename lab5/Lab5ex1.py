import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42) 
mean_vector = [5, 10, 2]
cov_matrix = [[3, 2, 2], [2, 10, 1], [2, 1, 2]]
data = np.random.multivariate_normal(mean=mean_vector, cov=cov_matrix, size=500)

data_centered = data - np.mean(data, axis=0)

cov_matrix_data = np.cov(data_centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix_data)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

transformed_data = np.dot(data_centered, eigenvectors)

cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(eigenvalues) + 1), eigenvalues / np.sum(eigenvalues), label="Individual Variance")
plt.step(range(1, len(eigenvalues) + 1), cumulative_variance, where='mid', label="Cumulative Variance")
plt.xlabel("Componente Principale")
plt.ylabel("Varianta Explicata")
plt.title("Varianta explicata de componentele principale")
plt.legend()
plt.show()

contamination_rate = 0.1

third_component = transformed_data[:, 2]
third_component_threshold = np.quantile(third_component, [contamination_rate, 1 - contamination_rate])
outliers_3rd = (third_component < third_component_threshold[0]) | (third_component > third_component_threshold[1])

second_component = transformed_data[:, 1]
second_component_threshold = np.quantile(second_component, [contamination_rate, 1 - contamination_rate])
outliers_2nd = (second_component < second_component_threshold[0]) | (second_component > second_component_threshold[1])

normalized_distances = np.sqrt(np.sum((transformed_data / np.std(transformed_data, axis=0)) ** 2, axis=1))
distance_threshold = np.quantile(normalized_distances, 1 - contamination_rate)
outliers_distance = normalized_distances > distance_threshold

fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c=np.where(outliers_3rd, 'r', 'b'))
ax1.set_title("Outliers (Componenta 3)")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.set_zlabel("Feature 3")

ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(data[:, 0], data[:, 1], data[:, 2], c=np.where(outliers_2nd, 'r', 'b'))
ax2.set_title("Outliers (Componenta 2)")
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Feature 2")
ax2.set_zlabel("Feature 3")

ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(data[:, 0], data[:, 1], data[:, 2], c=np.where(outliers_distance, 'r', 'b'))
ax3.set_title("Outliers (Distanta Normalizata)")
ax3.set_xlabel("Feature 1")
ax3.set_ylabel("Feature 2")
ax3.set_zlabel("Feature 3")

plt.tight_layout()
plt.show()