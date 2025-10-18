import os
import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data

os.makedirs("results", exist_ok=True)

X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, n_features=2,contamination=0.1, random_state=42)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=20)
plt.title("Ex1 - Date de antrenare")
plt.xlabel("X1")
plt.ylabel("X2")
plt.savefig("results/ex1.png")
plt.close()

print("Ex1 gata -> results/ex1.png")
