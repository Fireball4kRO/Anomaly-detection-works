import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# modelul 1D
np.random.seed(0)
n_samples = 100
a, b = 2, 1

mu_values = [0, 0.5, 1.0]
sigma_values = [0.1, 1, 5]

fig, axs = plt.subplots(len(mu_values), len(sigma_values), figsize=(15, 10))

for i, mu in enumerate(mu_values):
    for j, sigma in enumerate(sigma_values):
        X = np.random.uniform(-10, 10, n_samples).reshape(-1, 1)
        
        noise = np.random.normal(mu, sigma, n_samples).reshape(-1, 1)
        
        y = a * X + b + noise

        reg = LinearRegression().fit(X, y)
        leverage_scores = (X - np.mean(X))**2 / np.var(X)

        high_leverage_points = leverage_scores > np.quantile(leverage_scores, 0.95)

        axs[i, j].scatter(X, y, color='blue', label='Puncte')
        axs[i, j].scatter(X[high_leverage_points], y[high_leverage_points], color='red', label='High Leverage')
        axs[i, j].plot(X, reg.predict(X), color='green', label='Linie model')
        

        axs[i, j].set_title(f'µ={mu}, σ²={sigma**2}')
        axs[i, j].legend()

plt.tight_layout()
plt.show()

# model 2D
np.random.seed(0)
n_samples_2d = 100
a, b, c = 2, -1, 5

X1 = np.random.uniform(-10, 10, n_samples_2d).reshape(-1, 1)
X2 = np.random.uniform(-10, 10, n_samples_2d).reshape(-1, 1)
noise_2d = np.random.normal(0, 1, n_samples_2d).reshape(-1, 1)

y_2d = a * X1 + b * X2 + c + noise_2d
X_2d = np.hstack([X1, X2])

reg_2d = LinearRegression().fit(X_2d, y_2d)
leverage_scores_2d = np.sum((X_2d - np.mean(X_2d, axis=0))**2 / np.var(X_2d, axis=0), axis=1)

high_leverage_points_2d = leverage_scores_2d > np.quantile(leverage_scores_2d, 0.95)

plt.figure(figsize=(8, 6))
plt.scatter(X1, X2, color='blue', label='Puncte')
plt.scatter(X1[high_leverage_points_2d], X2[high_leverage_points_2d], color='red', label='High Leverage')
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.title("Model 2D cu puncte de leverage ridicat")
plt.show()