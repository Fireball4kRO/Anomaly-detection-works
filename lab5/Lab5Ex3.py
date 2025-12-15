import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os

if not os.path.exists('shuttle.mat'):
    print("EROARE: Lipseste shuttle.mat!")
    exit()

data = loadmat("shuttle.mat")
X = data["X"]
y = data["y"].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            Dense(8, activation="relu"),
            Dense(5, activation="relu"),
            Dense(3, activation="relu")
        ])
        self.decoder = tf.keras.Sequential([
            Dense(5, activation="relu"),
            Dense(8, activation="relu"),
            Dense(X_train_norm.shape[1], activation="sigmoid")
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder()
autoencoder.compile(optimizer="adam", loss="mse")

history = autoencoder.fit(
    X_train_norm, X_train_norm,
    epochs=100,
    batch_size=1024,
    validation_data=(X_test_norm, X_test_norm),
    verbose=1
)

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

train_reconstructions = autoencoder.predict(X_train_norm)
test_reconstructions = autoencoder.predict(X_test_norm)

train_errors = np.mean(np.square(X_train_norm - train_reconstructions), axis=1)
test_errors = np.mean(np.square(X_test_norm - test_reconstructions), axis=1)

contamination_rate = np.mean(y_train == 1)
threshold = np.quantile(train_errors, 1 - contamination_rate)

y_train_pred = (train_errors > threshold).astype(int)
y_test_pred = (test_errors > threshold).astype(int)

ba_train = balanced_accuracy_score(y_train, y_train_pred)
ba_test = balanced_accuracy_score(y_test, y_test_pred)

print(f"Balanced Accuracy (Train): {ba_train:.4f}")
print(f"Balanced Accuracy (Test):  {ba_test:.4f}")