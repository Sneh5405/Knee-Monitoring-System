import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "train_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "knee_angle_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

df = pd.read_csv(DATA_PATH)
window = 5

df['Mean'] = df['adc'].rolling(window).mean()
df['Std'] = df['adc'].rolling(window).std()
df['Max'] = df['adc'].rolling(window).max()
df['Min'] = df['adc'].rolling(window).min()
df['Range'] = df['Max'] - df['Min']
df['Delta'] = df['adc'].diff()
df['Slope'] = df['adc'].diff() / df['adc'].shift(1)

df = df.dropna()

X = df[['adc', 'Mean', 'Std', 'Delta', 'Slope', 'Range', 'Min', 'Max']].values
y = df['angle'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(32, activation='relu', input_shape=(8,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=150, batch_size=8, validation_split=0.1, verbose=1)

y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)

model.save(MODEL_PATH)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

fig, axes = plt.subplots(8, 1, figsize=(12, 14), sharex=True)
features_list = ['adc', 'Mean', 'Std', 'Delta', 'Slope', 'Range', 'Min', 'Max']
for i, name in enumerate(features_list):
    axes[i].plot(X[:, i], linewidth=0.8)
    axes[i].set_ylabel(name, fontsize=10)
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "features_plot.png"), dpi=150)
plt.close()

errors = y_test - y_pred
fig, ax = plt.subplots(figsize=(10, 5))
sorted_idx = np.argsort(y_test)
ax.plot(y_test[sorted_idx], errors[sorted_idx], marker="o")
ax.set_xlabel("Actual Angle")
ax.set_ylabel("Error (Actual - Predicted)")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "error_plot.png"), dpi=150)
plt.close()

adc_test = X_test[:, 0]
sorted_idx_adc = np.argsort(adc_test)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(adc_test[sorted_idx_adc], y_test[sorted_idx_adc], marker="o", markersize=4, linewidth=1.0, color="#E74C3C", label="Actual")
ax.plot(adc_test[sorted_idx_adc], y_pred[sorted_idx_adc], marker="s", markersize=4, linewidth=1.0, color="#27AE60", label="Predicted")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "actual_vs_predicted.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history.history["loss"], label="Training Loss", linewidth=1.2, color="#2980B9")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "model_loss.png"), dpi=150)
plt.close()