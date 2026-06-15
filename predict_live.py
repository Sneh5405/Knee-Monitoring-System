import numpy as np
import pandas as pd
import os
import pickle
from tensorflow.keras.models import load_model
from collections import Counter

WINDOW_SIZE = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "knee_angle_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
LIVE_DATA_PATH = os.path.join(BASE_DIR, "live_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "live_predictions.csv")

STANDING_MIN_MEAN = 110
SITTING_MIN_MEAN = 30
WALKING_RANGE_THRESH = 30

print("[1/4] Loading trained model...")
model = load_model(MODEL_PATH)
print("      Model loaded ✓")

print("[2/4] Loading live data & extracting features...")
df = pd.read_csv(LIVE_DATA_PATH)
adc = df["adc"].values
print(f"      {len(adc)} ADC readings loaded")

def extract_features(adc_values, window_size):
    df = pd.DataFrame({'adc': adc_values})

    df['Mean'] = df['adc'].rolling(window_size).mean()
    df['Std'] = df['adc'].rolling(window_size).std()
    df['Max'] = df['adc'].rolling(window_size).max()
    df['Min'] = df['adc'].rolling(window_size).min()
    df['Range'] = df['Max'] - df['Min']
    df['Delta'] = df['adc'].diff()
    df['Slope'] = df['adc'].diff() / df['adc'].shift(1)

    df = df.dropna()

    return df[['adc', 'Mean', 'Std', 'Delta', 'Slope', 'Range', 'Min', 'Max']].values

X_live = extract_features(adc, WINDOW_SIZE)
print(f"      Extracted {X_live.shape[0]} feature windows")

print("[3/4] Scaling features...")
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

X_live = scaler.transform(X_live)

print("[4/4] Predicting knee angles...")
predicted_angles = model.predict(X_live).flatten()
predicted_angles = np.clip(predicted_angles, 0, 140)

print(f"      Angle range: {predicted_angles.min():.1f}° – {predicted_angles.max():.1f}°")

def classify_activity(predicted_angles, window_size):
    activities = []
    for i in range(len(predicted_angles)):
        start = max(0, i - window_size + 1)
        window = predicted_angles[start:i+1]

        w_mean = np.mean(window)
        w_min = np.min(window)
        w_max = np.max(window)
        w_range = w_max - w_min

        if w_range >= WALKING_RANGE_THRESH:
            activities.append("Walking")
        elif w_mean >= STANDING_MIN_MEAN:
            activities.append("Standing")
        elif w_mean >= SITTING_MIN_MEAN:
            activities.append("Sitting")
        else:
            activities.append("Standing")

    return activities

activities = classify_activity(predicted_angles, WINDOW_SIZE)

activity_counts = Counter(activities)
print(f"      Activity counts: {dict(activity_counts)}")

results = pd.DataFrame({
    "window_index": np.arange(len(predicted_angles)),
    "predicted_angle": np.round(predicted_angles, 2),
    "activity": activities,
})

results.to_csv(OUTPUT_PATH, index=False)
print(f"\n📄 Results saved → {OUTPUT_PATH}")

print("\n── Preview (first 20 windows) ──")
print(results.head(20).to_string(index=False))

print(f"\n── Summary ──")
print(f"   Total windows:  {len(results)}")
print(f"   Mean angle:     {predicted_angles.mean():.1f}°")
print(f"   Angle std:      {predicted_angles.std():.1f}°")

for activity, count in sorted(activity_counts.items()):
    pct = 100.0 * count / len(activities)
    print(f"   {activity:10s}:  {count:4d} windows ({pct:.1f}%)")

print("\n Live prediction complete.")