import json
import joblib
import numpy as np
import os
import sys

MODEL_DIR = "../../AeonModel"
MODEL_PATH = os.path.join(MODEL_DIR, "death_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "death_scaler.pkl")

def flatten_timeseries(data):
    flat = []
    time_keys = sorted(map(float, data.keys()))
    for t in time_keys:
        snapshot = data[str(t)]
        flat.extend([
            snapshot.get("player_damage_taken", 0),
            snapshot.get("player_level", 0),
            snapshot.get("player_hp_pct", 0),
            *snapshot.get("teammates_damage_taken", []),
            *snapshot.get("teammates_levels", []),
            *snapshot.get("teammates_hp_pct", []),
            *snapshot.get("enemies_damage_taken", []),
            *snapshot.get("enemies_levels", []),
            *snapshot.get("enemies_hp_pct", [])
        ])
    return flat

def predict(path_to_file):
    with open(path_to_file, "r") as f:
        data = json.load(f)

    flat = flatten_timeseries(data)
    flat = np.array(flat).reshape(1, -1)

    # Pad if needed
    model = joblib.load(MODEL_PATH)
    expected_len = model.n_features_in_
    if flat.shape[1] < expected_len:
        flat = np.pad(flat, ((0, 0), (0, expected_len - flat.shape[1])), mode='constant')

    scaler = joblib.load(SCALER_PATH)
    flat_scaled = scaler.transform(flat)

    proba = model.predict_proba(flat_scaled)[0]
    pred = model.predict(flat_scaled)[0]

    classes = model.classes_

    print("🎯 Predicted Outcome:", "Death" if pred == 1 else "Survival")
    print("📊 Confidence:")
    for i, cls in enumerate(classes):
        label = "Death" if cls == 1 else "Survival"
        print(f" - {label:<9}: {proba[i]*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 predict_death_event.py path/to/event.json")
        sys.exit(1)

    predict(sys.argv[1])
