import os
import json
import numpy as np
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

INPUT_DIR = "../../OutputJsons/DeathsAndSurvives"
MODEL_DIR = "../../AeonModel"
os.makedirs(MODEL_DIR, exist_ok=True)

def extract_rank_from_filename(filename):
    match = re.search(r"_RT(\d+)", filename)
    return int(match.group(1)) if match else None

def flatten_timeseries(data):
    flat = []
    time_keys = sorted(map(float, data.keys()))
    for t in time_keys:
        snapshot = data[str(t)]
        flat.extend([
            *snapshot.get("player_pos", [0, 0]),
            *snapshot.get("teammates_damage_taken", []),
            *snapshot.get("enemies_damage_taken", []),
            snapshot.get("player_damage_taken", 0),
            snapshot.get("player_level", 0),
            *snapshot.get("teammates_levels", []),
            *snapshot.get("enemies_levels", []),
            snapshot.get("player_hp_pct", 0),
            *snapshot.get("teammates_hp_pct", []),
            *snapshot.get("enemies_hp_pct", []),
        ])
    return flat

def extract_label_from_path(filepath):
    parts = filepath.replace("\\", "/").split("/")  # Ensure compatibility across OS
    if "Deaths" in parts:
        return 1
    elif "Survivals" in parts:
        return 0
    return None

def load_dataset(input_dir):
    X, y = [], []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            full_path = os.path.join(root, fname)
            label = extract_label_from_path(full_path)
            if label is None:
                print(f"⚠️ Skipping file due to missing label: {full_path}")  # <-- ADD THIS
                continue
            with open(full_path, 'r') as f:
                data = json.load(f)
            rank = extract_rank_from_filename(fname)
                        
            if rank is None:
                print(f"⚠️ Skipping file due to missing rank: {full_path}")
                continue

            features = flatten_timeseries(data)
            #features.append(rank)
            X.append(features)
            y.append(label)
    return X, y

def train():
    print("📥 Loading data...")
    X, y = load_dataset(INPUT_DIR)

    print("📊 Labels in training set:", set(y))
    print("🔢 Label counts:", {label: y.count(label) for label in set(y)})

    max_len = max(len(x) for x in X)
    X_padded = [x + [0] * (max_len - len(x)) for x in X]

    print("🔢 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_padded)

    print("🎯 Training model...")
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
    )
    #model = CalibratedClassifierCV(rf, method="isotonic", cv=3)

    model.fit(X_scaled, y)

    print("💾 Saving model...")
    joblib.dump(model, os.path.join(MODEL_DIR, "death_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "death_scaler.pkl"))
    print("✅ Death prediction model training complete!")

if __name__ == "__main__":
    train()
