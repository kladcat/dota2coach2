import os
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import re

INPUT_DIR = "../OutputJsons/TimeSeriesForAverageTier/CKReplays"
MODEL_DIR = "../AeonModel"
os.makedirs(MODEL_DIR, exist_ok=True)


def extract_label_from_filename(filename):
    match = re.search(r"OFF_(\d{2})", filename)
    if not match:
        return "Unknown"
    
    tier_code = int(match.group(1))
    
    if 11 <= tier_code <= 15:
        return "Herald"
    elif 21 <= tier_code <= 25:
        return "Guardian"
    elif 31 <= tier_code <= 35:
        return "Crusader"
    elif 41 <= tier_code <= 45:
        return "Archon"
    elif 51 <= tier_code <= 55:
        return "Legend**s**"
    elif 61 <= tier_code <= 65:
        return "Ancient"
    elif 71 <= tier_code <= 75:
        return "Divine"
    elif tier_code >= 80:
        return "Immortal"
    else:
        return "Unknown"

def flatten_timeseries(data):
    flat = []
    time_keys = sorted(map(int, data.keys()))
    for t in time_keys:
        snapshot = data[str(t)]
        flat.extend([
            snapshot.get("networth", 0),
            snapshot.get("gold", 0),
            snapshot.get("xp", 0),
            snapshot.get("level", 0),
            snapshot.get("lh", 0),
            snapshot.get("denies", 0),
            snapshot.get("kills", 0),
            snapshot.get("deaths", 0),
            snapshot.get("assists", 0),
            snapshot.get("enemy_networth", 0) or 0,
        ])
    return flat

def load_dataset(input_dir):
    X = []
    y = []
    for fname in os.listdir(input_dir):
        if not fname.endswith(".json"):
            continue
        if "OFF_" not in fname:
            continue
        with open(os.path.join(input_dir, fname), 'r') as f:
            data = json.load(f)
        features = flatten_timeseries(data)
        label = extract_label_from_filename(fname)
        if label != "Unknown":
            X.append(features)
            y.append(label)
    return X, y

def train():
    print("📥 Loading data...")
    X, y = load_dataset(INPUT_DIR)
    max_len = max(len(x) for x in X)
    X_padded = [x + [0] * (max_len - len(x)) for x in X]

    print("🔢 Encoding and scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_padded)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("🎯 Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y_encoded)

    print("💾 Saving model...")
    joblib.dump(model, os.path.join(MODEL_DIR, "aeon_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "aeon_scaler.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "aeon_label_encoder.pkl"))
    print("✅ Training complete!")

if __name__ == "__main__":
    train()
