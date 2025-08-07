import os
import json
import numpy as np
import joblib
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# === Config ===
INPUT_DIR = "../../OutputJsons/DeathsAndSurvives"  # Root folder containing Deaths/ and Survivals/
OUTPUT_DIR = "../../AeonModel"

EXPECTED_FEATURES = 352  # Each sample should be padded to this length

def flatten_timeseries(data):
    flat = []
    for t in sorted(map(float, data.keys())):
        entry = data[str(t)]
        flat.extend([
            *entry.get("player_pos", [0, 0]),
            *entry.get("teammates_damage_taken", []),
            *entry.get("enemies_damage_taken", []),
            entry.get("player_damage_taken", 0),
            entry.get("player_level", 0),
            *entry.get("teammates_levels", []),
            *entry.get("enemies_levels", []),
            entry.get("player_hp_pct", 0),
            *entry.get("teammates_hp_pct", []),
            *entry.get("enemies_hp_pct", []),
        ])
    return flat

def pad_to_length(flat, length):
    if len(flat) < length:
        return flat + [0] * (length - len(flat))
    return flat

def load_dataset(root_dir):
    X, y = [], []

    # Traverse each match folder (e.g., 8379966117_1515765073_RT41)
    for match_folder in os.listdir(root_dir):
        match_path = os.path.join(root_dir, match_folder)
        if not os.path.isdir(match_path):
            continue

        # Expecting Deaths/ and Survivals/ inside each match folder
        for label_folder, label in [("Deaths", 1), ("Survivals", 0)]:
            subdir = os.path.join(match_path, label_folder)
            if not os.path.exists(subdir):
                continue
            for filename in os.listdir(subdir):
                if filename.endswith(".json"):
                    path = os.path.join(subdir, filename)
                    with open(path) as f:
                        try:
                            data = json.load(f)
                            flat = flatten_timeseries(data)
                            flat = pad_to_length(flat, EXPECTED_FEATURES)
                            X.append(flat)
                            y.append(label)
                        except Exception as e:
                            print(f"❌ Failed to load {path}: {e}")

    print(f"📊 Loaded dataset: {len(X)} samples")
    print(f"🔢 Label counts: {dict(Counter(y))}")
    return np.array(X), np.array(y)

def get_models():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42, use_label_encoder=False),
        "LightGBM": LGBMClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "CatBoost": CatBoostClassifier(iterations=200, verbose=False, random_seed=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, early_stopping=True, random_state=42),
    }

def train_and_save_all_models():
    print("📥 Loading dataset...")
    X, y = load_dataset(INPUT_DIR)

    print("🔢 Fitting scaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = get_models()

    for name, model in models.items():
        print(f"🎯 Training {name}...")
        model.fit(X_scaled, y)

        model_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(model, os.path.join(model_dir, "death_model.pkl"))
        joblib.dump(scaler, os.path.join(model_dir, "death_scaler.pkl"))

        print(f"💾 Saved model and scaler for {name} to {model_dir}")

    print("✅ All models trained and saved.")

if __name__ == "__main__":
    train_and_save_all_models()
