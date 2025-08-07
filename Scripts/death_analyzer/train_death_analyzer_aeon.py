import os
import json
import numpy as np
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
import argparse


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
        t_key = str(int(t))  # ← fixes the KeyError
        #snapshot = data[t_key]
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

def extract_hero_from_filename(filename):
    match = re.search(r"_(?:death|surv)_([a-z_]+)_", filename)
    return match.group(1) if match else None

def extract_time_from_filename(filename):
    match = re.search(r"_(\d+)\.json$", filename)
    return int(match.group(1)) if match else None
    

def shift_time_to_zero(data):
    time_keys = sorted(map(float, data.keys()))
    if not time_keys:
        return {}
    min_time = time_keys[0]
    shifted = {}
    for key in data:
        new_key = str(int(float(key) - min_time))  # key = "0", "1", ...
        shifted[new_key] = data[key]
    return shifted


def load_dataset(input_dir, time_filter, hero_filter):
    X, y = [], []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue

            event_time = extract_time_from_filename(fname)
            if event_time is None or event_time > time_filter:
                continue

            if hero_filter:
                hero_name = extract_hero_from_filename(fname)
                if hero_name != hero_filter.lower():
                    continue

            full_path = os.path.join(root, fname)
            label = extract_label_from_path(full_path)
            if label is None:
                print(f"⚠️ Skipping file due to missing label: {full_path}")  # <-- ADD THIS
                continue
            with open(full_path, 'r') as f:
                data = json.load(f)
                #original_data = json.load(f)
                #data = shift_time_to_zero(original_data)
            rank = extract_rank_from_filename(fname)
                        
            if rank is None:
                print(f"⚠️ Skipping file due to missing rank: {full_path}")
                continue

            features = flatten_timeseries(data)
            flat = pad_to_full_length(features, full_length=352)


            X.append(flat)  # use the padded version!
            y.append(label)
    return X, y

def get_model(name):
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    elif name == "logistic":
        return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    elif name == "xgb":
        return XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=42)
    elif name == "lgbm":
        return LGBMClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    elif name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    elif  name == "gradient":
        return GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,random_state=42)
    elif name == "decisiontree":
        return DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        )
    elif name == "CatBoostClassifier":
        return CatBoostClassifier(
                iterations=300,
                learning_rate=0.1,
                depth=6,
                loss_function='Logloss',
                random_seed=42,
                verbose=False
            )
    else:
        raise ValueError("Unsupported model")

def pad_to_full_length(flat, full_length=352):
    missing = full_length - len(flat)
    if missing > 0:
        return flat + [0] * missing
    return flat

def train(time_limit, hero_filter):
    print("📥 Loading data...")
    X, y = load_dataset(INPUT_DIR, time_filter=time_limit, hero_filter=hero_filter)

    print("📊 Labels in training set:", set(y))
    print("🔢 Label counts:", {label: y.count(label) for label in set(y)})

    max_len = max(len(x) for x in X)
    X_padded = [x + [0] * (max_len - len(x)) for x in X]

    print("🔢 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_padded)

    print("🎯 Training model...")
    model = get_model("xgb")
    #model = GradientBoostingClassifier(
    #n_estimators=200,
    #learning_rate=0.1,
    #random_state=42,
    #)

    #model = CalibratedClassifierCV(rf, method="isotonic", cv=3)

    model.fit(X_scaled, y)

    print("💾 Saving model...")
    joblib.dump(model, os.path.join(MODEL_DIR, "death_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "death_scaler.pkl"))
    print("✅ Death prediction model training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=int, default=9999, help="Only include events that occur at or before this second")
    parser.add_argument("--hero", type=str, default="", help="Only include samples from this hero name (e.g., chaos_knight)")
    args = parser.parse_args()

    train(time_limit=args.time, hero_filter=args.hero)
