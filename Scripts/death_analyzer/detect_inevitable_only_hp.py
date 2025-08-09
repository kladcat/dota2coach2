import sys
import json
import joblib
import numpy as np
import os
import re
from sklearn.preprocessing import MinMaxScaler

MODEL_DIR = "../../AeonModel"
MODEL_PATH = os.path.join(MODEL_DIR, "death_model_hp.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "death_scaler_hp.pkl")

THRESHOLD = 0.8  # 80% confidence


def flatten_timeseries(data):
    flat = []
    time_keys = sorted(map(float, data.keys()))
    #print("Time keys:", time_keys)  # Debug print to inspect time keys

    feature_names = [
        "enemies_damage_taken", "player_damage_taken", "player_level", "enemies_levels", 
        "player_hp_pct", "enemies_hp_pct"
    ]

    for t in time_keys:
        snapshot = data[str(t)]
        #flat.extend([
        #    *snapshot.get("enemies_damage_taken", []),
        #    snapshot.get("player_damage_taken", 0),
        #    snapshot.get("player_level", 0),
        #    *snapshot.get("enemies_levels", []),
        #    snapshot.get("player_hp_pct", 0),
        #    *snapshot.get("enemies_hp_pct", []),
        #])

        flat.extend([
            # Player Features
            snapshot.get("player_damage_taken", 0),
            snapshot.get("player_died", 0),
            snapshot.get("player_level", 0),
            snapshot.get("player_hp_pct", 0),
            snapshot.get("player_incapacitated", 0),
            
            # Teammate Features (1 to 5)
            snapshot.get("teammates_pos1", -1),
            snapshot.get("teammates_damage_taken_1", -1),
            snapshot.get("teammates_died_1", -1),
            snapshot.get("teammates_level_1", -1),
            snapshot.get("teammates_hp_pct_1", -1),
            
            snapshot.get("teammates_pos2", -1),
            snapshot.get("teammates_damage_taken_2", -1),
            snapshot.get("teammates_died_2", -1),
            snapshot.get("teammates_level_2", -1),
            snapshot.get("teammates_hp_pct_2", -1),
            
            snapshot.get("teammates_pos3", -1),
            snapshot.get("teammates_damage_taken_3", -1),
            snapshot.get("teammates_died_3", -1),
            snapshot.get("teammates_level_3", -1),
            snapshot.get("teammates_hp_pct_3", -1),
            
            snapshot.get("teammates_pos4", -1),
            snapshot.get("teammates_damage_taken_4", -1),
            snapshot.get("teammates_died_4", -1),
            snapshot.get("teammates_level_4", -1),
            snapshot.get("teammates_hp_pct_4", -1),
            
            snapshot.get("teammates_pos5", -1),
            snapshot.get("teammates_damage_taken_5", -1),
            snapshot.get("teammates_died_5", -1),
            snapshot.get("teammates_level_5", -1),
            snapshot.get("teammates_hp_pct_5", -1),
            
            # Enemy Features (1 to 5)
            snapshot.get("enemies_pos1", -1),
            snapshot.get("enemies_damage_taken_1", -1),
            snapshot.get("enemies_died_1", -1),
            snapshot.get("enemies_level_1", -1),
            snapshot.get("enemies_hp_pct_1", -1),
            
            snapshot.get("enemies_pos2", -1),
            snapshot.get("enemies_damage_taken_2", -1),
            snapshot.get("enemies_died_2", -1),
            snapshot.get("enemies_level_2", -1),
            snapshot.get("enemies_hp_pct_2", -1),
            
            snapshot.get("enemies_pos3", -1),
            snapshot.get("enemies_damage_taken_3", -1),
            snapshot.get("enemies_died_3", -1),
            snapshot.get("enemies_level_3", -1),
            snapshot.get("enemies_hp_pct_3", -1),
            
            snapshot.get("enemies_pos4", -1),
            snapshot.get("enemies_damage_taken_4", -1),
            snapshot.get("enemies_died_4", -1),
            snapshot.get("enemies_level_4", -1),
            snapshot.get("enemies_hp_pct_4", -1),
            
            snapshot.get("enemies_pos5", -1),
            snapshot.get("enemies_damage_taken_5", -1),
            snapshot.get("enemies_died_5", -1),
            snapshot.get("enemies_level_5", -1),
            snapshot.get("enemies_hp_pct_5", -1),
        ])

                # Print each feature's value alongside its name
        #for name in feature_names:
        #    values = snapshot.get(name, [])
        #    if isinstance(values, list):
        #        for v in values:
        #            flat.append(v)
        #            print(f"{name}: {v}")  # Debugging print
        #    else:
        #        flat.append(values)
        #        print(f"{name}: {values}")  # Debugging print
    return flat

def flatten_timeseries_up_to(data, up_to_second, path, fallback_rank=None):
    time_keys = sorted(map(float, data.keys()))
    flat = []
    for t in time_keys:
        if t > up_to_second:
            break
        snapshot = data[str(t)]
        flat.extend([
            *snapshot.get("enemies_damage_taken", []),
            snapshot.get("player_damage_taken", 0),
            snapshot.get("player_level", 0),
            *snapshot.get("enemies_levels", []),
            snapshot.get("player_hp_pct", 0),
            *snapshot.get("enemies_hp_pct", []),
        ])

    #rank = extract_rank_from_filename(os.path.basename(path), fallback_rank)
    #flat.append(rank)
    return flat

def predict_timeseries_basic(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    features = flatten_timeseries(data)
    print(f"Features after flattening: {features}")
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    expected_len = scaler.mean_.shape[0]

    if len(features) < expected_len:
        features += [0] * (expected_len - len(features))
    elif len(features) > expected_len:
        features = features[:expected_len]

    X_scaled = scaler.transform([features])
    print(f"X_scaled: {X_scaled.flatten()}")

    proba = model.predict_proba(X_scaled)[0]
    top_idx = np.argmax(proba)
    top_label = "Survival" if top_idx == 0 else "Death"
    confidence = proba[top_idx]

    return top_label, confidence

def predict_basic(input_path):
    predicted_tier, confidence = predict_timeseries_basic(input_path)

    print(f"🎯 Predicted average_tier: {predicted_tier} ")
    print(f"📈 Confidence: {confidence:.2%}")


def print_confidence_evolution(path, fake_rank=None):
    with open(path, "r") as f:
        data = json.load(f)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    expected_len = scaler.mean_.shape[0]

    time_keys = sorted(map(float, data.keys()))
    crossed = False

    print("📈 Confidence Evolution:\n")
    for t in time_keys:
        flat = flatten_timeseries_up_to(data, t, path, fake_rank)
        if len(flat) < expected_len:
            flat += [0] * (expected_len - len(flat))  # pad with zeros
        flat = np.array(flat).reshape(1, -1)

        flat_scaled = scaler.transform(flat)
        proba = model.predict_proba(flat_scaled)[0]
        death_confidence = proba[1]

        marker = ""
        if not crossed and death_confidence >= THRESHOLD:
            crossed = True
            marker = "⬅️ **THRESHOLD CROSSED**"

        #print(f"t={t:>5}s → Death: {death_confidence*100:6.2f}% {marker}")
        print(f"t={t:>5}s → Survival: {proba[0]*100:6.2f}% | Death: {proba[1]*100:6.2f}% {marker}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python3 predict_inevitable_second.py path/to/death_file.json [optional_rank_tier]")
        sys.exit(1)

    input_path = sys.argv[1]
    fake_rank = int(sys.argv[2]) if len(sys.argv) == 3 else None

    print_confidence_evolution(input_path, fake_rank)

    #predict_basic(input_path)



    
