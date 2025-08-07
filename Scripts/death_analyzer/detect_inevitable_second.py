import sys
import json
import joblib
import numpy as np
import os
import re

MODEL_DIR = "../../AeonModel"
MODEL_PATH = os.path.join(MODEL_DIR, "death_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "death_scaler.pkl")

THRESHOLD = 0.8  # 80% confidence

def extract_rank_from_filename(filename, fallback_rank=None):
    match = re.search(r"_RT(\d+)", filename)
    if match:
        return int(match.group(1))
    if fallback_rank is not None:
        return fallback_rank
    return 0  # Default fallback if nothing is provided

def flatten_timeseries_up_to(data, up_to_second, path, fallback_rank=None):
    time_keys = sorted(map(float, data.keys()))
    flat = []
    for t in time_keys:
        if t > up_to_second:
            break
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

    rank = extract_rank_from_filename(os.path.basename(path), fallback_rank)
    #flat.append(rank)
    return flat

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
        print("Usage: python3 detect_confidence_evolution.py path/to/death_file.json [optional_rank_tier]")
        sys.exit(1)

    input_path = sys.argv[1]
    fake_rank = int(sys.argv[2]) if len(sys.argv) == 3 else None

    print_confidence_evolution(input_path, fake_rank)
