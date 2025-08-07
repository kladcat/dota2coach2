import sys
import json
import joblib
import numpy as np
import os
import re

MODEL_DIR = "../../AeonModel"
THRESHOLD = 0.8  # 80% confidence


def extract_rank_from_filename(filename, fallback_rank=None):
    match = re.search(r"_RT(\d+)", filename)
    if match:
        return int(match.group(1))
    if fallback_rank is not None:
        return fallback_rank
    return 0


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
    return flat


def load_all_models(model_root):
    models = {}
    for model_name in os.listdir(model_root):
        model_path = os.path.join(model_root, model_name, "death_model.pkl")
        scaler_path = os.path.join(model_root, model_name, "death_scaler.pkl")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                expected_len = scaler.mean_.shape[0]
                models[model_name] = {
                    "model": model,
                    "scaler": scaler,
                    "expected_len": expected_len,
                }
            except Exception as e:
                print(f"⚠️ Failed to load {model_name}: {e}")
    return models


def print_confidence_model_by_model(path, fake_rank=None):
    with open(path, "r") as f:
        data = json.load(f)

    models = load_all_models(MODEL_DIR)
    if not models:
        print("❌ No models found in AeonModel.")
        return

    time_keys = sorted(map(float, data.keys()))
    available_seconds = [int(t) for t in time_keys if 0 <= t <= 10]

    for name, obj in models.items():
        print(f"\n🔍 Model: {name}")
        crossed = False

        for t in available_seconds:
            flat = flatten_timeseries_up_to(data, t, path, fake_rank)
            padded = flat + [0] * (obj["expected_len"] - len(flat))
            padded = np.array(padded).reshape(1, -1)
            scaled = obj["scaler"].transform(padded)
            proba = obj["model"].predict_proba(scaled)[0]
            confidence = proba[1]

            marker = ""
            if not crossed and confidence >= THRESHOLD:
                crossed = True
                marker = "⬅️ **THRESHOLD CROSSED**"

            print(f"t={t:>2}s → Death: {confidence*100:6.2f}% {marker}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python3 detect_inevitable_second.py path/to/death_file.json [optional_rank_tier]")
        sys.exit(1)

    input_path = sys.argv[1]
    fake_rank = int(sys.argv[2]) if len(sys.argv) == 3 else None

    print_confidence_model_by_model(input_path, fake_rank)
