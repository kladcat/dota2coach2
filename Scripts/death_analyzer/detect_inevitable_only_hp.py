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

def flatten_timeseries_up_to(data, up_to_second, path, fallback_rank=None):
    time_keys = sorted(map(float, data.keys()))  # Ensure time keys are sorted as floats
    flat = []

    print(f"Flattening timeseries up to second {up_to_second}...\n")

    for t in time_keys:
        if t > up_to_second:  # We should include time <= up_to_second
            continue  # Skip times greater than up_to_second
        
        snapshot = data[str(t)]
        
        # Debug: print the health value at each time point
        print(f"At time {t}, player_hp_pct = {snapshot.get('player_hp_pct', 'N/A')}")
        
        # Only extract the player_hp_pct feature
        #flat.append(snapshot.get("player_hp_pct", 0))
        flat.extend([
            *snapshot.get("enemies_damage_taken", []),
            snapshot.get("player_damage_taken", 0),
            snapshot.get("player_level", 0),
            *snapshot.get("enemies_levels", []),
            snapshot.get("player_hp_pct", 0),
            *snapshot.get("enemies_hp_pct", []),
        ])

    # Debug: print the flattened feature vector
    print(f"Flattened values: {flat}")

    # Return only the health percentage (a 1D list of player_hp_pct values)
    return flat

def print_confidence_evolution(path, fake_rank=None):
    with open(path, "r") as f:
        data = json.load(f)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Extract all player_hp_pct values to fit the scaler once
    all_hp_values = []  # To store all player_hp_pct values over time
    for t in sorted(map(float, data.keys())):
        flat = flatten_timeseries_up_to(data, t, path, fake_rank)
        all_hp_values.extend(flat)  # Add current time's hp to the list

    # Fit the MinMaxScaler on the entire range of player_hp_pct values
    all_hp_values_reshaped = np.array(all_hp_values).reshape(-1, 1)  # Reshape for scaling
    scaler.fit(all_hp_values_reshaped)  # Fit scaler once on the entire data range

    time_keys = sorted(map(float, data.keys()))
    crossed = False

    print("📈 Confidence Evolution:\n")
    for t in time_keys:
        # Extract player_hp_pct values up to the current second
        flat = flatten_timeseries_up_to(data, t, path, fake_rank)

        # Debugging print to check if the feature list is changing
        print(f"Raw values at time {t}: {flat}")

        # Ensure it's 2D for scaling: the scaler expects 2D array for each sample
        flat = np.array(flat).reshape(-1, 1)

        # Debug: Print the flattened (raw) and reshaped values
        print(f"Before scaling at time {t}: {flat}")

        # Now scale the feature using the MinMaxScaler (fit on entire data, transform on current)
        flat_scaled = scaler.transform(flat)

        # Debug: Check the scaled values
        print(f"Scaled input at time {t}: {flat_scaled}")

        proba = model.predict_proba(flat_scaled)[0]
        death_confidence = proba[1]

        marker = ""
        if not crossed and death_confidence >= THRESHOLD:
            crossed = True
            marker = "⬅️ **THRESHOLD CROSSED**"

        print(f"t={t:>5}s → Survival: {proba[0]*100:6.2f}% | Death: {proba[1]*100:6.2f}% {marker}")

def flatten_timeseries(data):
    flat = []
    time_keys = sorted(map(float, data.keys()))
    print("Time keys:", time_keys)  # Debug print to inspect time keys
    for t in time_keys:
        snapshot = data[str(t)]
        flat.extend([snapshot.get("player_hp_pct", 0)])
    return flat

def predict_timeseries_basic(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    features = flatten_timeseries(data)

    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    expected_len = scaler.mean_.shape[0]

    if len(features) < expected_len:
        features += [0] * (expected_len - len(features))
    elif len(features) > expected_len:
        features = features[:expected_len]

    X_scaled = scaler.transform([features])

    proba = model.predict_proba(X_scaled)[0]
    top_idx = np.argmax(proba)
    top_label = "Survival" if top_idx == 0 else "Death"
    confidence = proba[top_idx]

    return top_label, confidence

def predict_basic(input_path):
    predicted_tier, confidence = predict_timeseries_basic(input_path)

    print(f"🎯 Predicted average_tier: {predicted_tier} ")
    print(f"📈 Confidence: {confidence:.2%}")



if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python3 predict_inevitable_second.py path/to/death_file.json [optional_rank_tier]")
        sys.exit(1)

    input_path = sys.argv[1]
    fake_rank = int(sys.argv[2]) if len(sys.argv) == 3 else None

    #print_confidence_evolution(input_path, fake_rank)

    predict_basic(input_path)



    
