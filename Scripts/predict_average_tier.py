import os
import json
import joblib
import sys
import numpy as np


MODEL_DIR = "../AeonModel"

# Load model artifacts
model = joblib.load(os.path.join(MODEL_DIR, "aeon_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "aeon_scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "aeon_label_encoder.pkl"))

#    if 11 <= tier_code <= 15:
#        return "Herald"
#    elif 21 <= tier_code <= 25:
#        return "Guardian"
#    elif 31 <= tier_code <= 35:
#        return "Crusader"
#    elif 41 <= tier_code <= 45:
#        return "Archon"
#    elif 51 <= tier_code <= 55:
#        return "Legend**s**"
#    elif 61 <= tier_code <= 65:
#        return "Ancient"
#    elif 71 <= tier_code <= 75:
#        return "Divine"
#    elif tier_code >= 80:
#        return "Immortal"
#    else:
#        return "Unknown"

def decode_tier(tier_name):

    if hasattr(tier_name, "item"):
        tier_name = tier_name.item()
    tier_name = str(tier_name)

    if tier_name == 'Herald':
       return 12
    elif tier_name == 'Guardian':
       return 21
    elif tier_name == 'Crusader':
       return 33
    elif tier_name == 'Archon':
       return 44
    elif tier_name == 'Legend':
       return 51
    elif tier_name == 'Ancient':
       return 62
    elif tier_name == 'Divine':
       return 73
    elif tier_name == 'Immortal':
       return 81
    else:
        return 99

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

def predict_timeseries(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    features = flatten_timeseries(data)

    expected_len = scaler.mean_.shape[0]
    if len(features) < expected_len:
        features += [0] * (expected_len - len(features))
    elif len(features) > expected_len:
        features = features[:expected_len]

    X_scaled = scaler.transform([features])

    proba = model.predict_proba(X_scaled)[0]
    top_idx = np.argmax(proba)
    top_label = label_encoder.inverse_transform([top_idx])[0]
    confidence = proba[top_idx]

    return top_label, confidence, dict(zip(label_encoder.classes_, proba))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_average_tier.py <path_to_json_or_folder>")
        sys.exit(1)

    input_path = sys.argv[1]

    if os.path.isdir(input_path):
        print(f"📂 Folder mode: Scanning {input_path}")
        json_files = [f for f in os.listdir(input_path) if f.endswith(".json")]
        if not json_files:
            print("⚠️ No JSON files found.")
            sys.exit(0)

        for fname in sorted(json_files):
            fpath = os.path.join(input_path, fname)
            try:
                predicted_tier, confidence, all_probs = predict_timeseries(fpath)
                print(f"\n📁 {fname}")
                print(f"🎯 Predicted average_tier: {predicted_tier}   Rank: {decode_tier(predicted_tier)}")

                print(f"📈 Confidence: {confidence:.2%}")
                print("📊 Full distribution:")
                for tier, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
                    print(f" - {tier:<10}: {prob:.2%}")
            except Exception as e:
                print(f"❌ Failed to process {fname}: {e}")
    elif os.path.isfile(input_path):
        predicted_tier, confidence, all_probs = predict_timeseries(input_path)

        print(f"🎯 Predicted average_tier: {predicted_tier}   Rank: {decode_tier(predicted_tier)}")

        print(f"📈 Confidence: {confidence:.2%}")
        print("📊 Full distribution:")
        for tier, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
            print(f" - {tier:<10}: {prob:.2%}")

    else:
        print(f"❌ {input_path} is neither a file nor a folder.")
