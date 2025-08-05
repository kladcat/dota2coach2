import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import re
# === CONFIGURATION ===
MODEL_PATH = "../../AeonModel/tcn_death_model.keras"
SEQUENCE_LENGTH = 21 # expected length
EXPECTED_FEATURE_COUNT = 63  # must match the number used in training!

# === FLATTEN FUNCTION ===
def pad_feature_vectors(entry):
    flat = []
    for key, value in entry.items():
        if isinstance(value, bool):
            flat.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            flat.append(float(value))
        elif isinstance(value, list):
            if all(isinstance(x, (int, float)) for x in value):
                flat.extend([-1.0 if x is None else float(x) for x in value])
            elif all(isinstance(x, list) and len(x) == 2 for x in value):  # Vector2
                for pos in value:
                    flat.extend([float(pos[0]), float(pos[1])])
            elif all(isinstance(x, bool) for x in value):
                flat.extend([1.0 if x else 0.0 for x in value])
            else:
                flat.append(-1.0)  # fallback
        else:
            flat.append(-1.0)  # unknown type
    return flat

# === LOAD SAMPLE ===
def load_sample(path):
    with open(path) as f:
        data = json.load(f)
    features = data.get("features", [])
    if len(features) != SEQUENCE_LENGTH:
        raise ValueError(f"Expected sequence of length {SEQUENCE_LENGTH}, got {len(features)}")
    
    # --- Extract rank digit from filename
    match = re.search(r"_RT(\d{2})", path)
    rank_digit = float(match.group(1)[0]) if match else -1.0

    vectors = [pad_feature_vectors(entry) + [rank_digit] for entry in features]


    feature_dim = len(vectors[0])
    if not all(len(v) == feature_dim for v in vectors):
        raise ValueError(f"Inconsistent feature vector lengths in sample.")
    
    if feature_dim != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Sample has {feature_dim} features, but model expects {EXPECTED_FEATURE_COUNT}")

    return np.array([vectors], dtype=np.float32)

# === ENTRY POINT ===
if len(sys.argv) != 2:
    print("Usage: python3.10 predict_inevitable_second_tcn.py <path_to_sample.json>")
    sys.exit(1)

SEQUENCE_PATH = sys.argv[1]

print("📂 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded.")

print(f"📥 Loading sample from {SEQUENCE_PATH}...")
X = load_sample(SEQUENCE_PATH)
print(f"✅ Sample loaded with shape {X.shape} (sequence: {SEQUENCE_LENGTH}, features: {X.shape[2]})")

# === NORMALIZATION ===
X_mean = np.load("../../AeonModel/X_mean.npy")
X_std = np.load("../../AeonModel/X_std.npy")
X = (X - X_mean) / X_std

print("🤖 Predicting...")
proba = model.predict(X).flatten()[0]
print(f"\n💡 Final prediction (death probability): {proba:.3f}")

print("\n🕒 Per-second confidence evolution:")
for i in range(SEQUENCE_LENGTH):
    subseq = X[:, :i+1, :]
    padded = np.pad(subseq, ((0,0), (0,SEQUENCE_LENGTH - i - 1), (0,0)), constant_values=0)
    conf = model.predict(padded).flatten()[0]
    print(f"  ⏱️ t-{SEQUENCE_LENGTH - i:>2}: {conf:.3f}")
