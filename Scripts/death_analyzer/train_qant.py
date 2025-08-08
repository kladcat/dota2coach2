import os
import json
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === CONFIGURATION ===
DATA_DIR = "../../OutputJsons/DeathsAndSurvivesQANT"
SEQUENCE_LENGTH = 11
THRESHOLD = 0.5

# === LOAD DATA ===
def load_samples_from_folder(folder):
    def flatten_features(entry):
        flat = []
        for k, v in entry.items():
            if isinstance(v, (int, float, bool)):
                flat.append(float(v))
            elif isinstance(v, list):
                # Flat list
                if all(isinstance(x, (int, float, bool)) for x in v):
                    flat.extend(float(x) for x in v)
                # 2D positions
                elif all(isinstance(x, list) and len(x) == 2 for x in v):
                    for coord in v:
                        flat.extend(float(c) for c in coord)
                else:
                    continue  # skip unsupported
        return flat

    X, y = [], []
    for root, _, files in os.walk(folder):
        for file in files:
            if not file.endswith(".json"):
                continue

            # Extract rank from filename (e.g. _RT43 → 4)
            match = re.search(r"_RT(\d{2})", file)
            rank_digit = float(match.group(1)[0]) if match else -1.0

            path = os.path.join(root, file)
            with open(path) as f:
                data = json.load(f)

            label = data.get("label", None)
            if isinstance(label, str):
                label = 1 if label == "death" else 0
            if label not in [0, 1]:
                print(f"⚠️ Skipping {file}: invalid label {label}")
                continue

            seq = data.get("features", [])
            if len(seq) != SEQUENCE_LENGTH:
                print(f"⚠️ Skipping {file}: wrong seq length ({len(seq)} != {SEQUENCE_LENGTH})")
                continue

            try:
                vec_seq = [flatten_features(entry) + [rank_digit] for entry in seq]
                base_len = len(vec_seq[0])
                if not all(len(v) == base_len for v in vec_seq):
                    raise ValueError("Inconsistent feature lengths")
            except Exception as e:
                print(f"❌ Skipping {file}: {e}")
                continue

            X.append(vec_seq)
            y.append(label)

    return np.array(X), np.array(y)

print("📂 Loading data...")
X, y = load_samples_from_folder(DATA_DIR)
print(f"✅ Loaded {len(X)} samples: {np.sum(y)} deaths / {len(y) - np.sum(y)} survivals")

# === SPLIT ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === NORMALIZATION ===
X_mean = np.mean(X_train, axis=(0, 1), keepdims=True)
X_std = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-8
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std

# === MODEL ===
input_shape = (X.shape[1], X.shape[2])
model = Sequential([
    Input(shape=input_shape),
    Conv1D(128, kernel_size=3, padding="causal", activation="relu"),
    Dropout(0.1),
    Conv1D(64, kernel_size=3, padding="causal", activation="relu"),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# === TRAIN ===
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# Optional: handle class imbalance
# class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# class_weights = dict(enumerate(class_weights))

print("🚀 Training model...")
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    # class_weight=class_weights
)

# === EVALUATION ===
y_pred_prob = model.predict(X_val).flatten()
y_pred = (y_pred_prob >= THRESHOLD).astype(int)

print("\n📊 Classification Report:")
print(classification_report(y_val, y_pred, digits=3))
print(f"🔢 AUC Score: {roc_auc_score(y_val, y_pred_prob):.3f}")

# === SAVE ===
model.save("../../AeonModel/tcn_death_model.keras")
print("💾 Model saved to tcn_death_model.keras")

np.save("../../AeonModel/X_mean.npy", X_mean)
np.save("../../AeonModel/X_std.npy", X_std)
