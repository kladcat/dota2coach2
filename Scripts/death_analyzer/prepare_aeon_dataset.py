import os
import json
import numpy as np
from torch.utils.data import Dataset

class Dota2DeathDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = []
        for subdir, _, files in os.walk(root):
            for f in files:
                if f.endswith(".json"):
                    self.files.append(os.path.join(subdir, f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with open(path) as f:
            data = json.load(f)

        # Extract average_rank from filename
        filename = os.path.basename(path)
        try:
            rank = int(filename.split("_RT")[1].split("_")[0])
        except Exception:
            rank = 0

        # Parse time-ordered entries
        times = sorted(float(k) for k in data.keys())
        features = [self.extract_features(data[str(t)], rank) for t in times]

        X = np.array(features, dtype=np.float32)

        # Label: 1 if death, 0 if survival
        parent_folder = os.path.basename(os.path.dirname(path)).lower()
        y = 1 if "death" in parent_folder else 0

        return X, y

    def extract_features(self, entry, rank):
        return [
            *entry["player_pos"],
            *entry["teammates_damage_taken"],
            *entry["enemies_damage_taken"],
            entry["player_damage_taken"],
            entry["player_level"],
            *entry["teammates_levels"],
            *entry["enemies_levels"],
            entry["player_hp_pct"],
            *entry["teammates_hp_pct"],
            *entry["enemies_hp_pct"],
            rank  # average_rank as final input
        ]
