
import os
import json

ROOT_DIR = "../..//OutputJsons/DeathsAndSurvives"

errors = {
    "always_zero_damage": [],
    "never_low_hp": [],
    "never_died": []
}

def extract_label_from_path(filepath):
    parts = filepath.replace("\\", "/").split("/")  # Ensure compatibility across OS
    if "Deaths" in parts:
        return 1
    elif "Survivals" in parts:
        return 0
    return None

for root, _, files in os.walk(ROOT_DIR):

   
        for fname in files:
            if not fname.endswith(".json"):
                continue

            if extract_label_from_path(ROOT_DIR) == 1:

                path = os.path.join(root, fname)
                try:
                    with open(path) as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"❌ Failed to read {path}: {e}")
                    continue

                damage_values = []
                hp_values = []
                died_values = []

                for t in data.values():
                    damage_values.append(t.get("player_damage_taken", 0))
                    hp_values.append(t.get("player_hp_pct", 1.0))
                    died_values.append(t.get("player_died", False))

                if all(d == 0 for d in damage_values):
                    errors["always_zero_damage"].append(path)

                if all(hp >= 0.1 for hp in hp_values):
                    errors["never_low_hp"].append(path)

                if all(died is False for died in died_values):
                    errors["never_died"].append(path)
            else:

                path = os.path.join(root, fname)
                try:
                    with open(path) as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"❌ Failed to read {path}: {e}")
                    continue

                damage_values = []
                hp_values = []
                died_values = []

                for t in data.values():
                    damage_values.append(t.get("player_damage_taken", 0))
                    hp_values.append(t.get("player_hp_pct", 1.0))
                    died_values.append(t.get("player_died", False))

                if all(d == 0 for d in damage_values):
                    errors["always_zero_damage"].append(path)

print("\n==== Error Summary ====")
for err_type, files in errors.items():
    print(f"{err_type}: {len(files)}")
    if files:
        print("Examples:")
        for f in files[:5]:
            print(f" - {f}")
