import os
import json
import argparse

AEON_BASE = "../../OutputJsons/DeathsAndSurvivalsAeon"
SEQUENCE_LENGTH = 21  # Used to validate inputs

def convert_tcn_to_aeon(tcn_path, root_input):
    try:
        with open(tcn_path) as f:
            data = json.load(f)

        features = data.get("features", [])
        label = data.get("label", None)

        if not isinstance(features, list) or not isinstance(label, int):
            print(f"⚠️ Skipped (invalid format): {tcn_path}")
            return

        if len(features) != SEQUENCE_LENGTH:
            print(f"⚠️ Skipped (unexpected length {len(features)}): {tcn_path}")
            return

        aeon_data = []
        for timestep in features:
            timestep = timestep.copy()
            timestep["label"] = label
            aeon_data.append(timestep)

        # Build output path
        rel_path = os.path.relpath(tcn_path, start=root_input)
        out_path = os.path.join(AEON_BASE, rel_path)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path.replace(".json", "_Aeon.json"), "w") as out_f:
            json.dump(aeon_data, out_f, indent=2)

        print(f"✅ Converted: {tcn_path} → {out_path.replace('.json', '_Aeon.json')}")

    except Exception as e:
        print(f"❌ Failed to process {tcn_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert TCN samples to Aeon format.")
    parser.add_argument("input_path", help="Single TCN .json file or folder to scan")
    args = parser.parse_args()

    input_path = args.input_path

    if input_path.endswith(".json"):
        convert_tcn_to_aeon(input_path, os.path.dirname(input_path))
    else:
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".json"):
                    full_path = os.path.join(root, file)
                    convert_tcn_to_aeon(full_path, input_path)

if __name__ == "__main__":
    main()
