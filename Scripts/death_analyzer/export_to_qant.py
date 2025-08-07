import json
import os
import argparse
import re
from timeline_data_gatherer import DeathSurvivalDataGatherer

OUTPUT_DIR = "../../OutputJsons/DeathsAndSurvivesQANT"

def load_timeline(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def normalize_hero_unit(name):
    if not name.startswith("npc_dota_hero_"):
        return None
    suffix = name.replace("npc_dota_hero_", "")
    parts = suffix.split("_")
    camel = "".join(p.capitalize() for p in parts)
    return f"CDOTA_Unit_Hero_{camel}"

def normalize_name(name: str) -> str:
    name = name.lower().replace("_", "")
    if name.startswith("cdotaunit"):
        name = name[len("cdotaunit"):]
    elif name.startswith("npcdota"):
        name = name[len("npcdota"):]
    return name

def camel_to_snake(name):
    return re.sub(r'(?<=[a-z])([A-Z])', r'_\1', name).lower()

def detect_hero_name(timeline, unit):
    for e in timeline:
        if e.get("type") == "DOTA_ABILITY_LEVEL" and e.get("unit") == unit:
            ability = e.get("ability", "")
            match = re.match(r"(npc_dota_hero_[a-z_]+)_ability", ability)
            if match:
                return match.group(1)
    return f"npc_dota_hero_{camel_to_snake(unit.replace('CDOTA_Unit_Hero_', ''))}"

def convert_aeon_format_to_qant_dict(features_dict):
    qant_features = []
    for time_str, data in features_dict.items():
        qant_features.append({
            "time": float(time_str),
            **data
        })
    return qant_features

def save_qant_sample(qant_obj, out_dir, fname):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(qant_obj, f, indent=2)

def process_file(path):
    timeline = load_timeline(path)
    base = os.path.splitext(os.path.basename(path))[0]
    match_id = base.split("_")[0] if "_" in base else base
    match_dir = os.path.join(OUTPUT_DIR, base)
    out_deaths = os.path.join(match_dir, "Deaths")
    out_surv = os.path.join(match_dir, "Survivals")

    hero_units = set()
    for e in timeline:
        if e['type'] == "interval" and e['unit'].startswith("CDOTA_Unit_Hero_"):
            hero_units.add(e['unit'])

    for unit in hero_units:
        hero_name = detect_hero_name(timeline, unit)
        hero_short = hero_name.replace("npc_dota_hero_", "")
        gatherer = DeathSurvivalDataGatherer(timeline, hero_name, unit)

        # === Deaths ===
        for e in timeline:
            if e['type'] == "DOTA_COMBATLOG_DEATH":
                normalized_target = normalize_hero_unit(e.get("targetname", ""))
                if normalized_target == unit:
                    t = e["time"]
                    features = gatherer.extract_features(t, verbose=False)
                    qant_obj = {
                        "label": "death",
                        "hero": hero_short,
                        "match_id": match_id,
                        "features": convert_aeon_format_to_qant_dict(features)
                    }
                    fname = f"{base}_death_{hero_short}_{t}.json"
                    save_qant_sample(qant_obj, out_deaths, fname)

        # === Survivals ===
        survivals = gatherer._detect_survivals(timeline, hero_name, unit)
        for t, features in survivals:
            qant_obj = {
                "label": "survival",
                "hero": hero_short,
                "match_id": match_id,
                "features": convert_aeon_format_to_qant_dict(features)
            }
            fname = f"{base}_surv_{hero_short}_{t}.json"
            save_qant_sample(qant_obj, out_surv, fname)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    args = parser.parse_args()

    path = args.input_path
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.endswith(".jsonl"):
                process_file(os.path.join(path, f))
    else:
        process_file(path)

if __name__ == "__main__":
    main()
