import json
import os
import argparse
from collections import defaultdict
from math import hypot

from timeline_data_gatherer import DeathSurvivalDataGatherer  # add this import at the top


WINDOW_SECONDS = 10
OUTPUT_DIR = "../../OutputJsons/DeathsAndSurvives"
NEARBY_RADIUS = 20

def load_timeline(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def distance(p1, p2):
    return hypot(p1[0] - p2[0], p1[1] - p2[1])

def normalize(pos, origin):
    return [round(pos[0] - origin[0], 2), round(pos[1] - origin[1], 2)]

def normalize_hero_unit(name):
    """
    Convert things like 'npc_dota_hero_chaos_knight' → 'CDOTA_Unit_Hero_ChaosKnight'
    """
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


def extract_features(timeline, ref_time, hero_name, hero_unit):
    gatherer = DeathSurvivalDataGatherer(timeline, hero_name, hero_unit)
    return gatherer.extract_features(ref_time)


def detect_survivals(timeline, hero_name, hero_unit, window_seconds=10, damage_threshold=0.25, grace_period=5):
    survivals = []
    health_by_time = {}
    damage_events = defaultdict(int)
    death_times = set()

    for e in timeline:
        if e['type'] == "DOTA_COMBATLOG_DEATH" and normalize_name(e.get("targetname", "")) == hero_unit:
            death_times.add(e["time"])

    for e in timeline:
        t = e["time"]
        if e['type'] == "interval" and e['unit'] == hero_unit:
            hp = e.get("health", 0)
            max_hp = e.get("maxHealth", None)
            if isinstance(max_hp, (int, float)) and max_hp > 0:
                health_by_time[t] = hp / max_hp
        if e['type'] == "DOTA_COMBATLOG_DAMAGE" and normalize_name(e.get("targetname", "")) == hero_unit:
            damage_events[t] += e["value"]

    sorted_times = sorted(health_by_time.keys())
    i = 0
    while i < len(sorted_times) - 1:
        t1 = sorted_times[i]
        hp1 = health_by_time[t1]

        for j in range(i + 1, len(sorted_times)):
            t2 = sorted_times[j]
            hp2 = health_by_time[t2]

            if hp1 - hp2 >= damage_threshold:
                #clean = all(damage_events.get(t, 0) == 0 for t in range(t2 + 1, t2 + grace_period + 1))
                #if not clean:
                #    break

                #if any(dt in range(t1 - 2, t2 + grace_period + 1) for dt in death_times):
                #    break

                if any(dt in range(t1, t2 + grace_period + 1) for dt in death_times):
                    break

                features = extract_features(timeline, t2, hero_name, hero_unit)
                survivals.append((t2, features))
                i = j + grace_period
                break
        i += 1

    return survivals

def detect_survivals2(timeline, hero_name, hero_unit, window_seconds=10, damage_threshold=0.3, grace_period=5):
    survivals = []
    health_by_time = {}
    damage_events = defaultdict(int)
    death_times = set()

    for e in timeline:
        if e['type'] == "DOTA_COMBATLOG_DEATH" and normalize_name(e.get("targetname", "")) == hero_unit:
            death_times.add(e["time"])

    for e in timeline:
        t = e["time"]
        if e['type'] == "interval" and e['unit'] == hero_unit:
            hp = e.get("health", 0)
            max_hp = e.get("maxHealth", None)
            if isinstance(max_hp, (int, float)) and max_hp > 0:
                health_by_time[t] = hp / max_hp
        if e['type'] == "DOTA_COMBATLOG_DAMAGE" and normalize_name(e.get("targetname", "")) == hero_unit:
            damage_events[t] += e["value"]

    sorted_times = sorted(health_by_time.keys())
    i = 0
    while i < len(sorted_times) - 1:
        t1 = sorted_times[i]
        hp1 = health_by_time[t1]

        for j in range(i + 1, len(sorted_times)):
            t2 = sorted_times[j]

            # ⏱️ Enforce max window duration
            if t2 - t1 > window_seconds:
                break

            hp2 = health_by_time[t2]

            if hp1 - hp2 >= damage_threshold:
                # ❌ Skip if the player died during the window
                if any(dt in range(t1, t2 + grace_period + 1) for dt in death_times):
                    break

                features = extract_features(timeline, t2, hero_name, hero_unit)
                survivals.append((t2, features))
                i = j + grace_period
                break
        i += 1

    return survivals

def save_sample(features, path):
    with open(path, "w") as f:
        json.dump(features, f, indent=2)

def detect_hero_name(timeline, unit):
    for e in timeline:
        if e.get("type") == "DOTA_ABILITY_LEVEL" and e.get("unit") == unit:
            ability = e.get("ability", "")
            match = re.match(r"(npc_dota_hero_[a-z_]+)_ability", ability)
            if match:
                return match.group(1)
    # fallback
    return f"npc_dota_hero_{camel_to_snake(unit.replace('CDOTA_Unit_Hero_', ''))}"

import re
def camel_to_snake(name):
    return re.sub(r'(?<=[a-z])([A-Z])', r'_\1', name).lower()

def process_file(path):
    timeline = load_timeline(path)
    base = os.path.splitext(os.path.basename(path))[0]
    out_deaths = os.path.join(OUTPUT_DIR, base, "Deaths")
    out_surv = os.path.join(OUTPUT_DIR, base, "Survivals")
    os.makedirs(out_deaths, exist_ok=True)
    os.makedirs(out_surv, exist_ok=True)

    hero_units = set()
    for e in timeline:
        if e['type'] == "interval" and e['unit'].startswith("CDOTA_Unit_Hero_"):
            hero_units.add(e['unit'])

    for unit in hero_units:
        hero_name = detect_hero_name(timeline, unit)
        #print(f"🔍 Processing hero {hero_name} ({unit})")

        # === Export deaths ===
        for e in timeline:
            if e['type'] == "DOTA_COMBATLOG_DEATH":
                normalized_target = normalize_hero_unit(e.get("targetname", ""))
                #print(f"💀 Checking death: target={e.get('targetname')} → normalized={normalized_target}, expected={unit}")
                if normalized_target == unit:
                    t = e["time"]
                    features = extract_features(timeline, t, hero_name, unit)
                    fname = f"{base}_death_{hero_name.replace('npc_dota_hero_', '')}_{t}.json"
                    save_sample(features, os.path.join(out_deaths, fname))
                    #print(f"✅ Exported death: {fname}")

        # === Export survivals ===
        survivals = detect_survivals2(timeline, hero_name, unit)
        for t, features in survivals:
            fname = f"{base}_surv_{hero_name.replace('npc_dota_hero_', '')}_{t}.json"
            save_sample(features, os.path.join(out_surv, fname))


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
