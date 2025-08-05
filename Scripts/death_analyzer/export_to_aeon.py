import json
import os
import argparse
from collections import defaultdict
from math import hypot

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

def extract_features(timeline, ref_time, hero_name, hero_unit):
    start_time = ref_time - WINDOW_SECONDS
    buckets = defaultdict(lambda: {
        "player_pos": None,
        "player_damage_taken": 0,
        "player_died": False,
        "player_level": 1,
        "player_hp_pct": 0,
        "teammates": {},
        "enemies": {},
    })

    hero_ids = set()
    for e in timeline:
        if e['type'] == "interval" and e['unit'].startswith("CDOTA_Unit_Hero_"):
            hero_ids.add(e['unit'])

    hero_ids = sorted(hero_ids)
    enemy_ids = [hid for hid in hero_ids if hid != hero_unit][:5]
    teammate_ids = [hid for hid in hero_ids if hid != hero_unit and hid not in enemy_ids][:4]

    for e in timeline:
        t = e["time"]
        if t < start_time or t > ref_time:
            continue
        bucket = buckets[t]

        if e['type'] == "interval":
            if e['unit'] == hero_unit:
                bucket["player_pos"] = (e['x'], e['y'])
                bucket["player_level"] = e.get("level", 1)
                hp = e.get("health", 0)
                max_hp = e.get("maxHealth", None)
                if isinstance(max_hp, (int, float)) and max_hp > 0:
                    bucket["player_hp_pct"] = round(hp / max_hp, 3)
                else:
                    bucket["player_hp_pct"] = 0
            elif e['unit'] in teammate_ids + enemy_ids:
                group = "teammates" if e['unit'] in teammate_ids else "enemies"
                hp = e.get("health", 0)
                max_hp = e.get("maxHealth", None)
                if isinstance(max_hp, (int, float)) and max_hp > 0:
                    hp_pct = round(hp / max_hp, 3)
                else:
                    hp_pct = 0
                bucket[group].setdefault(e['unit'], {})
                bucket[group][e['unit']].update({
                    "pos": (e['x'], e['y']),
                    "level": e.get("level", 1),
                    "hp_pct": hp_pct,
                })

        if e['type'] == "DOTA_COMBATLOG_DAMAGE":
            target = normalize_hero_unit(e.get("targetname", ""))
            if target == hero_unit:
                bucket["player_damage_taken"] += e["value"]
            elif target in teammate_ids:
                bucket["teammates"].setdefault(target, {}).setdefault("damage_taken", 0)
                bucket["teammates"][target]["damage_taken"] += e["value"]
            elif target in enemy_ids:
                bucket["enemies"].setdefault(target, {}).setdefault("damage_taken", 0)
                bucket["enemies"][target]["damage_taken"] += e["value"]

        if e['type'] == "DOTA_COMBATLOG_DEATH":
            target = normalize_hero_unit(e.get("targetname", ""))
            if target == hero_unit:
                bucket["player_died"] = True
            elif target in teammate_ids:
                bucket["teammates"].setdefault(target, {})["died"] = True
            elif target in enemy_ids:
                bucket["enemies"].setdefault(target, {})["died"] = True

    output = {}
    for t, v in sorted(buckets.items()):
        if not v["player_pos"]:
            continue

        teammates_pos = []
        enemies_pos = []

        for tid in teammate_ids:
            data = v["teammates"].get(tid)
            if data and "pos" in data and distance(v["player_pos"], data["pos"]) <= NEARBY_RADIUS:
                teammates_pos.append(normalize(data["pos"], v["player_pos"]))

        for eid in enemy_ids:
            data = v["enemies"].get(eid)
            if data and "pos" in data and distance(v["player_pos"], data["pos"]) <= NEARBY_RADIUS:
                enemies_pos.append(normalize(data["pos"], v["player_pos"]))

        teammates_data, enemies_data = [], []
        for tid in teammate_ids:
            data = v["teammates"].get(tid)
            if data and "pos" in data and distance(v["player_pos"], data["pos"]) <= NEARBY_RADIUS:
                teammates_data.append(data)

        for eid in enemy_ids:
            data = v["enemies"].get(eid)
            if data and "pos" in data and distance(v["player_pos"], data["pos"]) <= NEARBY_RADIUS:
                enemies_data.append(data)

        output[str(float(t))] = {
                    "player_pos": [0.0, 0.0],
                    "teammates_pos": [normalize(d["pos"], v["player_pos"]) for d in teammates_data],
                    "enemies_pos": [normalize(d["pos"], v["player_pos"]) for d in enemies_data],
                    "player_damage_taken": v["player_damage_taken"],
                    "player_died": v["player_died"],
                    "player_level": v["player_level"],
                    "player_hp_pct": v["player_hp_pct"],
                    "teammates_levels": [d.get("level", 1) for d in teammates_data],
                    "enemies_levels": [d.get("level", 1) for d in enemies_data],
                    "teammates_damage_taken": [d.get("damage_taken", 0) for d in teammates_data],
                    "enemies_damage_taken": [d.get("damage_taken", 0) for d in enemies_data],
                    "teammates_died": [d.get("died", False) for d in teammates_data],
                    "enemies_died": [d.get("died", False) for d in enemies_data],
                    "teammates_hp_pct": [d.get("hp_pct", 1.0) for d in teammates_data],
                    "enemies_hp_pct": [d.get("hp_pct", 1.0) for d in enemies_data],
                    "player_incapacitated": False
                }
        #output[str(float(t))] = {
        #    "player_pos": [0.0, 0.0],
        #    "teammates_pos": teammates_pos,
        #    "enemies_pos": enemies_pos,
        #    "player_damage_taken": v["player_damage_taken"],
        #    "player_died": v["player_died"],
        #    "player_level": v["player_level"],
        #    "player_hp_pct": v["player_hp_pct"],
        #    "teammates_levels": [v["teammates"].get(tid, {}).get("level", 1) for tid in teammate_ids],
        #    "enemies_levels": [v["enemies"].get(eid, {}).get("level", 1) for eid in enemy_ids],
        #    "teammates_damage_taken": [v["teammates"].get(tid, {}).get("damage_taken", 0) for tid in teammate_ids],
        #    "enemies_damage_taken": [v["enemies"].get(eid, {}).get("damage_taken", 0) for eid in enemy_ids],
        #    "teammates_died": [v["teammates"].get(tid, {}).get("died", False) for tid in teammate_ids],
        #    "enemies_died": [v["enemies"].get(eid, {}).get("died", False) for eid in enemy_ids],
        #    "teammates_hp_pct": [v["teammates"].get(tid, {}).get("hp_pct", 1.0) for tid in teammate_ids],
        #    "enemies_hp_pct": [v["enemies"].get(eid, {}).get("hp_pct", 1.0) for eid in enemy_ids]
            #"player_incapacitated": False
        #}

    return output

def detect_survivals(timeline, hero_name, hero_unit, window_seconds=10, damage_threshold=0.5, grace_period=5):
    survivals = []
    health_by_time = {}
    damage_events = defaultdict(int)
    death_times = set()

    for e in timeline:
        if e['type'] == "DOTA_COMBATLOG_DEATH" and normalize_hero_unit(e.get("targetname", "")) == hero_unit:
            death_times.add(e["time"])

    for e in timeline:
        t = e["time"]
        if e['type'] == "interval" and e['unit'] == hero_unit:
            hp = e.get("health", 0)
            max_hp = e.get("maxHealth", None)
            if isinstance(max_hp, (int, float)) and max_hp > 0:
                health_by_time[t] = hp / max_hp
        if e['type'] == "DOTA_COMBATLOG_DAMAGE" and normalize_hero_unit(e.get("targetname", "")) == hero_unit:
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
                clean = all(damage_events.get(t, 0) == 0 for t in range(t2 + 1, t2 + grace_period + 1))
                if not clean:
                    break

                if any(dt in range(t1 - 2, t2 + grace_period + 1) for dt in death_times):
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
        survivals = detect_survivals(timeline, hero_name, unit)
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
