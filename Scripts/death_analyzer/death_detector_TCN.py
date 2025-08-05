import json
import sys
from collections import defaultdict
from math import hypot
import re
import argparse
import os

WINDOW_SECONDS = 10
RANK = "Ancient"
NEARBY_RADIUS = 20  # Units considered "nearby"

def load_timeline(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def distance(p1, p2):
    return hypot(p1[0] - p2[0], p1[1] - p2[1])

def normalize(pos, origin):
    return [round(pos[0] - origin[0], 2), round(pos[1] - origin[1], 2)]

def normalize_hero_unit(name):
    """
    Convert combat log 'targetname' like 'npc_dota_hero_chaos_knight'
    into 'CDOTA_Unit_Hero_Chaos_Knight' to match interval unit names.
    """
    if not name.startswith("npc_dota_hero_"):
        return None
    suffix = name.replace("npc_dota_hero_", "")
    camel = "_".join([part.capitalize() for part in suffix.split("_")])
    return f"CDOTA_Unit_Hero_{camel}"

def extract_features_verbose(timeline, death_time, hero_unit,hero_name,hero_name_to_unit,killer_name,  verbose=False):
    window_start = death_time - WINDOW_SECONDS
    raw_buckets = defaultdict(lambda: {
        "player_pos": None,
        "player_damage_taken": 0,
        "player_died": False,
        "player_level": None,
        "player_hp_pct": None,
        "teammates": {},
        "enemies": {},
        "damage_events": []
    })

    # Maps modifier name to a list of (start_time, end_time)
    incapacitation_periods = []
    active_incapacitations = {}  # modifier -> start_time

    hero_ids = set()
    unit_to_name = {}

    for e in timeline:
        if e['type'] == "interval" and e['unit'].startswith("CDOTA_Unit_Hero_"):
            hero_ids.add(e['unit'])
            unit_to_name[e['unit']] = e.get("name", e['unit'].replace("CDOTA_Unit_", "npc_dota_"))

    hero_ids = list(hero_ids)
    hero_ids.sort()
    enemy_ids = [hid for hid in hero_ids if hid != hero_unit][:5]
    teammate_ids = [hid for hid in hero_ids if hid != hero_unit and hid not in enemy_ids][:4]

    #print("[DEBUG] HERO_UNIT:", hero_ids)
    #print("[DEBUG] Enemy IDs:", enemy_ids)
    #print("[DEBUG] Teammate IDs:", teammate_ids)

    def detect_incapacitated(event):
        if event.get("type") != "DOTA_COMBATLOG_MODIFIER_ADD":
            return False
        inflictor = event.get("inflictor", "")
        if not inflictor:
            return False
        return any(kw in inflictor.lower() for kw in ["stun", "fear", "bash", "silence", "hex", "gaze", "disable"])


    for e in timeline:
        t = e["time"]
        if t < window_start or t > death_time:
            continue
        bucket = raw_buckets[t]

        if e['type'] == "DOTA_COMBATLOG_MODIFIER_ADD" and e.get("targetname") == hero_name:
            inflictor = e.get("inflictor", "")
            if inflictor and any(kw in inflictor.lower() for kw in ["stun", "fear", "bash", "silence", "hex", "gaze", "disable"]):
                active_incapacitations[inflictor] = t

        if e['type'] == "DOTA_COMBATLOG_MODIFIER_REMOVE" and e.get("targetname") == hero_name:
            inflictor = e.get("inflictor", "")
            if inflictor in active_incapacitations:
                start = active_incapacitations.pop(inflictor)
                incapacitation_periods.append((start, t))

        if e['type'] == "interval":
            if e['unit'] == hero_unit:
                bucket["player_pos"] = (e['x'], e['y'])
                bucket["player_level"] = e.get("level")
                hp = e.get("health", 0)
                max_hp = e.get("maxHealth", 1)
                bucket["player_hp_pct"] = round(hp / max_hp, 3) if isinstance(max_hp, (int, float)) and max_hp > 0 else 0

            elif e['unit'] in teammate_ids:
                bucket["teammates"].setdefault(e['unit'], {})["pos"] = (e['x'], e['y'])
                bucket["teammates"][e['unit']]["level"] = e.get("level")
                hp = e.get("health", 0)
                max_hp = e.get("maxHealth", 1)
                bucket["teammates"][e['unit']]["hp_pct"] = round(hp / max_hp, 3) if isinstance(max_hp, (int, float)) and max_hp > 0 else 0

            elif e['unit'] in enemy_ids:
                bucket["enemies"].setdefault(e['unit'], {})["pos"] = (e['x'], e['y'])
                bucket["enemies"][e['unit']]["level"] = e.get("level")
                hp = e.get("health", 0)
                max_hp = e.get("maxHealth", 1)
                bucket["enemies"][e['unit']]["hp_pct"] = round(hp / max_hp, 3) if isinstance(max_hp, (int, float)) and max_hp > 0 else 0


        if e['type'] == "DOTA_COMBATLOG_DAMAGE":
            targetname = e.get("targetname", "")
            matched_unit = hero_name_to_unit.get(targetname)

            #if e['targetname'] == hero_name:
            #    bucket["player_damage_taken"] += e['value']
            #print(f"[DEBUG] Checking damage event: {targetname} → {matched_unit} vs {hero_unit}")


            if matched_unit == hero_unit:
                #print(f"✅ MATCHED HERO — Adding {e['value']} to player_damage_taken")
                bucket["player_damage_taken"] += e['value']
            if verbose:
                    norm_target = normalize_hero_unit(e.get("targetname", ""))
                    #if True:
                    if "creep" not in e.get("attackername") and "creep" not in e.get("targetname"):
                        bucket["damage_events"].append({
                            "attackername": e.get("attackername"),
                            "sourcename": e.get("sourcename"),
                            "value": e.get("value"),
                            "inflictor": e.get("inflictor"),
                            "targetname": e.get("targetname")
                        })

            if e['targetname'] in [name.replace("CDOTA_Unit_", "npc_dota_") for name in teammate_ids + enemy_ids]:
                norm_name = "CDOTA_Unit_" + e['targetname'].replace("npc_dota_", "")
                target_group = bucket["teammates"] if norm_name in teammate_ids else bucket["enemies"]
                target_group.setdefault(norm_name, {}).setdefault("damage_taken", 0)
                target_group[norm_name]["damage_taken"] += e['value']
                if verbose:
                    norm_target = normalize_hero_unit(e.get("targetname", ""))
                    if "creep" not in e.get("attackername") and "creep" not in e.get("targetname"):
                        bucket["damage_events"].append({
                            "attackername": e.get("attackername"),
                            "sourcename": e.get("sourcename"),
                            "value": e.get("value"),
                            "inflictor": e.get("inflictor"),
                            "targetname": e.get("targetname")
                        })

        if e['type'] == "DOTA_COMBATLOG_DEATH":
            #if e['targetname'] == hero_name:
            if normalize_name(e.get("targetname")) == normalize_name(hero_name):
                bucket["player_died"] = True
            #elif e['targetname'] in [name.replace("CDOTA_Unit_", "npc_dota_") for name in teammate_ids + enemy_ids]:
            elif normalize_hero_unit(e.get("targetname", "")) in teammate_ids + enemy_ids:
                norm_name = "CDOTA_Unit_" + e['targetname'].replace("npc_dota_", "")
                target_group = bucket["teammates"] if norm_name in teammate_ids else bucket["enemies"]
                target_group.setdefault(norm_name, {})["died"] = True

    incapacitated_ticks = set()
    for start, end in incapacitation_periods:
        for tick in range(start, end + 1):
            incapacitated_ticks.add(tick)

    output = {}
    for t, v in sorted(raw_buckets.items()):
        player_pos = v["player_pos"]
        if not player_pos:
            continue

        teammates_pos, teammates_damage, teammates_died, teammates_levels, teammates_hp, teammates_names = [], [], [], [], [], []

        #print(f"[DEBUG] Processing t={t}, player_pos={player_pos}")
        #print(f"[DEBUG] teammate_ids: {teammate_ids}")
        #print(f"[DEBUG] v['teammates']: {list(v['teammates'].keys())}")

        #for tid in teammate_ids:
        #    data = v["teammates"].get(tid)
        #    if not data:
        #        print(f"  [MISSING] No data for teammate {tid}")
        #    elif "pos" not in data:
        #        print(f"  [MISSING] No pos for teammate {tid}")
        #    elif distance(player_pos, data["pos"]) > NEARBY_RADIUS:
        #        print(f"  [TOO FAR] {tid} is too far: {distance(player_pos, data['pos'])}")
        #    else:
        #        print(f"  [OK] {tid} will be added with pos={data['pos']}")

        for tid in teammate_ids:
            data = v["teammates"].get(tid)
            if data and "pos" in data and distance(player_pos, data["pos"]) <= NEARBY_RADIUS:
                
                teammates_pos.append(normalize(data["pos"], player_pos))
                #print(f"Data pos: {data['pos']}  player_pos: {player_pos}  Normalized Pos: {normalize(data['pos'], player_pos)}")

                teammates_damage.append(data.get("damage_taken", 0))
                teammates_died.append(data.get("died", False))
                teammates_levels.append(data.get("level", 1))
                teammates_hp.append(data.get("hp_pct", 0))
                if verbose:
                    teammates_names.append(unit_to_name.get(tid, tid))

        enemies_pos, enemies_damage, enemies_died, enemies_levels, enemies_hp, enemies_names = [], [], [], [], [], []
        for eid in enemy_ids:
            data = v["enemies"].get(eid)
            if data and "pos" in data and distance(player_pos, data["pos"]) <= NEARBY_RADIUS:
                #print(f"Data pos: {data['pos']}  player_pos: {player_pos}  Normalized Pos: {normalize(data['pos'], player_pos)}")
                enemies_pos.append(normalize(data["pos"], player_pos))
                enemies_damage.append(data.get("damage_taken", 0))
                enemies_died.append(data.get("died", False))
                enemies_levels.append(data.get("level", 1))
                enemies_hp.append(data.get("hp_pct", 0))
                if verbose:
                    enemies_names.append(unit_to_name.get(eid, eid))

        output[str(float(t))] = {
            "player_pos": [0.0, 0.0],
            "teammates_pos": teammates_pos,
            "enemies_pos": enemies_pos,
            "player_damage_taken": v["player_damage_taken"],
            "teammates_damage_taken": teammates_damage,
            "enemies_damage_taken": enemies_damage,
            "player_died": v["player_died"],
            "teammates_died": teammates_died,
            "enemies_died": enemies_died,
            "player_level": v.get("player_level", 1),
            "teammates_levels": teammates_levels,
            "enemies_levels": enemies_levels,
            "player_hp_pct": v.get("player_hp_pct", 0),
            "teammates_hp_pct": teammates_hp,
            "enemies_hp_pct": enemies_hp,
            "player_incapacitated": t in incapacitated_ticks


        }

        #print(f"teammates_pos: {teammates_pos}  enemies_pos: {enemies_pos}")


        if verbose:
            output[str(float(t))]["teammates_names"] = teammates_names
            output[str(float(t))]["enemies_names"] = enemies_names
            output[str(float(t))]["damage_events"] = v["damage_events"]


    return output

def interpolate_time_series(features: dict, factor: int) -> dict:
    if factor <= 0:
        return features

    new_features = {}
    times = sorted(float(t) for t in features.keys())

    for i in range(len(times) - 1):
        t1 = times[i]
        t2 = times[i + 1]
        t1_str = str(t1)
        t2_str = str(t2)
        new_features[t1_str] = features[t1_str]

        for j in range(1, factor + 1):
            alpha = j / (factor + 1)
            interp_time = round(t1 + (t2 - t1) * alpha, 1)
            interp_key = str(interp_time)
            entry1 = features[t1_str]
            entry2 = features[t2_str]

            def interp_pos(p1, p2):
                return [round(p1[k] + alpha * (p2[k] - p1[k]), 2) for k in range(2)]

            def interp_list(list1, list2):
                return [interp_pos(list1[i], list2[i]) for i in range(min(len(list1), len(list2)))]

        new_features[interp_key] = {
            "player_pos": [0.0, 0.0],
            "teammates_pos": interp_list(entry1["teammates_pos"], entry2["teammates_pos"]),
            "enemies_pos": interp_list(entry1["enemies_pos"], entry2["enemies_pos"]),

            "player_damage_taken": 0,
            #"teammates_damage_taken": [0] * len(entry1["teammates_pos"]),
            #"enemies_damage_taken": [0] * len(entry1["enemies_pos"]),
            "player_died": False,
            #"teammates_died": [False] * len(entry1["teammates_pos"]),
            #"enemies_died": [False] * len(entry1["enemies_pos"]),

            "player_level": entry1["player_level"],
            "teammates_levels": entry1["teammates_levels"],
            "enemies_levels": entry1["enemies_levels"],

            "player_hp_pct": entry1.get("player_hp_pct", 0),
            "teammates_hp_pct": entry1.get("teammates_hp_pct", [0] * len(entry1["teammates_pos"])),
            "enemies_hp_pct": entry1.get("enemies_hp_pct", [0] * len(entry1["enemies_pos"])),
            "player_incapacitated": entry1.get("player_incapacitated", False)
        }

    last_time = str(times[-1])
    new_features[last_time] = features[last_time]
    return dict(sorted(new_features.items()))

def detect_survivals(timeline, hero_name, hero_unit, hero_name_to_unit,window_seconds=10, damage_threshold=0.5, grace_period=5, verbose_flag=False):

    #print("hello from survivals1")
    survivals = []
    health_by_time = {}
    damage_events = defaultdict(int)
    death_times = set()
    # Pre-compute player deaths
    for e in timeline:
        if e['type'] == "DOTA_COMBATLOG_DEATH" and e.get("targetname") == hero_name:
            death_times.add(e["time"])

    for e in timeline:
        t = e["time"]
        if e['type'] == "interval" and e.get("unit") == hero_unit:
            if "health" in e and isinstance(e.get("maxHealth"), (int, float)) and e["maxHealth"] > 0:
                hp_pct = e["health"] / e["maxHealth"]
                health_by_time[t] = hp_pct

        if e['type'] == "DOTA_COMBATLOG_DAMAGE" and e.get("targetname") == hero_name.replace("CDOTA_Unit_", "npc_dota_"):
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
                clean = True
                for future_t in range(t2 + 1, t2 + grace_period + 1):
                    if damage_events.get(future_t, 0) > 0:
                        clean = False
                        break

                if clean:
                    # 💥 Ensure player did not die during or near this survival window
                    invalid = any(
                        dt in range(t1 - 2, max(t2 + grace_period + 1, t2 + window_seconds + 1))
                        for dt in death_times
                    )
                    if invalid:
                        #print(f"[DEBUG] Skipping survival at {t2} due to nearby or follow-up death at {death_times}")
                        break

                    #print("hello from survivals2")
                    features = extract_features_verbose(timeline, t2, hero_unit,hero_name,hero_name_to_unit,killer_name="", verbose=verbose_flag)
                    survivals.append((t2, features))
                    i = j + grace_period
                    break
        i += 1

    return survivals

#def save_sample(feature_dict, path):
#    with open(path, "w") as f:
#        json.dump(feature_dict, f, indent=2, separators=(",", ": "))

def save_sample(feature_dict, path, rank,label):
    # feature_dict: dict[str(t): dict[feature -> value]]
    # label: 1 for death, 0 for survival

    # Sort by timestamp and extract only the per-second feature vectors

    #print("[DEBUG] sample input keys:", feature_dict[list(feature_dict.keys())[0]].keys())



    time_ordered_features = [
        v for k, v in sorted(feature_dict.items(), key=lambda item: float(item[0]))
    ]

    filtered_features = []
    for entry in time_ordered_features:
        filtered = {}
        entry["average_rank"] = rank
        for k, v in entry.items():
            if isinstance(v, (int, float, bool)):
                filtered[k] = v
            elif isinstance(v, list):
                if all(isinstance(x, (int, float)) for x in v):
                    filtered[k] = v
                elif all(isinstance(x, list) and len(x) == 2 and all(isinstance(i, (int, float)) for i in x) for x in v):
               # Allow list of [float, float] pairs (for positions)
                    filtered[k] = v
        
        filtered_features.append(filtered)

    # Save as proper training sample
    for i, entry in enumerate(filtered_features):
        #print(f"[DEBUG BEFORE PADDING] t={i}, teammates_pos: {entry.get('teammates_pos')}")
        #pad_features({0: entry})  # wrapped to reuse existing logic
        pad_features_single(entry)

        #print(f"[DEBUG AFTER PADDING]  t={i}, teammates_pos: {entry.get('teammates_pos')}")

    output = {
        "features": filtered_features,
        "label": label
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)

def normalize_name(name):
    return name.replace("_", "").lower() if name else ""

def process_deaths(timeline, hero_name, hero_unit, hero_name_to_unit,verbose_flag=False):
    """
    Return a list of (death_time, features_dict) pairs for separate saving.
    """

    
    samples = []
    for event in timeline:
        #if event['type'] == "DOTA_COMBATLOG_DEATH" and event.get("targetname") == hero_name:
        if event['type'] == "DOTA_COMBATLOG_DEATH" and normalize_name(event.get("targetname")) == normalize_name(hero_name):
            death_time = event["time"]
            killer_name = event.get("attackername", "")
            
            
            features = extract_features_verbose(timeline, death_time, hero_unit,hero_name,hero_name_to_unit,killer_name, verbose=verbose_flag)
            samples.append((death_time, features))
    return samples

def cdota_to_npc(cdota_unit):
    # Converts CDOTA_Unit_Hero_ChaosKnight -> npc_dota_hero_chaos_knight
    suffix = cdota_unit.replace("CDOTA_Unit_Hero_", "")
    parts = re.findall(r'[A-Z][a-z]*', suffix)
    return "npc_dota_hero_" + "_".join(p.lower() for p in parts)

def extract_hero_pairs_from_timeline(timeline, hero_list):
    # Step 1: Find npc_dota_hero_* names from early ability level events
    hero_names = set()
    for entry in timeline:
        if entry.get("type") == "DOTA_ABILITY_LEVEL":
            name = entry.get("targetname")
            if name and name.startswith("npc_dota_hero_"):
                hero_names.add(name)

    # Step 2: Match with CDOTA_Unit_* from known hero_list
    hero_pairs = []
    for unit in hero_list:
        npc_name = cdota_to_npc(unit)
        hero_pairs.append((npc_name, unit))

    print(f"[DEBUG] Hero name/unit pairs: {hero_pairs}")
    return hero_pairs



def process_file(input_path, interp_factor=0, verbose_arg=False):
    timeline = load_timeline(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    base_out_dir = os.path.join("../../OutputJsons/DeathsAndSurvivals", base_name)
    out_dir_deaths = os.path.join(base_out_dir, "Deaths")
    out_dir_survivals = os.path.join(base_out_dir, "Survivals")
    os.makedirs(out_dir_deaths, exist_ok=True)
    os.makedirs(out_dir_survivals, exist_ok=True)

    # === Extract all heroes once ===
    hero_units = set()
    for e in timeline:
        if e['type'] == "interval" and e['unit'].startswith("CDOTA_Unit_Hero_"):
            hero_units.add(e['unit'])

    hero_pairs = extract_hero_pairs_from_timeline(timeline, hero_units)
    hero_name_to_unit = {npc: unit for npc, unit in hero_pairs}

    #print(f"[DEBUG] Hero name/unit pairs: {hero_pairs}")

    total_deaths = 0
    total_survivals = 0

    
    match = re.search(r"_RT(\d)", input_path)
    rank_digit = int(match.group(1)[0]) if match else -1
   
    for hero_name, hero_unit in hero_pairs:
        #if hero_name != "npc_dota_hero_chaos_knight":
        #    continue  # ⛔ Skip all heroes except Chaos Knight
        # === Deaths ===
        deaths = process_deaths(timeline, hero_name, hero_unit, hero_name_to_unit,verbose_flag=verbose_arg)
        for death_time, features in deaths:
            if interp_factor > 0:
                features = interpolate_time_series(features, interp_factor)
            short_name = hero_name.replace("npc_dota_hero_", "")
            out_path = os.path.join(out_dir_deaths, f"{base_name}_death_{short_name}_{death_time}.json")
            #print(f"[DEBUG] t={death_time}, teammates_pos before save:", features.get(str(float(death_time)), {}).get("teammates_pos"))
            save_sample(features, out_path,rank_digit,label=1)
            print(f"💀 Saved death sample at: {out_path}")
        total_deaths += len(deaths)

        # === Survivals ===
        survivals = detect_survivals(timeline, hero_name, hero_unit, hero_name_to_unit,verbose_flag=verbose_arg)
        for surv_time, features in survivals:
            if interp_factor > 0:
                features = interpolate_time_series(features, interp_factor)
            short_name = hero_name.replace("npc_dota_hero_", "")
            out_path = os.path.join(out_dir_survivals, f"{base_name}_surv_{short_name}_{surv_time}.json")
            save_sample(features, out_path,rank_digit,label=0)
            print(f"🛡️ Saved survival sample at: {out_path}")
        total_survivals += len(survivals)

    print(f"✅ {base_name}: {total_deaths} deaths and {total_survivals} survivals saved")

def pad_features_single(entry, teammate_count=4, enemy_count=5):
    for key_base in ["pos", "damage_taken", "died", "levels", "hp_pct"]:
        for role, count in [("teammates", teammate_count), ("enemies", enemy_count)]:
            full_key = f"{role}_{key_base}"
            values = entry.get(full_key, [])

            if not isinstance(values, list):
                values = []

            if "pos" in key_base:
                pad_value = [0.0, 0.0]
            elif "died" in key_base:
                pad_value = False
            else:
                pad_value = -1

            if len(values) < count:
                values += [pad_value] * (count - len(values))
            else:
                values = values[:count]

            entry[full_key] = values
    
    # Apply similar padding logic to any other teammate/enemy fields if needed

def pad_features(features_dict, teammate_count=4, enemy_count=5):
    for t, entry in features_dict.items():
        for key_base in ["pos", "damage_taken", "died", "levels", "hp_pct"]:
            for role, count in [("teammates", teammate_count), ("enemies", enemy_count)]:
                full_key = f"{role}_{key_base}"
                values = entry.get(full_key, [])

                if "pos" in key_base:
                    pad_value = [0.0, 0.0]
                elif "died" in key_base:
                    pad_value = False
                else:
                    pad_value = -1

                if len(values) < count:
                    values += [pad_value] * (count - len(values))
                else:
                    values = values[:count]
                entry[full_key] = values

def main():
    parser = argparse.ArgumentParser(description="Generate per-death and per-survival files from one or multiple timeline files.")
    parser.add_argument("input_path", help="Path to a .jsonl file or a folder of them")
    parser.add_argument("--interp", type=int, default=1, help="Number of interpolated entries between each second")
    parser.add_argument("--verbose", action="store_true", help="Include extra debug info like names, hp, damage source")
    args = parser.parse_args()

    input_path = args.input_path
    interp_factor = args.interp
    verbose_arg = args.verbose

    if os.path.isdir(input_path):
        jsonl_files = [f for f in os.listdir(input_path) if f.endswith(".jsonl")]
        for file in jsonl_files:
            full_path = os.path.join(input_path, file)
            process_file(full_path, interp_factor, verbose_arg)
    else:
        process_file(input_path, interp_factor, verbose_arg)

if __name__ == "__main__":
    main()
