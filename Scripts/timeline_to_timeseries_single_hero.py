import json
import sys
import os
import requests

OPENDOTA_API_BASE = "https://api.opendota.com/api/players/"

def get_opponent_slot(slot):
    return slot + 128 if slot < 5 else slot - 128

def extract_steamid_from_epilogue(input_path, slot):
    """Returns the steamid of the player in the given slot."""
    print(f"🔍 [DEBUG] Looking for SteamID in epilogue of: {input_path}")
    with open(input_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print("⚠️ [DEBUG] Skipping malformed line.")
                continue

            if event.get("type") == "epilogue":
                print("✅ [DEBUG] Epilogue event found.")
                try:
                    epilogue_data = json.loads(event["key"])
                    players = epilogue_data["gameInfo_"]["dota_"]["playerInfo_"]
                    steamid = players[slot]["steamid_"] - 76561197960265728
                    print(f"🎯 [DEBUG] Slot {slot} has SteamID: {steamid}")
                    return steamid
                except Exception as e:
                    print(f"❌ [DEBUG] Failed to extract SteamID from epilogue structure: {e}")
                    return None

    print("⚠️ [DEBUG] Epilogue event not found in timeline.")
    return None


def get_rank_tier_from_opendota(steamid):
    print(f"🌐 [DEBUG] Querying OpenDota for steamid {steamid}")
    try:
        response = requests.get(f"{OPENDOTA_API_BASE}{steamid}")
        if response.ok:
            rank = response.json().get("rank_tier")
            print(f"📈 [DEBUG] OpenDota returned rank_tier: {rank}")
            return rank
        else:
            print(f"❌ [DEBUG] OpenDota API error: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ [DEBUG] Exception during OpenDota request: {e}")
    return None

def extract_single_hero_timeseries(input_path, output_path, target_slot=None):
    timeline = {}
    snapshots_by_time = {}

    with open(input_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("type") != "interval":
                continue

            time = event.get("time")
            slot = event.get("slot")
            if slot is None or time is None:
                continue

            snapshots_by_time.setdefault(time, {})[slot] = event

    for time, players_at_time in snapshots_by_time.items():
        hero_event = players_at_time.get(target_slot)
        if not hero_event:
            continue

        if not all(k in hero_event for k in ["networth", "gold", "xp", "level", "lh", "denies", "kills", "deaths", "assists"]):
            continue

        opponent_slot = get_opponent_slot(target_slot)
        opponent_event = players_at_time.get(opponent_slot)
        enemy_networth = opponent_event.get("networth") if opponent_event else None

        timeline[str(time)] = {
            "networth": hero_event["networth"],
            "gold": hero_event["gold"],
            "xp": hero_event["xp"],
            "level": hero_event["level"],
            "lh": hero_event["lh"],
            "denies": hero_event["denies"],
            "kills": hero_event["kills"],
            "deaths": hero_event["deaths"],
            "assists": hero_event["assists"],
            "enemy_networth": enemy_networth
        }

    # 🔍 Get steamid + rank tier of this slot for OFF_XX
    steamid = extract_steamid_from_epilogue(input_path, target_slot)
    tier_suffix = ""
    if steamid:
        rank_tier = get_rank_tier_from_opendota(steamid)
        if rank_tier:
            tier_suffix = f".OFF_{rank_tier:02d}"
            print(f"🧾 [DEBUG] Appending OFF_{rank_tier:02d} to filename.")
        else:
            print("⚠️ [DEBUG] Rank tier not found, skipping suffix.")
    else:
        print("⚠️ [DEBUG] SteamID not found, skipping suffix.")

    # 🔧 Inject suffix into filename if applicable
    if tier_suffix:
        base, ext = os.path.splitext(output_path)
        output_path = base + tier_suffix + ext

    with open(output_path, 'w') as out:
        json.dump(timeline, out, indent=2)

    print(f"✅ Time series written for slot={target_slot} → {output_path}")

# Usage: python timeline_to_timeseries_singlehero.py input.jsonl output.json [player_slot]
#     or: python timeline_to_timeseries_singlehero.py input_folder output_folder [player_slot]
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python timeline_to_timeseries_singlehero.py <input.jsonl|folder> <output.json|folder> [player_slot]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    slot = int(sys.argv[3]) if len(sys.argv) > 3 else None

    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        for filename in os.listdir(input_path):
            if filename.endswith(".jsonl"):
                in_file = os.path.join(input_path, filename)
                out_file = os.path.join(output_path, filename.replace(".jsonl", f".slot{slot}.json"))
                extract_single_hero_timeseries(in_file, out_file, slot)
    else:
        extract_single_hero_timeseries(input_path, output_path, slot)
