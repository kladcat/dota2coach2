import requests
import os
import time
import argparse

# === CONFIG ===
HERO_ID = 81  # Chaos Knight
MATCH_LIMIT = 15
DOWNLOAD_FOLDER = "../Replays/ck_replays/new"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# === CLI ARGUMENTS ===
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["download", "request", "list"], default="download",
                    help="download = get .dem.bz2 | request = ask OpenDota to parse | list = show match info only")
args = parser.parse_args()

# === HELPERS ===
def get_public_matches(less_than=None):
    url = "https://api.opendota.com/api/publicMatches"
    if less_than:
        url += f"?less_than_match_id={less_than}"
    return requests.get(url).json()

def get_match_details(match_id):
    r = requests.get(f"https://api.opendota.com/api/matches/{match_id}")
    if r.status_code == 200:
        return r.json()
    return None

def request_parse(match_id):
    print(f"📡 Requesting OpenDota to parse match {match_id}...")
    r = requests.post(f"https://api.opendota.com/api/request/{match_id}")
    if r.status_code == 200:
        print(f"✅ Request accepted: {r.json()}")
    else:
        print(f"❌ Failed to request parse for {match_id}: {r.status_code}")

def download_replay(match_id, replay_salt, cluster, avg_rank_tier):
    filename = f"{match_id}_{replay_salt}_RT{avg_rank_tier}.dem.bz2"
    local_path = os.path.join(DOWNLOAD_FOLDER, filename)
    url = f"http://replay{cluster}.valve.net/570/{match_id}_{replay_salt}.dem.bz2"
    print(f"⬇️  Downloading: {url}")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Saved: {local_path}")
    else:
        print(f"❌ Failed: {url} (status {r.status_code})")

# === MAIN LOOP ===
collected = 0
seen_matches = set()
last_match_id = None

while collected < MATCH_LIMIT:
    batch = get_public_matches(less_than=last_match_id)
    if not batch:
        break
    for match in batch:
        match_id = match["match_id"]
        if match_id in seen_matches:
            continue
        seen_matches.add(match_id)
        last_match_id = match_id

        if HERO_ID not in match.get("radiant_team", []) and HERO_ID not in match.get("dire_team", []):
            continue

        avg_rank_tier = match.get("avg_rank_tier", 0)
        duration = match.get("duration", 0)
        game_mode = match.get("game_mode")
        if avg_rank_tier < 10 or duration < 900 or game_mode != 22:
            continue

        if args.mode == "list":
            side = "Radiant" if HERO_ID in match["radiant_team"] else "Dire"
            result = "Win" if (match["radiant_win"] and side == "Radiant") or (not match["radiant_win"] and side == "Dire") else "Loss"
            print(f"📄 Match {match_id} — CK on {side} — {result} — RT{avg_rank_tier}")
            collected += 1
            continue

        if args.mode == "request":
            request_parse(match_id)
            collected += 1
            continue

        if args.mode == "download":
            details = get_match_details(match_id)
            if not details or not details.get("replay_salt"):
                print(f"⚠️  No replay available for {match_id}")
                continue
            download_replay(match_id, details["replay_salt"], details["cluster"], avg_rank_tier)
            collected += 1

        if collected >= MATCH_LIMIT:
            break
    time.sleep(1)
