import requests
import time

CHAOS_KNIGHT_ID = 81
MATCH_LIMIT = 10  # Total matches you want
matches_collected = []
last_match_id = None

while len(matches_collected) < MATCH_LIMIT:
    url = "https://api.opendota.com/api/publicMatches"
    if last_match_id:
        url += f"?less_than_match_id={last_match_id}"

    r = requests.get(url)
    if r.status_code != 200:
        print("Error fetching data")
        break

    data = r.json()
    if not data:
        break

    for match in data:
        if CHAOS_KNIGHT_ID in match.get("radiant_team", []) or CHAOS_KNIGHT_ID in match.get("dire_team", []):
            matches_collected.append(match)
            if len(matches_collected) >= MATCH_LIMIT:
                break

    last_match_id = data[-1]["match_id"]  # Go backwards
    time.sleep(1)  # Avoid hammering API

print(f"✅ Collected {len(matches_collected)} Chaos Knight matches")

# Optional: print summary
for m in matches_collected:
    side = "Radiant" if CHAOS_KNIGHT_ID in m["radiant_team"] else "Dire"
    won = m["radiant_win"] if side == "Radiant" else not m["radiant_win"]
    print(f"Match {m['match_id']} — CK on {side} — {'Win' if won else 'Loss'} — Rank Tier {m.get('avg_rank_tier')}")
