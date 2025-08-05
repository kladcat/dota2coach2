import json
import math
from collections import defaultdict

RANGE = 1000

# === CONFIG ===
INPUT_TIMELINE_PATH = "../OutputJsons/MinimizedTimelines/CKReplays/8378467054.jsonl"
OUTPUT_PATH = "../OutputJsons/LastHitFeedbackTimelines/last_hit_feedback.json"

# Adjust depending on the player's hero and slot
PLAYER_HERO = "CDOTA_Unit_Hero_ChaosKnight"
PLAYER_SLOT = 2  # 0–4 = Radiant, 5–9 = Dire

def distance(a, b):
    return math.hypot(a['x'] - b['x'], a['y'] - b['y'])

def is_creep(unitname):
    return "creep" in unitname and "hero" not in unitname

def get_team(slot):
    return "radiant" if slot <= 4 else "dire"

def run_last_hit_analyzer(timeline, hero_name="CDOTA_Unit_Hero_ChaosKnight", slot=2):
    feedback = []
    player_team = get_team(slot)
    positions_by_time = defaultdict(dict)  # {time: {unit_name: {x, y}}}

    for event in timeline:
        if event["type"] == "interval":
            unit = event["unit"]
            positions_by_time[event["time"]][unit] = {"x": event["x"], "y": event["y"]}

    for event in timeline:
        if event["type"] != "DOTA_COMBATLOG_DEATH":
            continue

        if not is_creep(event["targetname"]):
            continue

        t = event["time"]
        positions = positions_by_time.get(t, {})

        # Make sure our hero is on screen at this moment
        if hero_name not in positions:
            continue

        hero_pos = positions[hero_name]
        creep_pos = None

        # Try to estimate creep position by matching hero that killed it
        attacker = event.get("attackername")
        if attacker in positions:
            creep_pos = positions[attacker]
        else:
            continue  # Can't estimate creep position

        if distance(hero_pos, creep_pos) > RANGE:
            continue  # Creep was too far to last-hit

        # Was the last hit done by our hero?
        if attacker == hero_name:
            continue  # You got the last hit, good job

        # Now check for enemy hero presence
        for unit, pos in positions.items():
            if not unit.startswith("CDOTA_Unit_Hero_"):
                continue
            if unit == hero_name:
                continue
            enemy_team = get_team(int(unit.split("_")[-1])) != player_team
            if enemy_team and distance(pos, creep_pos) <= RANGE:
                break  # There was pressure
        else:
            # No break: no enemy hero nearby
            feedback.append({
                "time": t,
                "type": "feedback",
                "message": "❌ Missed last hit without enemy pressure"
            })

    return feedback

def main():
    # Load timeline data
    with open(INPUT_TIMELINE_PATH, "r") as f:
        timeline = [json.loads(line.strip()) for line in f if line.strip()]

    # Run last hit analyzer
    feedback_messages = run_last_hit_analyzer(timeline, hero_name=PLAYER_HERO, slot=PLAYER_SLOT)

    # Save results
    with open(OUTPUT_PATH, "w") as out:
        for entry in feedback_messages:
            json.dump(entry, out)
            out.write("\n")

    print(f"✅ Done! {len(feedback_messages)} last hit issues written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
