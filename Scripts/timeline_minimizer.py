import json
import sys
import os

def filter_events(input_path, output_path, interval_skip=0):
    filtered_events = []
    interval_tracker = {}

    with open(input_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed line in {input_path}: ", line)
                continue

            event_type = event.get("type")

            if event_type == "actions":
                continue

            if event_type == "interval":
                unit = event.get("unit")
                time = event.get("time")

                if not unit:
                    continue

                last_full_time = interval_tracker.get(unit, float('-inf'))

                if time % 10 == 0:
                    filtered_events.append(event)
                    interval_tracker[unit] = time
                elif interval_skip == 0 or time % (interval_skip + 1) == 0:
                    minimal = {
                        "time": time,
                        "type": event_type,
                        "unit": unit,
                        "x": event.get("x"),
                        "y": event.get("y"),
                        "health": event.get("health"),
                        "maxHealth": event.get("maxHealth"),
                        "level": event.get("level"),
                        "stuns": event.get("stuns")
                    }
                    filtered_events.append(minimal)
            else:
                filtered_events.append(event)

    with open(output_path, 'w') as out:
        for e in filtered_events:
            out.write(json.dumps(e) + '\n')

    print(f"✅ {input_path} → {output_path} with {len(filtered_events)} events")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python timeline_minimizer.py <input_file_or_folder> <output_file_or_folder> [interval_skip]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    interval_skip = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        for filename in os.listdir(input_path):
            if filename.endswith(".jsonl"):
                in_file = os.path.join(input_path, filename)
                out_file = os.path.join(output_path, filename)
                filter_events(in_file, out_file, interval_skip)
    else:
        filter_events(input_path, output_path, interval_skip)
