import os
import sys

from dota2_clarity_dem_parser import parse_path
from timeline_minimizer import filter_events
from timeline_to_timeseries_single_hero import extract_single_hero_timeseries

# Base output folders
BASE_RAW_JSONL_DIR = "../OutputJsons/FullTimelines/CKReplays"
BASE_MINIMIZED_DIR = "../OutputJsons/MinimizedTimelines/CKReplays"
BASE_TIMESERIES_DIR = "../OutputJsons/TimeSeriesForAverageTier/CKReplays"
BASE_TIMESERIES_FOR_PREDICT_DIR = "../OutputJsons/TimeSeriesToPredictAverageTier"

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def run_pipeline(input_dem_path, target_slot=3, interval_skip=0,
                 use_new_folder=False, predict_mode=False, start_from=1):
    # Add "/New" suffix if needed
    suffix = "/New" if use_new_folder else ""

    raw_jsonl_dir = BASE_RAW_JSONL_DIR + suffix
    minimized_dir = BASE_MINIMIZED_DIR + suffix
    timeseries_base = BASE_TIMESERIES_FOR_PREDICT_DIR if predict_mode else BASE_TIMESERIES_DIR
    timeseries_dir = timeseries_base + suffix

    ensure_dirs(raw_jsonl_dir, minimized_dir, timeseries_dir)

    # Step 1: Parse .dem files to raw .jsonl
    if start_from <= 1:
        print("🌀 Step 1: Parsing .dem to .jsonl ...")
        parse_path(input_dem_path, raw_jsonl_dir)
    else:
        print("⏭️ Skipping Step 1")

    # Step 2: Minimize timelines
    if start_from <= 2:
        print("🔧 Step 2: Minimizing timelines ...")
        if os.path.isdir(input_dem_path):
            # Batch mode
            for file in os.listdir(raw_jsonl_dir):
                if file.endswith(".jsonl"):
                    raw_path = os.path.join(raw_jsonl_dir, file)
                    minimized_path = os.path.join(minimized_dir, file)
                    filter_events(raw_path, minimized_path, interval_skip)
        else:
            # Single file mode
            dem_name = os.path.splitext(os.path.basename(input_dem_path))[0]
            jsonl_name = dem_name + ".jsonl"
            raw_path = os.path.join(raw_jsonl_dir, jsonl_name)
            minimized_path = os.path.join(minimized_dir, jsonl_name)
            if os.path.exists(raw_path):
                filter_events(raw_path, minimized_path, interval_skip)
            else:
                print(f"⚠️  Skipping minimization: {raw_path} not found.")

    # Step 3: Convert to time series
    if start_from <= 3:
        print("📊 Step 3: Extracting time series ...")
        if os.path.isdir(input_dem_path):
            # Batch mode
            for file in os.listdir(minimized_dir):
                if file.endswith(".jsonl"):
                    minimized_path = os.path.join(minimized_dir, file)
                    timeseries_path = os.path.join(
                        timeseries_dir,
                        file.replace(".jsonl", f".slot{target_slot}.json")
                    )
                    extract_single_hero_timeseries(minimized_path, timeseries_path, target_slot)
        else:
            # Single file mode
            dem_name = os.path.splitext(os.path.basename(input_dem_path))[0]
            minimized_name = dem_name + ".jsonl"
            minimized_path = os.path.join(minimized_dir, minimized_name)
            timeseries_path = os.path.join(
                timeseries_dir,
                minimized_name.replace(".jsonl", f".slot{target_slot}.json")
            )
            if os.path.exists(minimized_path):
                extract_single_hero_timeseries(minimized_path, timeseries_path, target_slot)
            else:
                print(f"⚠️  Skipping timeseries: {minimized_path} not found.")
    else:
        print("⏭️ Skipping Step 3")

    print("✅ Pipeline completed!")


def print_help():
    print("""
Dota2Coach Replay Pipeline
===========================

Usage:
  python dota2_pipeline.py <.dem file or folder> [slot] [interval_skip] [--new] [--predict] [--start-from=N]

Required Arguments:
  <.dem file or folder>   Path to a single .dem replay file or a folder containing .dem files

Optional Positional Arguments:
  [slot]                  Hero slot to extract (default: 3). Range is 0–9.
  [interval_skip]         Skip every N interval events to reduce density (default: 0, meaning no skipping)

Optional Flags:
  --new                   Write outputs into a subfolder called "/New" inside each base output directory
  --predict               Write timeseries files to the prediction folder instead of the training folder
  --start-from=N          Skip pipeline stages. Values:
                            1 = run all steps (default)
                            2 = skip .dem → .jsonl conversion
                            3 = skip both conversion and minimization (only generate timeseries)

Example Usage:
--------------
1. Full run with all defaults:
   python dota2_pipeline.py my_replay.dem

2. Use slot 2 and skip interval events (every other tick):
   python dota2_pipeline.py my_replay.dem 2 1

3. Skip the parsing step and start from minimization:
   python dota2_pipeline.py my_replay.dem --start-from=2

4. Output to /New folders and prepare for prediction:
   python dota2_pipeline.py my_replay.dem --new --predict

""")

# Entry point
if __name__ == "__main__":

    if "-h" in sys.argv or "--help" in sys.argv or len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage: python dota2_pipeline.py <.dem file or folder> [slot] [interval_skip] [--new] [--predict] [--start-from=N]")
        sys.exit(1)

    print("Starting pipeline!")
    dem_input = sys.argv[1]
    slot = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 3
    interval_skip = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 0
    use_new = "--new" in sys.argv
    predict_mode = "--predict" in sys.argv

    # Default to 1 unless --start-from=N is provided
    start_from = 1
    for arg in sys.argv:
        if arg.startswith("--start-from="):
            try:
                start_from = int(arg.split("=")[1])
            except ValueError:
                print("❌ Invalid value for --start-from=N (must be integer)")
                sys.exit(1)

    run_pipeline(dem_input, slot, interval_skip, use_new, predict_mode, start_from)
