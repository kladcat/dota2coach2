import os
import sys
import subprocess

def parse_file(dem_path, output_dir):
    base_name = os.path.splitext(os.path.basename(dem_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.jsonl")

    print(f"Processing: {dem_path} → {output_path}")

    try:
        result = subprocess.run(
            [
                "curl", "-s",
                "-X", "POST",
                "--data-binary", f"@{dem_path}",
                "http://localhost:5600"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120
        )

        if result.returncode == 0:
            with open(output_path, 'wb') as f:
                f.write(result.stdout)
            print(f"✅ Written: {output_path}")
        else:
            print(f"❌ Curl failed for {dem_path}: {result.stderr.decode()}")
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout while processing {dem_path}")
    except Exception as e:
        print(f"❌ Error while processing {dem_path}: {e}")

def parse_path(input_path, output_dir="./OutputJson"):
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(input_path) and input_path.endswith(".dem"):
        parse_file(input_path, output_dir)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".dem"):
                    full_path = os.path.join(root, file)
                    parse_file(full_path, output_dir)
    else:
        print(f"❌ Invalid input path: {input_path}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dota2_clarity_dem_parser.py <.dem file or folder>")
        sys.exit(1)

    input_path = sys.argv[1]
    parse_path(input_path)
