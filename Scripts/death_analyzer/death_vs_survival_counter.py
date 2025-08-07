import os

# Change this to the path where OutputJsons is located
ROOT_DIR = "../../OutputJsons/DeathsAndSurvives"

death_count = 0
survival_count = 0

for root, dirs, files in os.walk(ROOT_DIR):
    for dir_name in dirs:
        dir_path = os.path.join(root, dir_name)
        if dir_name == "Deaths":
            death_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            print(f"Found {len(death_files)} death files in: {dir_path}")
            death_count += len(death_files)
        elif dir_name == "Survivals":
            survival_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            print(f"Found {len(survival_files)} survival files in: {dir_path}")
            survival_count += len(survival_files)

print("\n===========================")
print(f"Total Deaths: {death_count}")
print(f"Total Survivals: {survival_count}")
print("===========================\n")

if death_count + survival_count > 0:
    ratio = death_count / (death_count + survival_count)
    print(f"Death ratio: {ratio:.2%}")
