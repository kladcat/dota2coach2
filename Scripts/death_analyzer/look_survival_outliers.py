import os
import json
import argparse

# Path to the 'Deaths' and 'Survivals' folder
root_dir = "../../OutputJsons/DeathsAndSurvives"

def check_survival_files(delete_files=False):
    deleted_files = []  # List to store deleted files
    total_files = 0  # Counter for total files processed
    deleted_count = 0  # Counter for deleted files
    faulty_files= 0
    # Loop through all subdirectories in the root directory
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json") and "_surv_" in file:  # Process only JSON files
                total_files += 1
                # Get the full path of the file
                file_path = os.path.join(subdir, file)
                
                # Open and read the JSON file
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Check the "player_hp_pct" for each entry in the JSON data
                for timestamp, entry in data.items():
                    player_hp_pct = entry.get("player_hp_pct", None)
                    if player_hp_pct is not None and player_hp_pct < 0.1:
                        print(f"Found survival with low HP: {file_path} - Time: {timestamp} - HP: {player_hp_pct}")
                        faulty_files +=1
                        # If delete_files is True, delete the file
                        if delete_files:
                            os.remove(file_path)
                            deleted_files.append(file_path)  # Add to deleted files list
                            deleted_count += 1
                        break  # Stop once we've found a low HP value

    # Print summary of deleted files
    print(f"\nTotal survival files processed: {total_files}")
    print(f"Total faulty files: {faulty_files}")
    print(f"Total deleted files: {deleted_count}")
    if deleted_count > 0:
        print("\nDeleted files:")
        for deleted_file in deleted_files:
            print(deleted_file)

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Process survival files and optionally delete those with low HP.")
    parser.add_argument(
        '--delete', 
        action='store_true', 
        help="Delete files with low HP entries (default: False)"
    )
    
    # Parse arguments
    args = parser.parse_args()

    # Run the function to check survival files, with the delete option controlled by the command-line argument
    check_survival_files(delete_files=args.delete)

if __name__ == "__main__":
    main()
