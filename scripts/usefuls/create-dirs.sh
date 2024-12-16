#!/bin/bash

# Array of main directories
main_dirs=("1718" "1819" "1920" "2021" "2122" "2223" "2324" "2425")

# Array of subdirectories to be created within each main directory
sub_dirs=("csv-files" "league-matches-stats-raw" "league-teams-stats-raw" "players-stats-raw" "referee-stats-raw" "league-table-stats-raw")

# Loop over each main directory
for main_dir in "${main_dirs[@]}"; do
  # Create the main directory
  mkdir -p "$main_dir"
  
  # Loop over each subdirectory
  for sub_dir in "${sub_dirs[@]}"; do
    # Create the subdirectory within the main directory
    mkdir -p "$main_dir/$sub_dir"
  done
done

echo "Directories created successfully."
