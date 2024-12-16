#!/bin/bash

# Script Name: delete_directories.sh
# Description: Deletes specified directories in a given base directory,
#              excluding 'Example-JSON-files', 'usefuls', and 'todays-matches'.
# Usage: ./delete_directories.sh /path/to/base/directory

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display usage information
usage() {
    echo "Usage: $0 /path/to/base/directory"
    exit 1
}

# Check if exactly one argument is provided
if [ "$#" -ne 1 ]; then
    echo "Error: Base directory path is required."
    usage
fi

BASE_DIR="$1"

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory does not exist: $BASE_DIR"
    exit 1
else
    echo "Base directory exists: $BASE_DIR"
fi

# Array of directories to delete (excluding the specified ones)
DIRS=(
    "bundesliga-1"
    "bundesliga-2"
    "champions-league"
    "community-shield"
    "conference-league"
    "efl-championship"
    "efl-league-1"
    "efl-league-2"
    "england-league-cup"
    "fa-cup"
    "liga-nos"
    "liga-pro"
    "ligue-1"
    "ligue-2"
    "netherlands-league-1"
    "netherlands-league-2"
    "premier-league"
    "scottish-premiership"
    "segunda-division"
    "serie-a"
    "serie-b"
    "europa-league"
    "la-liga"
)

# Iterate over the array and delete each directory
for DIR in "${DIRS[@]}"; do
    TARGET_DIR="$BASE_DIR/$DIR"
    if [ -d "$TARGET_DIR" ]; then
        rm -rf "$TARGET_DIR"
        echo "Deleted directory: $TARGET_DIR"
    else
        echo "Directory does not exist, skipping: $TARGET_DIR"
    fi
done

echo "All specified directories have been processed for deletion."
