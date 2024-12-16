#!/bin/bash

# Check if the directory name is provided as an argument
if [ -z "$1" ]; then
  echo "Error: No directory name provided."
  echo "Usage: $0 <directory-name>"
  exit 1
fi

# Define the base directory
BASE_DIR="/root/barnard/ML/Models/$1"

# Create the main directory and subdirectories
mkdir -p "$BASE_DIR/Datasets"
mkdir -p "$BASE_DIR/Predict"
mkdir -p "$BASE_DIR/Prediction-Results/Match-Result"
mkdir -p "$BASE_DIR/Prediction-Results/Clean-Sheet"
mkdir -p "$BASE_DIR/Prediction-Results/BTTS-FTS"
mkdir -p "$BASE_DIR/Train"
mkdir -p "$BASE_DIR/Training-Results/Match-Result"
mkdir -p "$BASE_DIR/Training-Results/Clean-Sheet"
mkdir -p "$BASE_DIR/Training-Results/BTTS-FTS"

echo "Directory structure created successfully at $BASE_DIR"
