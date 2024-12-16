import csv
import json
import os
import requests
from api import KEY  # Ensure api.py contains your API key

# Configuration
INPUT_CSV = '/root/barnard/data/betting/usefuls/combined-season-ids.csv'  # Path to your input CSV file
OUTPUT_DIR = '/root/barnard/data/betting/usefuls/League-Difficulties'    # Directory where output CSVs will be saved
API_BASE_URL = 'https://api.football-data-api.com/league-season'

def make_api_call(season_id):
    """
    Makes an API call to fetch league season data for a given season_id.
    
    Args:
        season_id (int): The season ID to fetch data for.
    
    Returns:
        dict: The JSON response from the API if successful, else None.
    """
    params = {
        'key': KEY,
        'season_id': season_id
    }
    try:
        response = requests.get(API_BASE_URL, params=params)
        response.raise_for_status()  # Raises HTTPError for bad responses
        data = response.json()
        if data.get('success'):
            return data.get('data', {})
        else:
            print(f"API response unsuccessful for season_id {season_id}.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed for season_id {season_id}: {e}")
        return None

def sanitize_filename(filename):
    """
    Sanitizes the filename by replacing or removing invalid characters.
    
    Args:
        filename (str): The original filename.
    
    Returns:
        str: The sanitized filename.
    """
    return filename.replace('/', '-').replace('\\', '-')

def main():
    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Dictionary to keep track of opened file handles
    file_handles = {}
    csv_writers = {}
    
    try:
        with open(INPUT_CSV, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                year = row['year'].strip()
                season_id = row['id'].strip()
                league_name = row['league-name'].strip()
                
                # Validate season_id
                if not season_id.isdigit():
                    print(f"Invalid season_id '{season_id}' for year '{year}'. Skipping.")
                    continue
                season_id = int(season_id)
                
                # Make the API call
                api_data = make_api_call(season_id)
                if not api_data:
                    print(f"No data retrieved for season_id {season_id}. Skipping.")
                    continue
                
                # Extract required fields
                extracted_data = {
                    'id': api_data.get('id', ''),
                    'league-name': api_data.get('name', ''),
                    'season': year,
                    'international_scale': api_data.get('international_scale', ''),
                    'domestic_scale': api_data.get('domestic_scale', '')
                }
                
                # Prepare the output filename
                sanitized_year = sanitize_filename(year)
                output_filename = f"{sanitized_year}-league-difficulties.csv"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                # If the file is not already open, open it and write the header
                if output_path not in file_handles:
                    file = open(output_path, mode='w', newline='', encoding='utf-8')
                    file_handles[output_path] = file
                    writer = csv.DictWriter(file, fieldnames=extracted_data.keys())
                    writer.writeheader()
                    csv_writers[output_path] = writer
                else:
                    writer = csv_writers[output_path]
                
                # Write the extracted data
                writer.writerow(extracted_data)
                print(f"Data for season_id {season_id} written to {output_filename}")
    finally:
        # Close all opened file handles
        for file in file_handles.values():
            file.close()
        print("All files have been saved successfully.")

if __name__ == "__main__":
    main()
