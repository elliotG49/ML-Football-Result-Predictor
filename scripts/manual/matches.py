import os
import csv
import time
import json
import requests
import logging
from pymongo import MongoClient, errors
from api import KEY

# Set up logging
logging.basicConfig(
    filename='/root/barnard/logs/manual_import.log',  # Log file path
    level=logging.INFO,  # Logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)

# MongoDB setup
client = MongoClient('localhost', 27017)
db = client.footballDB  # Replace with your database name
collection = db.matches  # Replace with your collection name

def fetch_and_insert_data(competition_name, year, competition_id):
    """Fetch data from the API and insert it directly into MongoDB."""
    base_url = f'https://api.football-data-api.com/league-matches?key={KEY}&season_id={competition_id}&include=stats'

    all_matches = []
    current_page = 1

    while True:
        league_matches_url = f'{base_url}&page={current_page}'
        logging.info(f"Fetching data from URL: {league_matches_url}")
        response = requests.get(league_matches_url)

        if response.status_code == 200:
            data = response.json()
            all_matches.extend(data['data'])

            if current_page >= data['pager']['max_page']:
                break

            current_page += 1
        else:
            logging.error(f"Error fetching data: {response.status_code}")
            break

    logging.info(f"Fetched all match data for {competition_name} {year}")

    # For each match, fetch detailed data and insert into MongoDB
    for match in all_matches:
        match_id = match['id']
        match_stats_url = f'https://api.football-data-api.com/match?key={KEY}&match_id={match_id}'
        logging.info(f"Fetching detailed data from URL: {match_stats_url}")
        response = requests.get(match_stats_url)

        if response.status_code == 200:
            detailed_data = response.json()
            insert_document(detailed_data, collection)
        else:
            logging.error(f"Error fetching data for match {match_id}: {response.status_code}")

def insert_document(data, collection):
    """Insert or update a document in MongoDB without overwriting existing fields."""
    # Extract the nested 'data' dictionary
    match_data = data.get('data')

    if match_data and isinstance(match_data, dict) and 'id' in match_data and 'season' in match_data:
        composite_id = f"{match_data['id']}_{match_data['season']}"
        match_data["_id"] = composite_id

        try:
            # Use the $set operator to update only the fields provided by the API
            update_result = collection.update_one(
                {"_id": composite_id},
                {"$set": match_data},
                upsert=True
            )
            if update_result.upserted_id:
                logging.info(f"Inserted new match document with id {match_data['id']} for season {match_data['season']}")
            elif update_result.modified_count > 0:
                logging.info(f"Updated match document with id {match_data['id']} for season {match_data['season']}")
            else:
                logging.info(f"No changes made to match document with id {match_data['id']} for season {match_data['season']}")
        except Exception as e:
            logging.error(f"Error updating/inserting document for id {match_data['id']}: {e}")
    else:
        match_id = match_data.get('id', 'unknown') if match_data else 'unknown'
        logging.error(f"Document for match_id {match_id} is missing 'id' or 'season' or does not contain a valid 'data' object, skipping...")

def process_csv_files(csv_files):
    """Process each CSV file to fetch and insert data for each competition ID."""
    for csv_file in csv_files:
        competition_name = os.path.basename(os.path.dirname(csv_file))  # Get competition name from the file path
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                year = row['season']
                competition_id = row['competition_id']
                logging.info(f"Processing {competition_name} for year {year} with competition ID {competition_id}")
                fetch_and_insert_data(competition_name, year, competition_id)

if __name__ == "__main__":
    csv_files = [
        '/root/barnard/tmp/temp2.csv',
        # Add more CSV file paths here
    ]
    process_csv_files(csv_files)
