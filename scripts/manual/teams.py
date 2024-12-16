import csv
import json
import requests
import os
import time
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
try:
    client = MongoClient('localhost', 27017)
    db = client.footballDB  # Replace with your database name
    collection = db.teams  # Replace with your collection name
    logging.info("Connected to MongoDB successfully.")
except errors.ConnectionFailure as e:
    logging.error(f"Could not connect to MongoDB: {e}")
    exit(1)

def fetch_and_insert_team_data(competition_name, season, season_id):
    """Fetch team data from the API and insert it into MongoDB."""
    base_url = f'https://api.football-data-api.com/league-teams?key={KEY}&season_id={season_id}&include=stats'
    current_page = 1

    while True:
        league_teams_url = f'{base_url}&page={current_page}'
        logging.info(f"Fetching data from URL: {league_teams_url}")
        response = requests.get(league_teams_url)

        if response.status_code == 200:
            data = response.json()
            teams_data = data.get('data', [])
            if not teams_data:
                logging.warning(f"No team data found for season ID {season_id} on page {current_page}")
                break

            # Insert or update each team data into MongoDB
            for team in teams_data:
                insert_document(team, collection)

            max_page = data.get('pager', {}).get('max_page', 1)
            if current_page >= max_page:
                break

            current_page += 1
        else:
            logging.error(f"Error fetching data: {response.status_code}")
            break

def insert_document(data, collection):
    """Insert or update a document in MongoDB."""
    if 'id' in data and 'season' in data and 'competition_id' in data:
        composite_id = f"{data['id']}_{data['season']}_{data['competition_id']}"
        data["_id"] = composite_id

        try:
            collection.replace_one(
                {"_id": composite_id},
                data,
                upsert=True
            )
            logging.info(f"Inserted/Updated team document with id {data['id']} for season {data['season']} in competition {data['competition_id']}")
        except errors.DuplicateKeyError:
            logging.warning(f"Duplicate document found for id {data['id']}, season {data['season']}, and competition {data['competition_id']}, skipping...")
        except errors.PyMongoError as e:
            logging.error(f"Error inserting/updating document in MongoDB: {e}")
    else:
        logging.error(f"Document is missing 'id', 'season', or 'competition_id', skipping...")

def process_csv_files(csv_files):
    """Process each CSV file to fetch and insert data for each competition ID."""
    for csv_file in csv_files:
        competition_name = os.path.basename(os.path.dirname(csv_file))  # Get competition name from the file path
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                season = row.get('season')
                season_id = row.get('competition_id')
                if not season or not season_id:
                    logging.warning(f"Missing 'season' or 'id' in row: {row}")
                    continue
                logging.info(f"Processing {competition_name} for season {season} with season ID {season_id}")
                fetch_and_insert_team_data(competition_name, season, season_id)

if __name__ == "__main__":
    try:
        csv_files = [
            '/root/barnard/data/betting/usefuls/all-leagues.csv'
        ]
        process_csv_files(csv_files)
        logging.info("Team data update process completed successfully.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        # Close the MongoDB connection when done
        client.close()
        logging.info("MongoDB connection closed.")
