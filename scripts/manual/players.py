import json
import os
import requests
import csv
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
client = MongoClient('localhost', 27017)
db = client.footballDB  # Replace with your database name
collection = db.players  # Replace with your collection name

# API limit constants
API_LIMIT_CHECK_INTERVAL = 100  # Check API limit every 100 calls
API_CALL_LIMIT = 1600  # API call limit per hour

def check_api_limit():
    api_url = f"https://api.football-data-api.com/test-call?key={KEY}"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                request_remaining = data.get("metadata", {}).get("request_remaining")
                logging.info(f"API Calls Remaining: {request_remaining}")
                if int(request_remaining) < 200:  # Threshold can be adjusted
                    logging.warning(f"Low API Calls Remaining: {request_remaining}. Pausing...")
                    time.sleep(60)  # Wait before continuing (adjust as necessary)
        else:
            logging.error(f"Failed to check API limit. HTTP Status Code: {response.status_code}")
    except Exception as e:
        logging.error(f"Error while checking API limit: {str(e)}")

def fetch_and_import_player_data(season_id):
    """Fetch player data from the API and import it directly into MongoDB."""
    base_url = f'https://api.football-data-api.com/league-players?key={KEY}&season_id={season_id}'
    player_stats_url_base = f'https://api.football-data-api.com/player-stats?key={KEY}'

    all_players = []
    current_page = 1
    api_call_count = 0

    while True:
        league_players_url = f'{base_url}&page={current_page}'
        logging.info(f"Fetching players from URL: {league_players_url}")
        response = requests.get(league_players_url)
        api_call_count += 1

        if response.status_code == 200:
            data = response.json()
            for player in data['data']:
                all_players.append(player)
                logging.info(f"Added player with ID {player['id']} to processing list.")

            if current_page >= data['pager']['max_page']:
                logging.info(f"Reached last page for season {season_id}.")
                break

            current_page += 1
        elif response.status_code == 429:
            logging.warning("Received 429 Too Many Requests error. Waiting before retrying...")
            time.sleep(60)  # Wait before retrying (adjust as needed)
            continue
        else:
            logging.error(f"Error fetching data: {response.status_code}")
            break

        if api_call_count % API_LIMIT_CHECK_INTERVAL == 0:
            check_api_limit()

    logging.info(f"Processing {len(all_players)} players for detailed stats and import.")
    
    # Fetch and import detailed player data into MongoDB
    for player in all_players:
        player_id = player.get('id')
        fetch_and_import_detailed_player_data(player_id, player_stats_url_base, season_id)


def fetch_and_import_detailed_player_data(player_id, player_stats_url_base, season_id):
    """Fetch and import detailed player data filtered by competition ID directly into MongoDB."""
    player_stats_url = f'{player_stats_url_base}&player_id={player_id}'
    logging.info(f"Fetching detailed stats from URL: {player_stats_url}")
    api_call_count = 0
    time.sleep(1)
    
    while True:
        response = requests.get(player_stats_url)
        api_call_count += 1

        if response.status_code == 200:
            try:
                json_data = response.json()
                all_data = json_data.get('data', [])
                
                # Ensure season_id in API data is compared as an integer
                filtered_data = [player for player in all_data if player.get('competition_id') == int(season_id)]
                
                if filtered_data:
                    logging.info(f"Inserting data for player ID {player_id}")
                    insert_document(filtered_data[0], collection)
                else:
                    logging.warning(f"No data found for player ID {player_id} in season {season_id}.")
                break  # Break the loop once data is successfully processed

            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON for player ID {player_id}: {e}")
                break
        elif response.status_code == 429:
            logging.warning("Received 429 Too Many Requests error. Waiting before retrying...")
            time.sleep(60)  # Wait before retrying (adjust as needed)
        else:
            logging.error(f"Error fetching data for player ID {player_id}: {response.status_code}")
            break

        if api_call_count % API_LIMIT_CHECK_INTERVAL == 0:
            check_api_limit()

def insert_document(data, collection):
    """Insert or update a document in MongoDB."""
    if 'id' in data and 'season' in data and 'competition_id' in data:
        composite_id = f"{data['id']}_{data['season']}_{data['competition_id']}"
        data["_id"] = composite_id

        try:
            result = collection.replace_one(
                {"_id": composite_id},
                data,
                upsert=True
            )
            if result.upserted_id:
                logging.info(f"Inserted document with id {data['id']} for season {data['season']} in competition {data['competition_id']}")
            else:
                logging.info(f"Updated document with id {data['id']} for season {data['season']} in competition {data['competition_id']}")
        except errors.DuplicateKeyError:
            logging.warning(f"Duplicate document found for id {data['id']}, season {data['season']}, and competition {data['competition_id']}, skipping...")
    else:
        logging.error(f"Document is missing 'id', 'season', or 'competition_id', skipping...")

def process_csv_files(csv_files):
    """Process each CSV file to fetch and save data for each competition ID."""
    for csv_file in csv_files:
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                season_id = row['id']
                logging.info(f"Processing season_id {season_id}")
                fetch_and_import_player_data(season_id)

if __name__ == "__main__":
    csv_files = [
        '/root/barnard/data/betting/usefuls/empty-collections.csv'
    ]
    process_csv_files(csv_files)
