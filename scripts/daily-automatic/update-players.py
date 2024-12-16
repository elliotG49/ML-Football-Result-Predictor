import json
import requests
import csv
import time
import logging
from pymongo import MongoClient, errors
from api import KEY
import sys
import traceback
from pushover import PKEY, USER_KEY  # Import Pushover API keys

# Set up logging
logging.basicConfig(
    filename='/root/barnard/logs/player_data_update.log',  # Log file path
    level=logging.INFO,  # Logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)

# Function to send Pushover notifications
def send_pushover_notification(message, title='Script Notification'):
    url = 'https://api.pushover.net/1/messages.json'
    payload = {
        'token': PKEY,
        'user': USER_KEY,
        'message': message,
        'title': title
    }
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        logging.info('Pushover notification sent successfully.')
    except requests.exceptions.RequestException as e:
        logging.error(f'Failed to send Pushover notification: {e}')

# MongoDB setup
try:
    client = MongoClient('localhost', 27017)
    db = client.footballDB  # Replace with your database name
    collection = db.players  # Replace with your collection name
    logging.info("Connected to MongoDB successfully.")
except errors.ConnectionFailure as e:
    logging.error(f"Could not connect to MongoDB: {e}")
    sys.exit(2)

def fetch_and_import_detailed_player_data(player_id, competition_id, player_stats_url_base, collection, errors):
    """Fetch and import detailed player data filtered by competition ID directly into MongoDB."""
    player_stats_url = f'{player_stats_url_base}&player_id={player_id}'
    logging.info(f"Fetching detailed stats from URL: {player_stats_url}")
    
    try:
        # Fetch the data from the API
        response = requests.get(player_stats_url)
        time.sleep(0.5)  # Rate limiting to 1 request per second

        if response.status_code == 200:
            try:
                json_data = response.json()
                logging.info(f"Successfully fetched data for player ID {player_id}")
                all_data = json_data.get('data', [])
                
                # Filter data by competition ID
                filtered_data = [player for player in all_data if player.get('competition_id') == int(competition_id)]
                
                if filtered_data:
                    # Directly insert the filtered data into MongoDB
                    for player_data in filtered_data:
                        insert_document(player_data, collection, errors)
                    logging.info(f"Data for player ID {player_id} imported into MongoDB")
                else:
                    warning_message = f"No data found for player ID {player_id} in competition ID {competition_id}"
                    logging.warning(warning_message)
                    errors.append(warning_message)

            except json.JSONDecodeError as e:
                error_message = f"Error decoding JSON for player ID {player_id}: {e}"
                logging.error(error_message)
                logging.error(traceback.format_exc())
                errors.append(error_message)
        else:
            error_message = f"Error fetching data for player ID {player_id}: HTTP {response.status_code}"
            logging.error(error_message)
            errors.append(error_message)
    except requests.exceptions.RequestException as e:
        error_message = f"Request failed for player ID {player_id}: {e}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        errors.append(error_message)

def insert_document(data, collection, errors):
    """Insert or update a document in MongoDB."""
    try:
        if 'id' in data and 'season' in data and 'competition_id' in data:
            composite_id = f"{data['id']}_{data['season']}_{data['competition_id']}"
            data["_id"] = composite_id

            try:
                collection.replace_one(
                    {"_id": composite_id},
                    data,
                    upsert=True
                )
                logging.info(f"Inserted/Updated document with id {data['id']} for season {data['season']} in competition {data['competition_id']}")
            except errors.DuplicateKeyError:
                warning_message = f"Duplicate document found for id {data['id']}, season {data['season']}, and competition {data['competition_id']}, skipping..."
                logging.warning(warning_message)
                errors.append(warning_message)
            except errors.PyMongoError as e:
                error_message = f"MongoDB error while inserting/updating document for player ID {data['id']}: {e}"
                logging.error(error_message)
                logging.error(traceback.format_exc())
                errors.append(error_message)
        else:
            error_message = f"Document is missing 'id', 'season', or 'competition_id', skipping..."
            logging.error(error_message)
            errors.append(error_message)
    except Exception as e:
        error_message = f"Unexpected error during document insertion for player ID: {data.get('id', 'Unknown')}: {e}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        errors.append(error_message)

def process_csv_file(csv_file, errors):
    """Process a CSV file to fetch and import data for each player and competition ID."""
    try:
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            logging.info(f"Processing CSV file: {csv_file}")
            for row in reader:
                player_id = row.get('player_id')
                competition_id = row.get('competition_id')
                
                if player_id and competition_id:
                    logging.info(f"Processing player ID {player_id} for competition ID {competition_id}")
                    fetch_and_import_detailed_player_data(player_id, competition_id, player_stats_url_base, collection, errors)
                else:
                    warning_message = f"Skipping row due to missing required fields: {row}"
                    logging.warning(warning_message)
                    errors.append(warning_message)
    except FileNotFoundError:
        error_message = f"CSV file not found: {csv_file}"
        logging.error(error_message)
        errors.append(error_message)
        sys.exit(2)
    except csv.Error as e:
        error_message = f"Error reading CSV file {csv_file}: {e}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        errors.append(error_message)
        sys.exit(2)
    except Exception as e:
        error_message = f"Unexpected error while processing CSV file {csv_file}: {e}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        errors.append(error_message)
        sys.exit(2)

if __name__ == "__main__":
    errors = []  # List to collect error messages
    try:
        if len(sys.argv) != 2:
            error_message = "Usage: python3 update_players.py <path_to_csv>"
            logging.error(error_message)
            errors.append(error_message)
            sys.exit(1)

        csv_file = sys.argv[1]
        player_stats_url_base = f'https://api.football-data-api.com/player-stats?key={KEY}'

        process_csv_file(csv_file, errors)
        if errors:
            # Send Pushover notification with errors
            message = "The 'update_players.py' script encountered errors:\n" + "\n".join(errors)
            send_pushover_notification(message, title="Script Failure")
        else:
            logging.info("All player data has been processed and imported successfully.")
            send_pushover_notification("The Update Players script ran successfully.", title="Script Success")
    except Exception as e:
        error_message = f"An unexpected error occurred in the main execution: {e}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        errors.append(error_message)
        # Send Pushover notification with errors
        message = "The 'update_players.py' script encountered errors:\n" + "\n".join(errors)
        send_pushover_notification(message, title="Script Failure")
        sys.exit(2)
    finally:
        client.close()
        logging.info("MongoDB connection closed.")
