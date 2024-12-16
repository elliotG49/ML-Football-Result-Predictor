import requests
import logging
from pymongo import MongoClient, errors
from api import KEY
import time
import json
import sys
import traceback
from pushover import PKEY, USER_KEY  # Import Pushover API keys

# Set up logging
logging.basicConfig(
    filename='/root/barnard/logs/league_table_update.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

# MongoDB connection setup
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['footballDB']
    league_table_collection = db['league_table']
    logging.info("Connected to MongoDB successfully.")
except errors.ConnectionFailure as e:
    logging.error(f"Could not connect to MongoDB: {e}")
    send_pushover_notification(f"Could not connect to MongoDB: {e}", title="League Table Update Error")
    sys.exit(1)

# Path to the todays_matches.json file
todays_matches_json_path = '/root/barnard/data/betting/todays-matches/todays-matches.json'

# Function to read competition IDs and gameweeks from todays_matches.json
def get_competitions_and_gameweeks(json_file_path):
    competitions = {}
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            matches = data.get('data', [])
            for match in matches:
                competition_id = match.get('competition_id')
                game_week = match.get('game_week') or match.get('gameweek') or match.get('round')
                if game_week is None:
                    logging.warning(f"No game_week found for match {match.get('id')}, setting to 0")
                    game_week = 0  # Default value if game_week is not available
                else:
                    game_week = int(game_week)

                if competition_id in competitions:
                    if competitions[competition_id] < game_week:
                        competitions[competition_id] = game_week
                else:
                    competitions[competition_id] = game_week
        logging.info("Competition IDs and gameweeks extracted from todays_matches.json successfully.")
    except FileNotFoundError:
        logging.error(f"todays_matches.json file not found at {json_file_path}")
        send_pushover_notification(f"todays_matches.json file not found at {json_file_path}", title="League Table Update Error")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {json_file_path}: {e}")
        send_pushover_notification(f"Error decoding JSON from {json_file_path}: {e}", title="League Table Update Error")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error while reading todays_matches.json: {e}")
        logging.error(traceback.format_exc())
        send_pushover_notification(f"Unexpected error while reading todays_matches.json: {e}", title="League Table Update Error")
        sys.exit(1)
    return competitions

# Function to fetch and update league table data for a competition_id
def update_league_table(competition_id, current_gameweek):
    url = f"https://api.football-data-api.com/league-tables?key={KEY}&season_id={competition_id}"
    logging.info(f"Fetching league table data for competition_id {competition_id} from URL: {url}")
    try:
        response = requests.get(url)
        logging.info(f"Received response with status code {response.status_code} for competition_id {competition_id}")
        response.raise_for_status()
        data = response.json()

        # Extract the league_table data
        league_table = data.get('data', {}).get('league_table', [])
        if not league_table:
            logging.warning(f"No league table data found for competition_id {competition_id}")
            return

        logging.info(f"Processing {len(league_table)} entries for competition_id {competition_id} at game_week {current_gameweek}")

        # Add metadata and upsert each team's data into the MongoDB collection
        for entry in league_table:
            try:
                # Ensure entry is a dictionary
                if not isinstance(entry, dict):
                    logging.error(f"Entry is not a dictionary: {entry}")
                    continue  # Skip this entry

                entry['competition_id'] = competition_id  # Already an integer
                entry['game_week'] = current_gameweek

                # Ensure 'id' field exists and is valid
                if 'id' in entry and entry['id'] is not None:
                    entry['id'] = int(entry['id'])
                else:
                    logging.error(f"Entry missing 'id' field or 'id' is null: {entry}")
                    continue  # Skip this entry

                # Upsert the data to avoid duplicate entries
                try:
                    result = league_table_collection.update_one(
                        {
                            'competition_id': entry['competition_id'],
                            'id': entry['id'],
                            'game_week': entry['game_week']
                        },
                        {'$set': entry},
                        upsert=True
                    )
                    if result.matched_count > 0:
                        logging.info(f"Updated league table entry for id {entry['id']} in competition_id {competition_id} at game_week {current_gameweek}")
                    elif result.upserted_id is not None:
                        logging.info(f"Inserted new league table entry for id {entry['id']} in competition_id {competition_id} at game_week {current_gameweek}")
                    else:
                        logging.info(f"No changes made for id {entry['id']} in competition_id {competition_id} at game_week {current_gameweek}")
                except errors.PyMongoError as e:
                    logging.error(f"MongoDB error while upserting data for id {entry['id']} in competition_id {competition_id}: {e.details}")

            except Exception as e:
                logging.error(f"Unexpected error occurred while processing entry for competition_id {competition_id}: {e}")
                logging.error(f"Entry causing error: {entry}")
                continue  # Move on to the next entry

    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred while fetching league table for competition_id {competition_id}: {e}")
    except ValueError as e:
        logging.error(f"JSON decoding failed for competition_id {competition_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error occurred for competition_id {competition_id}: {e}")

# Main function to run the update process
def main():
    logging.info("Starting league table update process.")

    # Clean up documents with null id or game_week
    league_table_collection.delete_many({
        '$or': [
            {'id': None},
            {'game_week': None},
            {'id': {'$exists': False}},
            {'game_week': {'$exists': False}}
        ]
    })
    logging.info("Cleaned up documents with null or missing id and game_week.")

    # Get competitions and gameweeks from todays_matches.json
    competitions = get_competitions_and_gameweeks(todays_matches_json_path)
    if not competitions:
        logging.warning("No competitions found in todays_matches.json.")
        send_pushover_notification("No competitions found in todays_matches.json.", title="League Table Update Warning")
        return

    # Process each competition
    competition_ids_processed = 0
    for competition_id, current_gameweek in competitions.items():
        logging.info(f"Processing competition_id {competition_id} at game_week {current_gameweek}")
        update_league_table(competition_id, current_gameweek)
        competition_ids_processed += 1

    logging.info(f"Processed {competition_ids_processed} competition IDs from todays_matches.json.")



    # Send Pushover notification upon successful completion
    send_pushover_notification("League table update process completed successfully.", title="League Table Update Success")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main execution: {e}")
        logging.error(traceback.format_exc())
        send_pushover_notification(f"An unexpected error occurred: {e}", title="League Table Update Error")
    finally:
        # Close the MongoDB connection when done
        client.close()
        logging.info("MongoDB connection closed.")
