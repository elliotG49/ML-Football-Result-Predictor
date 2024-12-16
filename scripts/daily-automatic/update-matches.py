import os
import csv
import time
import json
import requests
import logging
from pymongo import MongoClient, errors
from api import KEY
import sys
import traceback
from pushover import PKEY, USER_KEY  # Import Pushover API keys

# Set up logging
logging.basicConfig(
    filename='/root/barnard/logs/match_data_update.log',  # Log file path
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
    collection = db.matches  # Replace with your collection name
    logging.info("Connected to MongoDB successfully.")
except errors.ConnectionFailure as e:
    logging.error(f"Could not connect to MongoDB: {e}")
    sys.exit(2)

def fetch_and_import_match_data(match_id, match_stats_url_base, collection, errors):
    """Fetch and import detailed match data directly into MongoDB."""
    match_stats_url = f'{match_stats_url_base}&match_id={match_id}'
    logging.info(f"Fetching detailed data from URL: {match_stats_url}")
    
    try:
        # Fetch the data from the API
        response = requests.get(match_stats_url)
        # Rate limiting to 1 request per second

        if response.status_code == 200:
            try:
                data = response.json()
                logging.info(f"Successfully fetched data for match ID {match_id}")
                
                # Insert into MongoDB
                insert_document(data, collection, errors)
            except json.JSONDecodeError as e:
                error_message = f"Error decoding JSON for match ID {match_id}: {e}"
                logging.error(error_message)
                logging.error(traceback.format_exc())
                errors.append(error_message)
        else:
            error_message = f"Error fetching data for match ID {match_id}: HTTP {response.status_code}"
            logging.error(error_message)
            errors.append(error_message)
    except requests.exceptions.RequestException as e:
        error_message = f"Request failed for match ID {match_id}: {e}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        errors.append(error_message)

def insert_document(data, collection, errors):
    """Insert or update a document in MongoDB."""
    try:
        # Extract the nested 'data' dictionary
        match_data = data.get('data')

        if match_data and isinstance(match_data, dict) and 'id' in match_data and 'season' in match_data:
            composite_id = f"{match_data['id']}_{match_data['season']}"
            match_data["_id"] = composite_id

            try:
                collection.replace_one(
                    {"_id": composite_id},
                    match_data,
                    upsert=True
                )
                logging.info(f"Inserted/Updated match document with id {match_data['id']} for season {match_data['season']}")
            except errors.DuplicateKeyError:
                warning_message = f"Duplicate document found for id {match_data['id']} and season {match_data['season']}, skipping..."
                logging.warning(warning_message)
                errors.append(warning_message)
            except errors.PyMongoError as e:
                error_message = f"MongoDB error while inserting/updating document for match ID {match_data['id']}: {e}"
                logging.error(error_message)
                logging.error(traceback.format_exc())
                errors.append(error_message)
        else:
            error_message = f"Document is missing 'id' or 'season' or does not contain a valid 'data' object, skipping..."
            logging.error(error_message)
            errors.append(error_message)
    except Exception as e:
        error_message = f"Unexpected error during document insertion: {e}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        errors.append(error_message)

def process_csv_file(csv_file, errors):
    """Process a CSV file to fetch and import data for each match ID."""
    try:
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            logging.info(f"Processing CSV file: {csv_file}")
            for row in reader:
                match_id = row.get('match_id')
                
                if match_id:
                    logging.info(f"Processing match ID {match_id}")
                    fetch_and_import_match_data(match_id, match_stats_url_base, collection, errors)
                else:
                    warning_message = f"Skipping row due to missing 'match_id': {row}"
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
            error_message = "Usage: python3 update_matches.py <path_to_csv>"
            logging.error(error_message)
            errors.append(error_message)
            sys.exit(1)

        csv_file = sys.argv[1]
        match_stats_url_base = f'https://api.football-data-api.com/match?key={KEY}'

        process_csv_file(csv_file, errors)
        if errors:
            # Send Pushover notification with errors
            message = "The 'update_matches.py' script encountered errors:\n" + "\n".join(errors)
            send_pushover_notification(message, title="Script Failure")
        else:
            logging.info("All match data has been processed and imported successfully.")
            send_pushover_notification("The Update Matches script ran successfully.", title="Script Success")
    except Exception as e:
        error_message = f"An unexpected error occurred in the main execution: {e}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        errors.append(error_message)
        # Send Pushover notification with errors
        message = "The 'update_matches.py' script encountered errors:\n" + "\n".join(errors)
        send_pushover_notification(message, title="Script Failure")
        sys.exit(2)
    finally:
        client.close()
        logging.info("MongoDB connection closed.")
