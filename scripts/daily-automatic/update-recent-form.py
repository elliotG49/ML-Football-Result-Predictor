import csv
import requests
import time
import logging
from pymongo import MongoClient, errors
from api import KEY
import sys
import traceback
from pushover import PKEY, USER_KEY  # Import Pushover API keys

# Set up logging
logging.basicConfig(
    filename='/root/barnard/logs/recent_form_update.log', 
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

# MongoDB setup
try:
    client = MongoClient('localhost', 27017)
    db = client.footballDB  # Replace with your database name
    collection = db.recent_form  # Replace with your collection name
    logging.info("Connected to MongoDB successfully.")
except errors.ConnectionFailure as e:
    logging.error(f"Could not connect to MongoDB: {e}")
    sys.exit(2)

# Base URL for the API call
base_url = f'https://api.football-data-api.com/lastx?key={KEY}&team_id='

# Function to insert data into MongoDB
def insert_document(data, errors):
    # Ensure the data contains 'id', 'last_x_match_num', and 'competition_id' fields
    if 'id' in data and 'last_x_match_num' in data and 'competition_id' in data:
        # Create a composite _id field using id, last_x_match_num, and competition_id
        composite_id = f"{data['id']}_{data['last_x_match_num']}_{data['competition_id']}"
        data["_id"] = composite_id

        # Insert or update the document in MongoDB
        try:
            collection.replace_one(
                {"_id": composite_id},  # Find document with this composite _id
                data,                   # Replace or insert this data
                upsert=True             # Insert if not found
            )
            logging.info(f"Inserted/Updated recent form document with id {data['id']}, last_x_match_num {data['last_x_match_num']}, and competition_id {data['competition_id']}")
        except errors.PyMongoError as e:
            error_message = f"MongoDB error for document with id {data['id']}: {e}"
            logging.error(error_message)
            errors.append(error_message)
    else:
        error_message = f"Document is missing 'id', 'last_x_match_num', or 'competition_id', skipping..."
        logging.error(error_message)
        errors.append(error_message)

def process_csv_file(csv_file, errors):
    try:
        with open(csv_file, 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            headers = csvreader.fieldnames
            required_headers = {'team_id'}
            if not required_headers.issubset(set(headers)):
                missing = required_headers - set(headers)
                error_message = f"CSV file {csv_file} is missing required headers: {missing}"
                logging.error(error_message)
                errors.append(error_message)
                sys.exit(2)

            for row in csvreader:
                try:
                    team_id = row['team_id']
                except KeyError as e:
                    error_message = f"Missing expected column in CSV: {e}"
                    logging.error(error_message)
                    errors.append(error_message)
                    continue  # Skip this row and continue processing

                team_form_url = f'{base_url}{team_id}'
                logging.info(f"Fetching data for team ID {team_id} from URL: {team_form_url}")

                try:
                    response = requests.get(team_form_url)
                    # Rate limiting to 1 request every 0.5 seconds
                    time.sleep(0.5)

                    if response.status_code == 200:
                        try:
                            data = response.json()
                        except ValueError as e:
                            error_message = f"Invalid JSON response for team ID {team_id}: {e}"
                            logging.error(error_message)
                            errors.append(error_message)
                            continue

                        if 'data' not in data or not isinstance(data['data'], list):
                            error_message = f"'data' key missing or invalid in API response for team ID {team_id}."
                            logging.error(error_message)
                            errors.append(error_message)
                            continue

                        for item in data.get('data', []):
                            match_num = item.get('last_x_match_num')
                            try:
                                match_num = int(match_num)
                            except (TypeError, ValueError):
                                warning_message = f"Invalid match number '{match_num}' for team ID {team_id}. Skipping."
                                logging.warning(warning_message)
                                errors.append(warning_message)
                                continue

                            if match_num in [5, 6, 10]:
                                insert_document(item, errors)
                            else:
                                warning_message = f"Unexpected match number {match_num} for team ID {team_id}. Skipping."
                                logging.warning(warning_message)
                                errors.append(warning_message)
                    else:
                        error_message = f"Error fetching data for team ID {team_id}: HTTP {response.status_code}"
                        logging.error(error_message)
                        errors.append(error_message)
                except requests.exceptions.RequestException as e:
                    error_message = f"Request failed for team ID {team_id}: {e}"
                    logging.error(error_message)
                    errors.append(error_message)
        if not errors:
            logging.info("All recent form data has been fetched and inserted into MongoDB.")
    except FileNotFoundError:
        error_message = f"CSV file {csv_file} not found."
        logging.error(error_message)
        errors.append(error_message)
        sys.exit(2)
    except csv.Error as e:
        error_message = f"Error reading CSV file {csv_file}: {e}"
        logging.error(error_message)
        errors.append(error_message)
        sys.exit(2)
    except Exception as e:
        error_message = f"Unexpected error while processing CSV file {csv_file}: {e}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        errors.append(error_message)
        sys.exit(2)

def main():
    errors = []  # List to collect error messages
    try:
        if len(sys.argv) != 2:
            error_message = "Usage: python3 update_recent_form.py <path_to_csv>"
            logging.error(error_message)
            errors.append(error_message)
            sys.exit(1)

        csv_file = sys.argv[1]
        process_csv_file(csv_file, errors)
        if errors:
            # Send Pushover notification with errors
            message = "The 'update_recent_form.py' script encountered errors:\n" + "\n".join(errors)
            send_pushover_notification(message, title="Script Failure")
        else:
            send_pushover_notification("The Update Recent Form script ran successfully.", title="Script Success")
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        errors.append(error_message)
        # Send Pushover notification with errors
        message = "The 'update_recent_form.py' script encountered errors:\n" + "\n".join(errors)
        send_pushover_notification(message, title="Script Failure")
        sys.exit(2)
    finally:
        client.close()
        logging.info("MongoDB connection closed.")

if __name__ == "__main__":
    main()
