#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError

# === Configuration ===
MONGO_URI = 'mongodb://localhost:27017/'  # Replace with your MongoDB URI
DATABASE_NAME = 'footballDB'              # Replace with your database name
MATCHES_COLLECTION = 'matches'            # Replace with your matches collection name

# === Logging Setup ===
def setup_logging(log_file_path):
    """
    Configure the logging module.
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

# === Argument Parsing ===
def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Find the Next Upcoming Football Match')
    parser.add_argument('--league', type=str, help='Name of the league to filter matches (optional)')
    parser.add_argument('--competition_id', type=int, help='Competition ID to filter matches (optional)')
    args = parser.parse_args()
    return args

# === Main Functionality ===
def find_next_match(client, db_name, collection_name, current_time_unix, league=None, competition_id=None):
    """
    Find the next upcoming match based on the current Unix timestamp.
    
    Parameters:
        client (MongoClient): The MongoDB client.
        db_name (str): Name of the database.
        collection_name (str): Name of the collection.
        current_time_unix (int): Current Unix timestamp in seconds.
        league (str): Optional league name to filter matches.
        competition_id (int): Optional competition ID to filter matches.
    
    Returns:
        dict or None: The next match document or None if not found.
    """
    db = client[db_name]
    collection = db[collection_name]

    # Build the query
    query = {'date_unix': {'$gt': current_time_unix}}

    # Optional filters
    if league:
        query['league'] = league
    if competition_id:
        query['competition_id'] = competition_id

    try:
        # Find the next match
        next_match = collection.find(query).sort('date_unix', ASCENDING).limit(1).next()
        return next_match
    except StopIteration:
        logging.info("No upcoming matches found with the specified criteria.")
        return None
    except PyMongoError as e:
        logging.error(f"An error occurred while querying the database: {e}")
        return None

def display_match_info(match):
    """
    Display information about the match.
    
    Parameters:
        match (dict): The match document from MongoDB.
    """
    try:
        match_id = match.get('id', 'N/A')
        date_unix = match.get('date_unix', 0)
        date = datetime.fromtimestamp(date_unix, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        home_team = match.get('home_name', 'Unknown Home Team')
        away_team = match.get('away_name', 'Unknown Away Team')
        competition = match.get('competition_name', 'Unknown Competition')
        home_elo = match.get('home_elo_pre_match', 'N/A')
        away_elo = match.get('away_elo_pre_match', 'N/A')
        home_advantage = match.get('home_advantage_home_team', 'N/A')

        print("\n=== Next Upcoming Match ===")
        print(f"Match ID           : {match_id}")
        print(f"Date and Time      : {date}")
        print(f"Competition        : {competition}")
        print(f"Home Team          : {home_team} (ELO: {home_elo})")
        print(f"Away Team          : {away_team} (ELO: {away_elo})")
        print(f"Home Advantage     : {home_advantage}")
        print("===========================\n")
    except Exception as e:
        logging.error(f"Error displaying match information: {e}")

def main():
    # === Initialize Logging ===
    LOG_FILE_PATH = "/root/barnard/logs/find_next_match.log"  # Replace with your desired log path
    setup_logging(LOG_FILE_PATH)
    logging.info("Logging initialized.")

    # === Parse Command-Line Arguments ===
    args = parse_arguments()
    league = args.league
    competition_id = args.competition_id

    if league:
        logging.info(f"Filtering matches for league: {league}")
    if competition_id:
        logging.info(f"Filtering matches for competition_id: {competition_id}")

    try:
        # === Connect to MongoDB ===
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        collection = db[MATCHES_COLLECTION]
        logging.info(f"Connected to MongoDB at {MONGO_URI}, database: '{DATABASE_NAME}', collection: '{MATCHES_COLLECTION}'.")
    except PyMongoError as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        sys.exit(1)

    try:
        # === Get Current UTC Time as Unix Timestamp ===
        current_time = datetime.now(timezone.utc)
        current_time_unix = int(current_time.timestamp())
        logging.info(f"Current UTC Time: {current_time.isoformat()} (Unix Timestamp: {current_time_unix})")

        # === Find the Next Upcoming Match ===
        next_match = find_next_match(
            client=client,
            db_name=DATABASE_NAME,
            collection_name=MATCHES_COLLECTION,
            current_time_unix=current_time_unix,
            league=league,
            competition_id=competition_id
        )

        if next_match:
            logging.info(f"Next upcoming match found: Match ID {next_match.get('id', 'N/A')}")
            display_match_info(next_match)
        else:
            logging.info("No upcoming matches found based on the provided criteria.")

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
    finally:
        # === Close MongoDB Connection ===
        client.close()
        logging.info("Disconnected from MongoDB.")

if __name__ == "__main__":
    main()
