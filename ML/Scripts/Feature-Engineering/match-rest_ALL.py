from pymongo import MongoClient, ASCENDING
from datetime import datetime, timezone
import sys
import os
import yaml
import argparse

# Configuration
MONGO_URI = 'mongodb://localhost:27017'  # Replace with your MongoDB URI
DB_NAME = 'footballDB'                   # Replace with your database name
MATCHES_COLLECTION = 'matches'           # Your matches collection name

def load_config(league_name):
    """
    Load the YAML configuration file for the specified league.
    """
    config_path = os.path.join('/root/barnard/ML/Configs', f'{league_name}.yaml')  # Adjust the path as needed
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file for league '{league_name}' not found at {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def calculate_full_days(previous_timestamp, current_timestamp):
    """
    Calculate the number of full days between two Unix timestamps.

    Args:
        previous_timestamp (int): Unix timestamp of the previous match in seconds.
        current_timestamp (int): Unix timestamp of the current match in seconds.

    Returns:
        int: Number of full days between the two matches.
    """
    previous_date = datetime.fromtimestamp(previous_timestamp, tz=timezone.utc)
    current_date = datetime.fromtimestamp(current_timestamp, tz=timezone.utc)
    delta = current_date - previous_date
    return delta.days

def calculate_rest_days(league_name):
    try:
        # Load the YAML configuration for the specified league
        config = load_config(league_name)
        # Extract competition_ids from the config
        competition_ids_dict = config.get('competition_ids', {})
        if not competition_ids_dict:
            print(f"No competition_ids found in the configuration for league '{league_name}'.")
            return

        # Get all competition_ids
        competition_ids = list(competition_ids_dict.values())
        print(f"Competition IDs for league '{league_name}': {competition_ids}")

        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        matches_collection = db[MATCHES_COLLECTION]
        print("Connected successfully to MongoDB.")

        # Step 1: Get the list of team IDs in all target competitions
        team_ids_set = set()
        team_ids_cursor = matches_collection.find(
            {'competition_id': {'$in': competition_ids}},
            {'homeID': 1, 'awayID': 1}
        )
        for match in team_ids_cursor:
            home_id = match.get('homeID')
            away_id = match.get('awayID')
            if home_id:
                team_ids_set.add(home_id)
            if away_id:
                team_ids_set.add(away_id)
        print(f"Total Teams in competitions {competition_ids}: {len(team_ids_set)} teams found.")

        # If no teams found, exit
        if not team_ids_set:
            print(f"No teams found for competitions {competition_ids}.")
            return

        # Fetch all matches involving these teams, sorted by date_unix ascending
        all_matches_cursor = matches_collection.find({
            '$or': [
                {'homeID': {'$in': list(team_ids_set)}},
                {'awayID': {'$in': list(team_ids_set)}}
            ]
        }).sort('date_unix', ASCENDING)

        # Initialize a dictionary to keep track of the last match date for each team
        last_match_dates = {}

        # Iterate through all matches involving these teams in chronological order
        for match in all_matches_cursor:
            match_date_unix = match.get('date_unix')
            competition_id = match.get('competition_id')
            home_id = match.get('homeID')
            away_id = match.get('awayID')
            match_id = match.get('id')  # Assuming 'id' is the unique identifier for matches

            if match_date_unix is None or home_id is None or away_id is None or match_id is None:
                print(f"Match with ID {match_id} is missing required fields. Skipping.")
                continue

            # Update the last match date for both teams
            if home_id not in last_match_dates:
                last_match_dates[home_id] = []
            if away_id not in last_match_dates:
                last_match_dates[away_id] = []

            # Process home team
            home_rest_days = None
            if last_match_dates[home_id]:
                previous_home_match_date_unix = last_match_dates[home_id][-1]
                home_rest_days = calculate_full_days(previous_home_match_date_unix, match_date_unix)
            last_match_dates[home_id].append(match_date_unix)

            # Process away team
            away_rest_days = None
            if last_match_dates[away_id]:
                previous_away_match_date_unix = last_match_dates[away_id][-1]
                away_rest_days = calculate_full_days(previous_away_match_date_unix, match_date_unix)
            last_match_dates[away_id].append(match_date_unix)

            # If the match is in one of the target competitions, update the rest days
            if competition_id in competition_ids:
                # Prepare the update document with 'team_a_rest_days' and 'team_b_rest_days'
                update_fields = {
                    'team_a_rest_days': home_rest_days,
                    'team_b_rest_days': away_rest_days
                }

                # Update the corresponding match document in the 'matches' collection
                try:
                    result = matches_collection.update_one(
                        {'id': match_id},           # Filter to find the correct match
                        {'$set': update_fields},    # Fields to update
                        upsert=False                # Do not insert if not found
                    )

                    if result.matched_count > 0:
                        print(f"Updated match_id {match_id} with rest_days.")
                    else:
                        print(f"No document found for match_id {match_id}. Skipping update.")
                except Exception as e:
                    print(f"Failed to update match_id {match_id}: {e}")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

    finally:
        # Close the MongoDB connection
        client.close()
        print("\nDisconnected from MongoDB.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Calculate Rest Days for a League')
    parser.add_argument('league', type=str, help='Name of the league (e.g., premier_league)')
    args = parser.parse_args()
    league_name = args.league

    calculate_rest_days(league_name)
