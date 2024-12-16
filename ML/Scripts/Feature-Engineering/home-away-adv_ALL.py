from pymongo import MongoClient, ASCENDING
from collections import deque
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
    config_path = os.path.join('/root/barnard/ML/Configs', f'{league_name}.yaml')  # Adjust the path as neededW
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file for league '{league_name}' not found at {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def calculate_home_away_advantage(league_name):
    try:
        # Load the YAML configuration for the specified league
        config = load_config(league_name)
        # Extract competition_ids from the config
        competition_ids_dict = config.get('competition_ids', {})
        if not competition_ids_dict:
            print(f"No competition_ids found in the configuration for league '{league_name}'.")
            return

        # Flatten the competition_ids_dict to get a list of competition_ids
        competition_ids = list(competition_ids_dict.values())
        print(f"Competition IDs for league '{league_name}': {competition_ids}")

        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        matches_collection = db[MATCHES_COLLECTION]
        print("Connected successfully to MongoDB.")

        # (Optional) Create a unique index on 'id' to prevent duplicates if updating existing documents
        try:
            matches_collection.create_index([("id", ASCENDING)], unique=True)
            print(f"Ensured unique index on 'id' in '{MATCHES_COLLECTION}' collection.")
        except Exception as e:
            print(f"Index creation failed or already exists: {e}")

        # Step 1: Fetch all matches sorted by date_unix ascending, filtered by competition_ids
        query = {'competition_id': {'$in': competition_ids}}
        all_matches_cursor = matches_collection.find(query).sort("date_unix", ASCENDING)

        # Initialize dictionaries to keep track of the last 50 home and away wins for each team
        team_home_wins = {}  # {team_id: deque([1, 0, 1, ...], maxlen=50)}
        team_away_wins = {}  # {team_id: deque([1, 0, 1, ...], maxlen=50)}

        # Iterate through all matches in chronological order
        for match in all_matches_cursor:
            match_date_unix = match.get('date_unix')
            competition_id = match.get('competition_id')
            home_id = match.get('homeID')
            away_id = match.get('awayID')
            match_id = match.get('id')  # Assuming 'id' is the unique identifier for matches
            winning_team = match.get('winningTeam')

            if home_id is None or away_id is None or match_id is None:
                print(f"Match_id {match_id} is missing team IDs or match ID. Skipping.")
                continue

            # Initialize deques for teams if they don't exist
            if home_id not in team_home_wins:
                team_home_wins[home_id] = deque(maxlen=50)
            if home_id not in team_away_wins:
                team_away_wins[home_id] = deque(maxlen=50)
            if away_id not in team_home_wins:
                team_home_wins[away_id] = deque(maxlen=50)
            if away_id not in team_away_wins:
                team_away_wins[away_id] = deque(maxlen=50)

            # Calculate Home Team Home Advantage
            home_home_history = team_home_wins[home_id]
            home_home_wins = sum(home_home_history)
            home_home_matches = len(home_home_history)
            home_home_percentage = (home_home_wins / home_home_matches) if home_home_matches > 0 else 0.0

            # Calculate Home Team Away Advantage
            home_away_history = team_away_wins[home_id]
            home_away_wins = sum(home_away_history)
            home_away_matches = len(home_away_history)
            home_away_percentage = (home_away_wins / home_away_matches) if home_away_matches > 0 else 0.0

            # Calculate Home Advantage for Home Team
            home_advantage_home_team = home_home_percentage - home_away_percentage

            # Calculate Away Team Home Advantage
            away_home_history = team_home_wins[away_id]
            away_home_wins = sum(away_home_history)
            away_home_matches = len(away_home_history)
            away_home_percentage = (away_home_wins / away_home_matches) if away_home_matches > 0 else 0.0

            # Calculate Away Team Away Advantage
            away_away_history = team_away_wins[away_id]
            away_away_wins = sum(away_away_history)
            away_away_matches = len(away_away_history)
            away_away_percentage = (away_away_wins / away_away_matches) if away_away_matches > 0 else 0.0

            # Calculate Home Advantage for Away Team
            home_advantage_away_team = away_home_percentage - away_away_percentage

            # Calculate Away Team Advantage for Home Team
            # Defined as Away Win Percentage - Home Win Percentage
            away_team_advantage_home_team = home_away_percentage - home_home_percentage

            # Calculate Away Team Advantage for Away Team
            # Defined as Home Win Percentage - Away Win Percentage
            away_team_advantage_away_team = away_away_percentage - away_home_percentage

            # Prepare the update document with both Home and Away Team Advantages
            update_fields = {
                'home_advantage_home_team': home_advantage_home_team,
                'home_advantage_away_team': home_advantage_away_team,
                'away_team_advantage_home_team': away_team_advantage_home_team,
                'away_team_advantage_away_team': away_team_advantage_away_team
            }

            # Update the corresponding match document in the 'matches' collection
            try:
                result = matches_collection.update_one(
                    {'id': match_id},           # Filter to find the correct match
                    {'$set': update_fields},    # Fields to update
                    upsert=False                 # Do not insert if not found
                )

                if result.matched_count > 0:
                    print(f"Updated match_id {match_id} with home and away advantages.")
                else:
                    print(f"No document found for match_id {match_id}. Skipping update.")
            except Exception as e:
                print(f"Failed to update match_id {match_id}: {e}")

            # Update the match histories after processing the current match
            # Determine the outcome for home and away teams based on 'winningTeam'
            if winning_team == home_id:
                # Home team won
                team_home_wins[home_id].append(1)  # Home team won at home
                team_away_wins[away_id].append(0)  # Away team lost away
            elif winning_team == away_id:
                # Away team won
                team_home_wins[home_id].append(0)  # Home team lost at home
                team_away_wins[away_id].append(1)  # Away team won away
            elif winning_team == -1:
                # Draw
                team_home_wins[home_id].append(0)  # No win for home team
                team_away_wins[away_id].append(0)  # No win for away team
            else:
                # Unknown outcome
                print(f"Match_id {match_id} has an invalid 'winningTeam' value: {winning_team}. Skipping history update.")
                continue

        print("\nHome and Away advantage calculation and update completed.")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

    finally:
        # Close the MongoDB connection
        client.close()
        print("Disconnected from MongoDB.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Calculate Home and Away Advantages for a League')
    parser.add_argument('league', type=str, help='Name of the league (e.g., premier_league)')
    args = parser.parse_args()
    league_name = args.league

    calculate_home_away_advantage(league_name)
