import subprocess
import argparse
from pymongo import MongoClient, ASCENDING
import yaml
import os
from datetime import datetime
import json
import pandas as pd

# List of domestic league names
domestic_league_names = [
    'albania_superliga',
    'czech_republic_first_league',
    'france_ligue_2',
    'latvia_virsliga',
    'romania_liga_i',
    'turkey_super_lig',
    'armenia_armenian_premier_league',
    'denmark_superliga',
    'england_premier_league',
    'germany_2_bundesliga',
    'moldova_moldovan_national_division',
    'scotland_premiership',
    'ukraine_ukrainian_premier_league',
    'austria_bundesliga',
    'estonia_meistriliiga',
    'germany_bundesliga',
    'netherlands_eerste_divisie',
    'serbia_superliga',
    'azerbaijan_premyer_liqasi',
    'greece_super_league',
    'netherlands_eredivisie',
    'slovakia_super_lig',
    'belgium_pro_league',
    'england_championship',
    'hungary_nb_i',
    'norway_eliteserien',
    'spain_la_liga',
    'bulgaria_first_league',
    'italy_serie_a',
    'poland_ekstraklasa',
    'spain_segunda_division',
    'england_efl_league_one',
    'finland_veikkausliiga',
    'italy_serie_b',
    'portugal_liga_nos',
    'sweden_allsvenskan',
    'croatia_prva_hnl',
    'england_efl_league_two',
    'france_ligue_1',
    'kazakhstan_kazakhstan_premier_league',
    'portugal_liga_pro',
    'switzerland_super_league',
]

def load_config(league_name):
    """
    Load the YAML configuration file for the specified league.
    """
    config_path = os.path.join('/root/barnard/ML/Configs', f'{league_name}.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file for league '{league_name}' not found at {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_current_season_id(league_name):
    """
    Retrieve the current season's competition_id and country code from the YAML configuration.
    """
    config = load_config(league_name)
    competition_ids_dict = config.get('competition_ids', {})
    league_country_code = config.get('country')
    if not competition_ids_dict:
        raise ValueError(f"No 'competition_ids' found in the configuration for league '{league_name}'.")
    if not league_country_code:
        raise ValueError(f"No 'country' found in the configuration for league '{league_name}'.")

    # Define the current season; you can modify this to be dynamic if needed
    current_season = '2024/2025'
    target_competition_id = competition_ids_dict.get(current_season)
    if not target_competition_id:
        raise ValueError(f"No competition_id found for season '{current_season}' in the configuration for league '{league_name}'.")

    # If target_competition_id is a tuple or list, extract the first element
    if isinstance(target_competition_id, (list, tuple)):
        target_competition_id = target_competition_id[0]

    print(f"Current season '{current_season}' has competition_id: {target_competition_id}")
    return target_competition_id, league_country_code

def get_next_game_week(competition_id):
    """
    Connects to MongoDB, queries the matches collection for the earliest incomplete match
    in the given competition, and returns the game_week of that match as an integer.
    """
    # Configuration
    MONGO_URI = 'mongodb://localhost:27017'  # Replace with your MongoDB URI
    DB_NAME = 'footballDB'                   # Replace with your database name
    MATCHES_COLLECTION = 'matches'           # Your matches collection name

    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        matches_collection = db[MATCHES_COLLECTION]
        print("Connected successfully to MongoDB.")

        # Ensure competition_id is an integer
        competition_id = int(competition_id)

        # Build the query
        query = {
            "competition_id": competition_id,
            "status": "incomplete"
        }
        projection = {
            "game_week": 1,
            "date_unix": 1,
            "_id": 0  # Exclude the _id field from the result
        }
        sort = [("date_unix", ASCENDING)]

        # Perform the query
        cursor = matches_collection.find(query, projection).sort(sort).limit(1)

        # Get the game_week from the result
        match = next(cursor, None)  # Get the first document from the cursor
        if match:
            game_week = match.get("game_week")
            if game_week is not None:
                game_week_int = int(game_week)
                print(f"Next game_week is: {game_week_int}")
                return game_week_int
            else:
                print("No game_week found in the document.")
                return None
        else:
            print("No incomplete matches found for the given competition_id.")
            return None

    except Exception as e:
        print(f"An error occurred while fetching the next game week: {e}")
        return None
    finally:
        # Close the MongoDB connection
        client.close()
        print("Disconnected from MongoDB.")

def data_preparation(league_name):
    """
    Prepare data by running feature engineering scripts for the given league.
    """
    scripts = [
        "/root/barnard/ML/Scripts/Feature-Engineering/match-rest_ALL.py",
        "/root/barnard/ML/Scripts/Feature-Engineering/home-away-adv_ALL.py",
        "/root/barnard/ML/Scripts/Feature-Engineering/calcaulate-elos.py",
        "/root/barnard/ML/Scripts/Dataset-Collection/Prepare_MR_Dataset.py"
    ]

    if league_name == 'ALL':
        # For scripts that need to be run per league
        for league in domestic_league_names:
            for script in scripts:
                if 'match-rest_ALL.py' in script or 'home-away-adv_ALL.py' in script:
                    try:
                        print(f"Running {script} with league '{league}'...")
                        command = ["python3", script, league]
                        result = subprocess.run(command, check=True, capture_output=True, text=True)
                        print(f"Output of {script}:\n{result.stdout}")
                    except subprocess.CalledProcessError as e:
                        print(f"Error running {script}: {e.stderr}")
                elif 'calcaulate-elos.py' in script:
                    try:
                        # Load the country code from the YAML config
                        config = load_config(league)
                        country_code = config.get('country')
                        if not country_code:
                            print(f"No country code found in config for league '{league}'.")
                            continue
                        print(f"Running {script} with country code '{country_code}'...")
                        command = ["python3", script, country_code]
                        result = subprocess.run(command, check=True, capture_output=True, text=True)
                        print(f"Output of {script}:\n{result.stdout}")
                    except Exception as e:
                        print(f"Error running {script}: {e}")
                elif 'Prepare_MR_Dataset.py' in script:
                    try:
                        print(f"Running {script} with league '{league}'...")
                        # Specify the dataset_type argument if needed
                        command = ["python3", script, league, "Match_Result"]
                        result = subprocess.run(command, check=True, capture_output=True, text=True)
                        print(f"Output of {script}:\n{result.stdout}")
                    except Exception as e:
                        print(f"Error running {script}: {e}")
    else:
        # Handle individual league
        for script in scripts:
            if 'match-rest_ALL.py' in script or 'home-away-adv_ALL.py' in script:
                try:
                    print(f"Running {script} with league '{league_name}'...")
                    command = ["python3", script, league_name]
                    result = subprocess.run(command, check=True, capture_output=True, text=True)
                    print(f"Output of {script}:\n{result.stdout}")
                except subprocess.CalledProcessError as e:
                    print(f"Error running {script}: {e.stderr}")
            elif 'calcaulate-elos.py' in script:
                try:
                    # Load the country code from the YAML config
                    config = load_config(league_name)
                    country_code = config.get('country')
                    if not country_code:
                        print(f"No country code found in config for league '{league_name}'.")
                        continue
                    print(f"Running {script} with country code '{country_code}'...")
                    command = ["python3", script, country_code]
                    result = subprocess.run(command, check=True, capture_output=True, text=True)
                    print(f"Output of {script}:\n{result.stdout}")
                except Exception as e:
                    print(f"Error running {script}: {e}")
            elif 'Prepare_MR_Dataset.py' in script:
                try:
                    print(f"Running {script} with league '{league_name}'...")
                    # Specify the dataset_type argument if needed
                    command = ["python3", script, league_name, "Match_Result"]
                    result = subprocess.run(command, check=True, capture_output=True, text=True)
                    print(f"Output of {script}:\n{result.stdout}")
                except Exception as e:
                    print(f"Error running {script}: {e}")

def update_matches(league_name):
    """
    Update matches for the specified league.
    """
    script = "/root/barnard/scripts/daily-automatic/update-matches_v2.py"
    if league_name == 'ALL':
        for league in domestic_league_names:
            try:
                print(f"Running {script} with league '{league}'...")
                command = ["python3", script, league]
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                print(f"Output of {script}:\n{result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"Error running {script}: {e.stderr}")
    else:
        try:
            print(f"Running {script} with league '{league_name}'...")
            command = ["python3", script, league_name]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Output of {script}:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e.stderr}")

def current_season_test(league_name):
    """
    Test the model's performance on the current season for the specified league.
    Runs the evaluation script which now saves metrics to MongoDB.
    """
    script = "/root/barnard/ML/Scripts/Predict/Match-Result_Predict_Current-Season.py"
    try:
        print(f"Running {script} with league '{league_name}'...")
        command = ["python3", script, league_name]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Output of {script}:\n{result.stdout}")
        # Since metrics are now saved to MongoDB, no need to load them from JSON
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e.stderr}")

def run_team_accuracy(league_name):
    """
    Run the team_accuracy.py script for the specified league.
    """
    script = "/root/barnard/ML/Scripts/Feature-Engineering/team_accuracy.py"
    try:
        print(f"Running {script} with league '{league_name}'...")
        command = ["python3", script, league_name]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Output of {script}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error running {script}: {e}")

def predict_specific_gameweek(league_name, game_week=None):
    """
    Predict the specified or next game week for the given league.
    """
    script = "/root/barnard/ML/Scripts/Predict/Auto_Match-Result_Predict.py"
    try:
        print(f"Running {script} with league '{league_name}'...")
        command = ["python3", script, league_name]
        if game_week is not None:
            command.extend(['--game_week', str(game_week)])
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Output of {script}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e.stderr}")

def game_week_type(value):
    if value.upper() == 'ALL':
        return 'ALL'
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Game week must be an integer or 'ALL', got '{value}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction pipeline.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test', action='store_true', help='Test the model on the current season.')
    group.add_argument('--predict', action='store_true', help='Predict the next game week.')

    parser.add_argument('league', type=str, help='Name of the league (e.g., england_premier_league) or ALL for all leagues.')
    parser.add_argument('--game_week', type=game_week_type, help="Specify the game week to predict (integer or 'ALL') (only for --predict).")

    args = parser.parse_args()

    league_name = args.league

    if args.predict:
        league_names = [league_name] if league_name != 'ALL' else domestic_league_names

        for league in league_names:
            update_matches(league)
            data_preparation(league)
            # Get game_week if not provided
            game_week = args.game_week
            if game_week is None:
                # No game_week provided, get next game week
                competition_id, _ = get_current_season_id(league)
                competition_id = int(competition_id)
                game_week = get_next_game_week(competition_id)
                if game_week is None:
                    print(f"No upcoming game week found for {league}. Skipping.")
                    continue
                # Predict for the next game week
                predict_specific_gameweek(league, game_week)
            elif game_week == 'ALL':
                # User requested to predict all game weeks up to the next one
                competition_id, _ = get_current_season_id(league)
                competition_id = int(competition_id)
                next_game_week = get_next_game_week(competition_id)
                if next_game_week is None:
                    print(f"No upcoming game week found for {league}. Skipping.")
                    continue
                # Loop from next_game_week down to 1
                for gw in range(next_game_week, 0, -1):
                    print(f"Predicting for league '{league}', game_week {gw}")
                    predict_specific_gameweek(league, gw)
            else:
                # User provided a specific game week
                predict_specific_gameweek(league, game_week)
    elif args.test:
        league_names = [league_name] if league_name != 'ALL' else domestic_league_names
        for league in league_names:
            update_matches(league)
            data_preparation(league)
            current_season_test(league)
            # Run the team_accuracy.py script after the test sequence
            run_team_accuracy(league)
        # Removed metrics collection and printing since metrics are now saved to MongoDB
        print(f"All metrics have been successfully saved to the 'leagues' collection in MongoDB")
    else:
        print("Invalid arguments. Use --test or --predict followed by league name or ALL.")
        exit(1)
