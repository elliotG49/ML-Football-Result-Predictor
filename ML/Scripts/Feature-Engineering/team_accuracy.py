import argparse
import yaml
from pymongo import MongoClient, errors
import datetime
import os
from collections import defaultdict

def load_config(config_name, config_base_path='/root/barnard/ML/Configs'):
    config_path = os.path.join(config_base_path, f'{config_name}.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Config file '{config_path}' loaded successfully.")
        return config
    except Exception as e:
        print(f"Error loading config file '{config_path}': {e}")
        exit(1)

def connect_to_mongo(mongo_uri='mongodb://localhost:27017/', db_name='footballDB'):
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        print("Connected to MongoDB.")
        return client, db
    except errors.PyMongoError as e:
        print(f"Error connecting to MongoDB: {e}")
        exit(1)

def get_matches(db, competition_id):
    matches_collection = db['matches']
    try:
        # First, find the minimum game_week for the competition
        min_game_week_doc = matches_collection.find_one(
            {'competition_id': competition_id},
            sort=[('game_week', 1)],
            projection={'game_week': 1}
        )
        if min_game_week_doc is None:
            print(f"No matches found for competition_id {competition_id}.")
            exit(1)
        min_game_week = min_game_week_doc.get('game_week')
        print(f"Minimum game_week for competition_id {competition_id} is {min_game_week}.")

        # Now, retrieve matches excluding the first game_week
        # Assuming game_week starts at 1, exclude game_week ==1
        matches_cursor = matches_collection.find(
            {
                'competition_id': competition_id,
                'status': 'complete',
                'game_week': {'$gt': 1}
            }
        )
        matches = list(matches_cursor)
        if not matches:
            print(f"No matches found for competition_id {competition_id} excluding the first game_week.")
            exit(1)
        print(f"Retrieved {len(matches)} matches for competition_id {competition_id}, excluding the first game_week.")
        return matches
    except errors.PyMongoError as e:
        print(f"Error retrieving matches: {e}")
        exit(1)

def calculate_team_accuracies(matches, confidence_threshold=0.45):
    team_stats = {}

    for match in matches:
        # Extract necessary data from match document
        home_team = match.get('home_name')
        away_team = match.get('away_name')
        home_team_id = match.get('homeID')
        away_team_id = match.get('awayID')
        winning_team = match.get('winningTeam')
        predictions = match.get('Predictions', {}).get('Match_Result', {})
        final_predicted_class = predictions.get('Final_Predicted_Class')
        prob_draw = predictions.get('Prob_Draw')
        prob_away_win = predictions.get('Prob_Away_Win')
        prob_home_win = predictions.get('Prob_Home_Win')

        # Skip match if predictions are missing
        if not predictions or final_predicted_class is None:
            continue

        # Determine prediction confidence based on Final_Predicted_Class
        if final_predicted_class == 'Home Win':
            predicted_confidence = prob_home_win
        elif final_predicted_class == 'Away Win':
            predicted_confidence = prob_away_win
        elif final_predicted_class == 'Draw':
            predicted_confidence = prob_draw
        else:
            continue  # Unknown predicted class

        # Actual winning team
        actual_winning_team_id = winning_team

        # For both home and away teams
        for team_name, team_id, team_role in [(home_team, home_team_id, 'home'), (away_team, away_team_id, 'away')]:
            team_key = (team_name, team_id)

            if team_key not in team_stats:
                # Initialize stats for the team, including teamID
                team_stats[team_key] = {
                    'teamID': team_id,
                    'predictions_made': 0,
                    'predictions_correct': 0,
                    'th_predictions_made': 0,
                    'th_predictions_correct': 0
                }

            team_stat = team_stats[team_key]
            team_stat['predictions_made'] += 1

            # Determine if prediction was correct from team's perspective
            if team_role == 'home':
                if final_predicted_class == 'Home Win' and actual_winning_team_id == home_team_id:
                    team_stat['predictions_correct'] +=1
                elif final_predicted_class == 'Away Win' and actual_winning_team_id != home_team_id and actual_winning_team_id != -1:
                    team_stat['predictions_correct'] +=1
                elif final_predicted_class == 'Draw' and actual_winning_team_id == -1:
                    team_stat['predictions_correct'] +=1
            elif team_role == 'away':
                if final_predicted_class == 'Away Win' and actual_winning_team_id == away_team_id:
                    team_stat['predictions_correct'] +=1
                elif final_predicted_class == 'Home Win' and actual_winning_team_id != away_team_id and actual_winning_team_id != -1:
                    team_stat['predictions_correct'] +=1
                elif final_predicted_class == 'Draw' and actual_winning_team_id == -1:
                    team_stat['predictions_correct'] +=1

            # Threshold calculations
            if predicted_confidence >= confidence_threshold:
                team_stat['th_predictions_made'] +=1

                # Check if the prediction was correct under threshold
                if team_role == 'home':
                    if final_predicted_class == 'Home Win' and actual_winning_team_id == home_team_id:
                        team_stat['th_predictions_correct'] +=1
                    elif final_predicted_class == 'Away Win' and actual_winning_team_id != home_team_id and actual_winning_team_id != -1:
                        team_stat['th_predictions_correct'] +=1
                    elif final_predicted_class == 'Draw' and actual_winning_team_id == -1:
                        team_stat['th_predictions_correct'] +=1
                elif team_role == 'away':
                    if final_predicted_class == 'Away Win' and actual_winning_team_id == away_team_id:
                        team_stat['th_predictions_correct'] +=1
                    elif final_predicted_class == 'Home Win' and actual_winning_team_id != away_team_id and actual_winning_team_id != -1:
                        team_stat['th_predictions_correct'] +=1
                    elif final_predicted_class == 'Draw' and actual_winning_team_id == -1:
                        team_stat['th_predictions_correct'] +=1

    # Calculate accuracies
    # Prepare a new dictionary to store results with team names as keys
    final_team_stats = {}
    for (team_name, team_id), stats in team_stats.items():
        predictions_made = stats['predictions_made']
        predictions_correct = stats['predictions_correct']
        th_predictions_made = stats['th_predictions_made']
        th_predictions_correct = stats['th_predictions_correct']

        base_accuracy = (predictions_correct / predictions_made) * 100 if predictions_made else 0
        threshold_accuracy = (th_predictions_correct / th_predictions_made) * 100 if th_predictions_made else None

        # Update stats with accuracies
        stats['Base Accuracy'] = round(base_accuracy, 2)
        stats['Threshold Accuracy'] = round(threshold_accuracy, 2) if threshold_accuracy is not None else None

        # Use team name as key in the final dictionary
        final_team_stats[team_name] = stats

    return final_team_stats

def save_team_accuracies_to_db(db, competition_id, team_stats):
    leagues_collection = db['leagues']
    try:
        # Prepare the update
        update_query = {'competition_id': int(competition_id)}
        update_operation = {
            '$set': {
                'team_accuracy': team_stats
            }
        }

        # Update the document
        result = leagues_collection.update_one(
            update_query,
            update_operation,
            upsert=False
        )

        if result.matched_count > 0:
            print(f"Updated team accuracies for competition_id {competition_id} in 'leagues' collection.")
        else:
            print(f"No existing document found for competition_id {competition_id} in 'leagues' collection.")
            print(f"Please ensure that a document with competition_id {competition_id} exists.")
    except errors.PyMongoError as e:
        print(f"Error saving team accuracies to MongoDB: {e}")
    finally:
        print("Team accuracies saved to MongoDB.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Calculate per-team prediction accuracies.')
    parser.add_argument('league_name', type=str, help='Name of the league config file (without .yaml extension).')
    parser.add_argument('--season', type=str, default='2024/2025', help='Season year (e.g., 2024/2025).')
    args = parser.parse_args()

    league_name = args.league_name
    season = args.season

    # Load config
    config = load_config(league_name)

    # Get competition_id
    competition_ids = config.get('competition_ids', {})
    competition_id = competition_ids.get(season)
    if not competition_id:
        print(f"Competition ID not found for season {season} in config file.")
        exit(1)

    # Connect to MongoDB
    client, db = connect_to_mongo()

    # Get matches
    matches = get_matches(db, competition_id)

    # Calculate team accuracies
    team_stats = calculate_team_accuracies(matches, confidence_threshold=0.45)

    # Save to MongoDB
    save_team_accuracies_to_db(db, competition_id, team_stats)

    # Close MongoDB connection
    client.close()
    print("Disconnected from MongoDB.")
