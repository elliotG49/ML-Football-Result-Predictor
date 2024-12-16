import argparse
import yaml
import os
from pymongo import MongoClient, errors, ASCENDING
import csv
import pandas as pd
import numpy as np
from functools import reduce
import operator
import math
from datetime import datetime

def load_config(config_name, config_base_path='/root/barnard/ML/Configs'):
    config_path = os.path.join(config_base_path, f'{config_name}.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Config file '{config_path}' loaded successfully.")
        return config
    except Exception as e:
        print(f"Error loading config file '{config_path}': {e}")
        return None

def connect_to_mongo(mongo_uri='mongodb://localhost:27017/', db_name='footballDB'):
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        print("Connected to MongoDB.")
        return client, db
    except errors.PyMongoError as e:
        print(f"Error connecting to MongoDB: {e}")
        exit(1)

def get_next_game_week(db, competition_id):
    """
    Queries the matches collection for the earliest incomplete match
    in the given competition, and returns the game_week of that match as an integer.
    """
    matches_collection = db['matches']
    try:
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
                print(f"Next game_week for competition_id {competition_id} is: {game_week_int}")
                return game_week_int
            else:
                print(f"No game_week found in the document for competition_id {competition_id}.")
                return None
        else:
            print(f"No incomplete matches found for competition_id {competition_id}.")
            return None

    except Exception as e:
        print(f"An error occurred while fetching the next game week: {e}")
        return None

def get_matches(db, competition_id, game_week):
    matches_collection = db['matches']
    try:
        # Query matches with the specified criteria
        matches_cursor = matches_collection.find({
            'competition_id': competition_id,
            'game_week': {'$in': [game_week, 0]},  # Include both current game week and 0
            'status': 'incomplete',
            'Predictions.Match_Result': {'$exists': True}
        })
        
        matches = list(matches_cursor)

        if not matches:
            print(f"No matches found for competition_id {competition_id} in game_week {game_week} or 0.")
            return []
        else:
            print(f"Retrieved {len(matches)} matches for competition_id {competition_id} in game_week {game_week} or 0.")
        return matches
    except errors.PyMongoError as e:
        print(f"Error retrieving matches: {e}")
        return []

def process_matches(matches):
    processed_matches = []
    for match in matches:
        home_id = match.get('homeID')
        away_id = match.get('awayID')
        home_name = match.get('home_name', 'Unknown')
        away_name = match.get('away_name', 'Unknown')
        winning_team = match.get('winningTeam')  # This will be None for incomplete matches
        predictions = match.get('Predictions', {}).get('Match_Result', {})
        final_predicted_class = predictions.get('Final_Predicted_Class')
        
        # Get predicted probabilities
        prob_draw = predictions.get('Prob_Draw', 0)
        prob_away_win = predictions.get('Prob_Away_Win', 0)
        prob_home_win = predictions.get('Prob_Home_Win', 0)
        
        # Convert final predicted class to integer
        if final_predicted_class == 'Home Win':
            predicted_team = home_id
            mcs = prob_home_win  # Match Confidence Score
            our_prediction = 'Home Win'
        elif final_predicted_class == 'Away Win':
            predicted_team = away_id
            mcs = prob_away_win
            our_prediction = 'Away Win'
        elif final_predicted_class == 'Draw':
            predicted_team = -1
            mcs = prob_draw
            our_prediction = 'Draw'
        else:
            continue  # Skip if the prediction class is invalid
        
        # Extract odds_comparison data
        odds_comparison = match.get('odds_comparison', {})  # Get new odds
        
        # Get match date
        date_unix = match.get('date_unix')
        if not date_unix:
            print(f"Missing date_unix for match {match.get('_id')}")
            continue
        
        processed_match = {
            'matchID': match.get('_id'),
            'homeID': home_id,
            'home_name': home_name,
            'awayID': away_id,
            'away_name': away_name,
            'winningTeam': winning_team,
            'predictedTeam': predicted_team,
            'MCS': mcs,
            'our_prediction': our_prediction,  # Add our_prediction for labeling
            'probabilities': {
                'Prob_Draw': prob_draw,
                'Prob_Away_Win': prob_away_win,
                'Prob_Home_Win': prob_home_win
            },
            'odds_comparison': odds_comparison,  # Add odds_comparison to match data
            'date_unix': date_unix  # Include match date
        }
        processed_matches.append(processed_match)
    
    return processed_matches

def get_league_and_team_accuracies(db, competition_id):
    leagues_collection = db['leagues']
    try:
        league_doc = leagues_collection.find_one({'competition_id': competition_id})
        if not league_doc:
            print(f"No league document found for competition_id {competition_id}.")
            return None, None

        league_threshold_accuracy = league_doc.get('threshold_accuracy')
        team_accuracies = {}
        for team_name, stats in league_doc.get('team_accuracy', {}).items():
            team_id = stats.get('teamID')
            threshold_accuracy = stats.get('Threshold Accuracy')
            team_accuracies[team_id] = {
                'team_name': team_name,
                'threshold_accuracy': threshold_accuracy
            }

        return league_threshold_accuracy, team_accuracies
    except errors.PyMongoError as e:
        print(f"Error retrieving league accuracies: {e}")
        return None, None

def compute_betting_score(processed_matches, MCS_CW, LTAS_CW, TTAS_CW):
    # Normalize weights to sum to 1
    total_weight = MCS_CW + LTAS_CW + TTAS_CW
    if total_weight == 0:
        print("Total weight is zero. Cannot compute betting score.")
        return []

    normalized_MCS_CW = MCS_CW / total_weight
    normalized_LTAS_CW = LTAS_CW / total_weight
    normalized_TTAS_CW = TTAS_CW / total_weight

    print(f"Normalized Weights - MCS_CW: {normalized_MCS_CW}, LTAS_CW: {normalized_LTAS_CW}, TTAS_CW: {normalized_TTAS_CW}")

    results = []
    for match in processed_matches:
        home_id = match['homeID']
        away_id = match['awayID']
        home_name = match['home_name']
        away_name = match['away_name']
        predicted_team = match['predictedTeam']
        winning_team = match['winningTeam']  # Will be None for incomplete matches
        MCS = match['MCS']
        league_threshold_accuracy = match['league_threshold_accuracy']
        team_accuracies = match['team_accuracies']
        league_name = match['league_name']

        # Get team threshold accuracy scores
        home_accuracy = team_accuracies.get(home_id, {}).get('threshold_accuracy')
        away_accuracy = team_accuracies.get(away_id, {}).get('threshold_accuracy')

        # Ensure we have both team accuracies
        if home_accuracy is None or away_accuracy is None or league_threshold_accuracy is None:
            print(f"Missing accuracy data for match {match['matchID']}")
            continue

        # Compute TTAS as the average of both team accuracies
        TTAS = (home_accuracy + away_accuracy) / 2

        # Normalize TTAS
        normalized_TTAS = TTAS / 100.0

        # Use LTAS directly as it's already between 0 and 1
        normalized_LTAS = league_threshold_accuracy

        # Calculate betting score
        betting_score = (MCS * normalized_MCS_CW) + (normalized_LTAS * normalized_LTAS_CW) + (normalized_TTAS * normalized_TTAS_CW)

        print(f"Computed Betting Score for match {match['matchID']}: {betting_score}")

        # Collect match data
        match_data = {
            'league_name': league_name,
            'matchID': match['matchID'],
            'homeID': home_id,
            'home_name': home_name,
            'awayID': away_id,
            'away_name': away_name,
            'predictedTeam': predicted_team,
            'our_prediction': match['our_prediction'],  # Include our_prediction
            'winningTeam': winning_team,
            'betting_score': betting_score,
            'MCS': MCS,
            'LTAS': league_threshold_accuracy,
            'normalized_LTAS': normalized_LTAS,
            'TTAS': TTAS,
            'normalized_TTAS': normalized_TTAS,
            'MCS_CW': normalized_MCS_CW,
            'LTAS_CW': normalized_LTAS_CW,
            'TTAS_CW': normalized_TTAS_CW,
            'probabilities': match['probabilities'],
            'odds_comparison': match['odds_comparison'],  # Include odds_comparison
            'date_unix': match['date_unix']  # Include date_unix
        }
        results.append(match_data)
    return results

def get_best_odds(odds_comparison, predicted_team, home_id, away_id):
    """
    Extracts the best (highest) decimal odds and the corresponding bookmaker
    based on the predicted team.
    """
    # Initialize variables
    best_odds = -1.0
    best_bookmaker = 'Unknown'
    
    # Determine the outcome key based on predicted team
    if predicted_team == home_id:
        outcome_key = '1'
    elif predicted_team == away_id:
        outcome_key = '2'
    elif predicted_team == -1:
        outcome_key = 'X'
    else:
        return best_odds, best_bookmaker  # Invalid prediction
    
    # Handle odds_comparison being a dict or list
    ft_result_odds = {}
    if isinstance(odds_comparison, dict):
        # odds_comparison is a dict with market names as keys
        ft_result_odds = odds_comparison.get('FT Result', {})
    elif isinstance(odds_comparison, list):
        # odds_comparison is a list of market dictionaries
        for market in odds_comparison:
            if market.get('marketName') == 'FT Result':
                ft_result_odds = market.get('odds', {})
                break
    
    # Extract odds for the predicted outcome
    outcome_odds = ft_result_odds.get(outcome_key, {})
    
    for bookmaker, odd_str in outcome_odds.items():
        try:
            odd = float(odd_str)
            if odd > best_odds:
                best_odds = odd
                best_bookmaker = bookmaker
        except ValueError:
            continue  # Ignore invalid odds
    
    if best_odds == -1.0:
        # If no valid odds found, set default
        best_odds = 1.0
        best_bookmaker = 'Unknown'
    
    return best_odds, best_bookmaker

def save_results_to_csv(results, output_file='betting_scores_next_gameweek.csv'):
    if not results:
        print(f"No results to save to {output_file}.")
        return

    # Flatten nested dictionaries for CSV output
    flattened_results = []
    for result in results:
        flat_result = result.copy()
        # Flatten probabilities
        for key, value in result['probabilities'].items():
            flat_result[key] = value
        # Remove nested dictionaries
        flat_result.pop('probabilities', None)
        flat_result.pop('odds_comparison', None)  # Remove odds_comparison to keep CSV clean

        # Extract the best odds and corresponding bookmaker
        best_odds, best_bookmaker = get_best_odds(
            result['odds_comparison'],
            result['predictedTeam'],
            result['homeID'],
            result['awayID']
        )
        flat_result['best_bookmaker'] = best_bookmaker
        flat_result['best_odd'] = best_odds

        # Remove 'odds_comparison' if present
        flat_result.pop('odds_comparison', None)

        flattened_results.append(flat_result)

    # Create DataFrame
    df = pd.DataFrame(flattened_results)

    # Define the column order (optional)
    columns = [
        'league_name', 'matchID', 'homeID', 'home_name', 'awayID', 'away_name',
        'predictedTeam', 'our_prediction', 'winningTeam', 'betting_score',
        'MCS', 'LTAS', 'normalized_LTAS', 'TTAS', 'normalized_TTAS',
        'MCS_CW', 'LTAS_CW', 'TTAS_CW', 'date_unix',
        'Prob_Draw', 'Prob_Away_Win', 'Prob_Home_Win',
        'best_bookmaker', 'best_odd'
    ]

    # Ensure all columns are present
    columns = [col for col in columns if col in df.columns]

    df.to_csv(output_file, index=False, columns=columns, encoding='utf-8')
    print(f"Results saved to {output_file}")

def get_odds_for_predicted_class(row):
    odds_comparison = row.get('odds_comparison', {})
    predicted_team = row['predictedTeam']
    home_id = row['homeID']
    away_id = row['awayID']

    # Initialize odds list
    odds_list = []

    # Initialize ft_result_odds
    ft_result_odds = {}

    # Handle odds_comparison being a dict or list
    if isinstance(odds_comparison, dict):
        # odds_comparison is a dict with market names as keys
        ft_result_odds = odds_comparison.get('FT Result', {})
    elif isinstance(odds_comparison, list):
        # odds_comparison is a list of market dictionaries
        for market in odds_comparison:
            if market.get('marketName') == 'FT Result':
                ft_result_odds = market.get('odds', {})
                break
    else:
        # odds_comparison is neither dict nor list
        ft_result_odds = {}

    # Determine outcome key based on predicted team
    if predicted_team == home_id:
        outcome_key = '1'
    elif predicted_team == away_id:
        outcome_key = '2'
    elif predicted_team == -1:
        outcome_key = 'X'
    else:
        outcome_key = None

    if outcome_key and outcome_key in ft_result_odds:
        outcome_odds = ft_result_odds.get(outcome_key, {})
        if isinstance(outcome_odds, dict):
            for bookmaker, odd_str in outcome_odds.items():
                try:
                    odd = float(odd_str)
                    odds_list.append({'bookmaker': bookmaker, 'odd': odd})
                except ValueError:
                    pass  # Ignore invalid odds

    # If odds_list is empty, add a default entry
    if not odds_list:
        odds_list.append({'bookmaker': 'Unknown', 'odd': 1.0})

    return pd.Series({'odds_list': odds_list})

def save_bet_tracking_csv(results, output_base_path='/root/barnard/pga_betting_score/bet_tracking'):
    """
    Saves the betting data to CSV files formatted for bet tracking software.
    Each file will contain up to 1000 bets, separated by semicolons.
    """
    if not results:
        print("No results to save for bet tracking.")
        return
    
    # Ensure output directory exists
    os.makedirs(output_base_path, exist_ok=True)
    
    # Prepare data for CSV
    bet_rows = []
    for match in results:
        # Convert Unix timestamp to datetime
        match_datetime = datetime.utcfromtimestamp(match['date_unix'])
        date_str = match_datetime.strftime('%Y-%m-%d %H:%M')  # Format: YYYY-MM-DD HH:mm
        
        # Type: 'S' for Simple
        bet_type = 'S'
        
        # Sport: 'Football'
        sport = 'Football'
        
        # Label: '{home_name} VS {away_name} --Prediction {our prediction}'
        label = f"{match['home_name']} VS {match['away_name']} --Prediction {match['our_prediction']}"
        
        # Odds: Best odds in Decimal format
        best_odds, best_bookmaker = get_best_odds(
            match['odds_comparison'],
            match['predictedTeam'],
            match['homeID'],
            match['awayID']
        )
        
        # Stake: 5
        stake = 5
        
        # State: 'P' for Pending
        state = 'P'
        
        # Append the row
        bet_rows.append({
            'Date': date_str,
            'Type': bet_type,
            'Sport': sport,
            'Label': label,
            'Odds': best_odds,
            'Stake': stake,
            'State': state,
            'Bookmaker': best_bookmaker
        })
    
    # Calculate the number of files needed
    max_bets_per_file = 1000
    total_bets = len(bet_rows)
    num_files = math.ceil(total_bets / max_bets_per_file)
    
    for i in range(num_files):
        start_idx = i * max_bets_per_file
        end_idx = start_idx + max_bets_per_file
        subset = bet_rows[start_idx:end_idx]
        
        # Define the output file path
        if num_files == 1:
            output_file = os.path.join(output_base_path, 'bet_tracking_all_bets.csv')  # UPDATED
        else:
            output_file = os.path.join(output_base_path, f'bet_tracking_part_{i+1}.csv')
        
        # Create DataFrame
        df = pd.DataFrame(subset)
        
        # Define the column order
        columns = ['Date', 'Type', 'Sport', 'Label', 'Odds', 'Stake', 'State', 'Bookmaker']
        
        # Save to CSV with semicolon delimiter
        df.to_csv(output_file, index=False, sep=';', columns=columns, encoding='utf-8')
        print(f"Bet tracking CSV saved to {output_file} with {len(subset)} bets.")

def save_filtered_bet_tracking_csv(results, filter_condition, filename, output_base_path='/root/barnard/pga_betting_score/bet_tracking'):
    """
    Saves filtered betting data to a CSV file based on the provided filter condition.
    """
    if not results:
        print(f"No results to save for {filename}.")
        return
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Apply filter condition
    filtered_df = df[filter_condition].copy()
    
    if filtered_df.empty:
        print(f"No matches found for {filename} with the specified filters.")
        return
    
    # Extract best odds and bookmaker
    filtered_df['best_odds'], filtered_df['best_bookmaker'] = zip(*filtered_df.apply(
        lambda row: get_best_odds(row['odds_comparison'], row['predictedTeam'], row['homeID'], row['awayID']),
        axis=1
    ))
    
    # Prepare bet tracking rows
    bet_rows = []
    for _, row in filtered_df.iterrows():
        # Convert Unix timestamp to datetime
        match_datetime = datetime.utcfromtimestamp(row['date_unix'])
        date_str = match_datetime.strftime('%Y-%m-%d %H:%M')  # Format: YYYY-MM-DD HH:mm
        
        # Type: 'S' for Simple
        bet_type = 'S'
        
        # Sport: 'Football'
        sport = 'Football'
        
        # Label: '{home_name} VS {away_name} --Prediction {our prediction}'
        label = f"{row['home_name']} VS {row['away_name']} --Prediction {row['our_prediction']}"
        
        # Odds: Best odds in Decimal format
        best_odds = row['best_odds']
        best_bookmaker = row['best_bookmaker']
        
        # Stake: 5
        stake = 5
        
        # State: 'P' for Pending
        state = 'P'
        
        # Append the row
        bet_rows.append({
            'Date': date_str,
            'Type': bet_type,
            'Sport': sport,
            'Label': label,
            'Odds': best_odds,
            'Stake': stake,
            'State': state,
            'Bookmaker': best_bookmaker
        })
    
    # Calculate the number of files needed
    max_bets_per_file = 1000
    total_bets = len(bet_rows)
    num_files = math.ceil(total_bets / max_bets_per_file)
    
    for i in range(num_files):
        start_idx = i * max_bets_per_file
        end_idx = start_idx + max_bets_per_file
        subset = bet_rows[start_idx:end_idx]
        
        # Define the output file path
        if num_files == 1:
            output_file = os.path.join(output_base_path, filename)  # Use provided filename
        else:
            output_file = os.path.join(output_base_path, f"{filename.split('.csv')[0]}_part_{i+1}.csv")
        
        # Create DataFrame
        df_subset = pd.DataFrame(subset)
        
        # Define the column order
        columns = ['Date', 'Type', 'Sport', 'Label', 'Odds', 'Stake', 'State', 'Bookmaker']
        
        # Save to CSV with semicolon delimiter
        df_subset.to_csv(output_file, index=False, sep=';', columns=columns, encoding='utf-8')
        print(f"Filtered bet tracking CSV saved to {output_file} with {len(subset)} bets.")

def process_league(league_name, season, db, MCS_CW, LTAS_CW, TTAS_CW):
    results = []
    try:
        # Load config
        config = load_config(league_name)
        if not config:
            print(f"Skipping league {league_name} due to missing config.")
            return []

        # Get competition_id
        competition_ids = config.get('competition_ids', {})
        competition_id = competition_ids.get(season)
        if not competition_id:
            print(f"Competition ID not found for season {season} in config file for league {league_name}.")
            return []

        # Get next game week
        game_week = get_next_game_week(db, competition_id)
        if game_week is None:
            print(f"No next game week found for league {league_name}")
            return []

        # Retrieve matches for the next game week
        matches = get_matches(db, competition_id, game_week)
        if not matches:
            print(f"No matches to process for league {league_name} in game week {game_week}")
            return []

        # Process matches
        processed_matches = process_matches(matches)
        if not processed_matches:
            print(f"No processed matches for league {league_name}")
            return []

        # Filter matches with MCS >= 0.45
        threshold_mcs = 0.45
        filtered_processed_matches = [match for match in processed_matches if match['MCS'] >= threshold_mcs]
        if not filtered_processed_matches:
            print(f"No matches with MCS >= {threshold_mcs} for league {league_name}")
            return []

        # Get league and team accuracies
        league_threshold_accuracy, team_accuracies = get_league_and_team_accuracies(db, competition_id)
        if league_threshold_accuracy is None or team_accuracies is None:
            print(f"Missing league or team accuracies for league {league_name}")
            return []

        # Add league and team accuracies to each match data
        for match in filtered_processed_matches:
            match['league_threshold_accuracy'] = league_threshold_accuracy
            match['team_accuracies'] = team_accuracies
            match['league_name'] = league_name

        # Compute betting score using the filtered matches
        results = compute_betting_score(
            filtered_processed_matches,
            MCS_CW,
            LTAS_CW,
            TTAS_CW
        )
    except Exception as e:
        print(f"An error occurred while processing league {league_name}: {e}")
    return results

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Extract matches and compute betting scores for next game week.')
    parser.add_argument('--season', type=str, default='2024/2025', help='Season year (e.g., 2024/2025).')
    args = parser.parse_args()

    season = args.season

    # New weights as per the updated configuration
    MCS_CW = 0.7  # Weight for Match Confidence Score
    TTAS_CW = 0.3  # Weight for Team Threshold Accuracy Score
    LTAS_CW = 0.0  # Weight for League Threshold Accuracy Score

    print(f"Using weights - MCS_CW: {MCS_CW}, TTAS_CW: {TTAS_CW}, LTAS_CW: {LTAS_CW}")

    # Connect to MongoDB
    client, db = connect_to_mongo()

    # List of league names
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

    all_results = []

    for league_name in domestic_league_names:
        print(f"\nProcessing league: {league_name}")
        league_results = process_league(league_name, season, db, MCS_CW, LTAS_CW, TTAS_CW)
        if league_results:
            all_results.extend(league_results)
        else:
            print(f"No results for league: {league_name}")

    if all_results:
        # At this point, all_results contains all processed matches with odds_comparison
        # No need to explode odds_list here since we're saving only the best odds
        pass  # This line can be removed or kept as a placeholder
    else:
        print("No results to process further.")

    # Convert all_results to DataFrame for easier filtering
    df_all = pd.DataFrame(all_results)

    # Save all results to CSV with only the best odds
    save_results_to_csv(all_results, output_file='/root/barnard/pga_betting_score/betting_scores_next_gameweek.csv')

    # Save top 50 matches by betting score to a separate CSV
    if not df_all.empty:
        # Sort the results by betting score in descending order
        df_all_sorted = df_all.sort_values(by='betting_score', ascending=False)
    
        # Get the top 50 matches
        top_50_results = df_all_sorted.head(50).to_dict(orient='records')
    
        # Save the top 50 results to a separate CSV
        save_results_to_csv(top_50_results, output_file='/root/barnard/pga_betting_score/top_50_betting_scores_next_gameweek.csv')
    else:
        print("No results to save for top 50 betting scores.")
        top_50_results = []

    # Generate Bet Tracking CSVs
    if not df_all.empty:
        # 1. Save all bets as per existing functionality
        save_bet_tracking_csv(all_results, output_base_path='/root/barnard/pga_betting_score/bet_tracking')

        # 2. CSV for all games with betting scores between 0.6 and 0.78
        condition_2 = (df_all['betting_score'] >= 0.6) & (df_all['betting_score'] <= 0.78)
        save_filtered_bet_tracking_csv(
            results=all_results,
            filter_condition=condition_2,
            filename='bet_tracking_bet_score_0.6_0.78.csv',
            output_base_path='/root/barnard/pga_betting_score/bet_tracking'
        )

        # 3. CSVs based on specific TTAS and MCS ranges
        # Define the ranges as a list of dictionaries
        filter_conditions = [
            {'TTAS_min': 75, 'TTAS_max': 80, 'MCS_min': 60, 'MCS_max': 65, 'filename': 'bet_tracking_TTAS_75_80_MCS_60_65.csv'},  # a & b are same
            {'TTAS_min': 75, 'TTAS_max': 80, 'MCS_min': 70, 'MCS_max': 75, 'filename': 'bet_tracking_TTAS_75_80_MCS_70_75.csv'},
            {'TTAS_min': 80, 'TTAS_max': 85, 'MCS_min': 50, 'MCS_max': 55, 'filename': 'bet_tracking_TTAS_80_85_MCS_50_55.csv'},
            {'TTAS_min': 80, 'TTAS_max': 85, 'MCS_min': 55, 'MCS_max': 60, 'filename': 'bet_tracking_TTAS_80_85_MCS_55_60.csv'},
            {'TTAS_min': 80, 'TTAS_max': 85, 'MCS_min': 65, 'MCS_max': 70, 'filename': 'bet_tracking_TTAS_80_85_MCS_65_70.csv'},
            {'TTAS_min': 80, 'TTAS_max': 85, 'MCS_min': 70, 'MCS_max': 75, 'filename': 'bet_tracking_TTAS_80_85_MCS_70_75.csv'},
            {'TTAS_min': 85, 'TTAS_max': 90, 'MCS_min': 55, 'MCS_max': 60, 'filename': 'bet_tracking_TTAS_85_90_MCS_55_60.csv'},
            {'TTAS_min': 85, 'TTAS_max': 90, 'MCS_min': 70, 'MCS_max': 75, 'filename': 'bet_tracking_TTAS_85_90_MCS_70_75.csv'},
            {'TTAS_min': 85, 'TTAS_max': 90, 'MCS_min': 75, 'MCS_max': 80, 'filename': 'bet_tracking_TTAS_85_90_MCS_75_80.csv'},
        ]

        for condition in filter_conditions:
            TTAS_min = condition['TTAS_min']
            TTAS_max = condition['TTAS_max']
            MCS_min = condition['MCS_min']
            MCS_max = condition['MCS_max']
            filename = condition['filename']
            
            # Define the filter condition
            filter_cond = (
                (df_all['TTAS'] >= TTAS_min) &
                (df_all['TTAS'] <= TTAS_max) &
                (df_all['MCS'] >= MCS_min) &
                (df_all['MCS'] <= MCS_max)
            )
            
            # Save the filtered CSV
            save_filtered_bet_tracking_csv(
                results=all_results,
                filter_condition=filter_cond,
                filename=filename,
                output_base_path='/root/barnard/pga_betting_score/bet_tracking'
            )
    else:
        print("No results to save for bet tracking.")

    # Close MongoDB connection
    client.close()
    print("Disconnected from MongoDB.")
