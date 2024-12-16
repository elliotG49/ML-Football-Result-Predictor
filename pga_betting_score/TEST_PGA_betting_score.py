import argparse
import yaml
import os
from pymongo import MongoClient, errors
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt  # Import matplotlib for plotting

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

def get_matches(db, competition_id, season):
    matches_collection = db['matches']
    try:
        # Query matches with the specified criteria
        matches_cursor = matches_collection.find({
            'competition_id': competition_id,
            'status': 'complete',
            'Predictions.Match_Result': {'$exists': True}
        })

        matches = list(matches_cursor)

        if not matches:
            print(f"No matches found for competition_id {competition_id} in season {season}.")
            return []
        else:
            print(f"Retrieved {len(matches)} matches for competition_id {competition_id}.")
        return matches
    except errors.PyMongoError as e:
        print(f"Error retrieving matches: {e}")
        return []

def process_matches(matches, league_threshold_accuracy, team_accuracies, league_name):
    processed_matches = []
    for match in matches:
        home_id = match.get('homeID')
        away_id = match.get('awayID')
        home_name = match.get('home_name', 'Unknown')
        away_name = match.get('away_name', 'Unknown')
        winning_team = match.get('winningTeam')
        predictions = match.get('Predictions', {}).get('Match_Result', {})
        final_predicted_class = predictions.get('Final_Predicted_Class')
        game_week = match.get('game_week')
        date_unix = match.get('date_unix')

        # Get predicted probabilities
        prob_draw = predictions.get('Prob_Draw', 0)
        prob_away_win = predictions.get('Prob_Away_Win', 0)
        prob_home_win = predictions.get('Prob_Home_Win', 0)
        
        try:
            date_time = datetime.fromtimestamp(date_unix)
        except (TypeError, OSError) as e:
            print(f"Invalid date_unix for match {match.get('_id')}: {e}")
            continue

        # Convert final predicted class to integer
        if final_predicted_class == 'Home Win':
            predicted_team = home_id
            mcs = prob_home_win  # Match Confidence Score
        elif final_predicted_class == 'Away Win':
            predicted_team = away_id
            mcs = prob_away_win
        elif final_predicted_class == 'Draw':
            predicted_team = -1
            mcs = prob_draw
        else:
            print(f"Invalid Final_Predicted_Class for match {match.get('_id')}: {final_predicted_class}")
            continue  # Skip if the prediction class is invalid

        # Extract odds_comparison data
        odds_comparison = match.get('odds_comparison', {})  # Store the entire odds_comparison

        # Build the processed match data with only relevant team accuracies
        processed_match = {
            'matchID': match.get('_id'),
            'homeID': home_id,
            'home_name': home_name,
            'awayID': away_id,
            'away_name': away_name,
            'winningTeam': winning_team,
            'predictedTeam': predicted_team,
            'MCS': mcs,
            'game_week': game_week,
            'date': date_time,
            'probabilities': {
                'Prob_Draw': prob_draw,
                'Prob_Away_Win': prob_away_win,
                'Prob_Home_Win': prob_home_win
            },
            'odds_comparison': odds_comparison,  # Add odds_comparison to match data
            'league_threshold_accuracy': league_threshold_accuracy,
            'team_accuracies': {
                home_id: team_accuracies.get(home_id, {}).get('threshold_accuracy'),
                away_id: team_accuracies.get(away_id, {}).get('threshold_accuracy')
            },
            'league_name': league_name
        }

        # Check for missing accuracies and handle accordingly
        if processed_match['team_accuracies'][home_id] is None:
            print(f"Missing threshold_accuracy for home team ID {home_id} in match {home_id} vs {away_id}.")
            # Optionally assign a default value or skip
            processed_match['team_accuracies'][home_id] = 0.0  # Example default

        if processed_match['team_accuracies'][away_id] is None:
            print(f"Missing threshold_accuracy for away team ID {away_id} in match {home_id} vs {away_id}.")
            # Optionally assign a default value or skip
            processed_match['team_accuracies'][away_id] = 0.0  # Example default

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

def compute_stake(betting_score):
    return 5  # Assign a flat stake of £5 to all matches

def compute_betting_score(row, MCS_CW, LTAS_CW, TTAS_CW):
    # Normalize weights to sum to 1
    total_weight = MCS_CW + LTAS_CW + TTAS_CW
    if total_weight == 0:
        return None  # Cannot compute betting score with zero total weight

    normalized_MCS_CW = MCS_CW / total_weight
    normalized_LTAS_CW = LTAS_CW / total_weight
    normalized_TTAS_CW = TTAS_CW / total_weight

    home_id = row['homeID']
    away_id = row['awayID']
    predicted_team = row['predictedTeam']
    MCS = row['MCS']
    league_threshold_accuracy = row['league_threshold_accuracy']
    team_accuracies = row['team_accuracies']

    # Get team threshold accuracy scores
    home_accuracy = team_accuracies.get(home_id)
    away_accuracy = team_accuracies.get(away_id)

    # Ensure we have both team accuracies
    if home_accuracy is None or away_accuracy is None or league_threshold_accuracy is None:
        # Cannot compute betting score without necessary accuracies
        return None

    # Compute TTAS as the average of both team accuracies
    TTAS = (home_accuracy + away_accuracy) / 2

    # Normalize TTAS
    normalized_TTAS = TTAS / 100.0

    # Use LTAS directly as it's already between 0 and 1
    normalized_LTAS = league_threshold_accuracy

    # Calculate betting score
    betting_score = (MCS * normalized_MCS_CW) + (normalized_LTAS * normalized_LTAS_CW) + (normalized_TTAS * normalized_TTAS_CW)

    return betting_score

def get_ft_result_odds(row):
    odds_comparison = row.get('odds_comparison', {})

    # Initialize odds dictionaries
    odds = {
        'best_odds_1': None, 'best_organisation_1': None,
        'avg_odds_1': None,
        'best_odds_X': None, 'best_organisation_X': None,
        'avg_odds_X': None,
        'best_odds_2': None, 'best_organisation_2': None,
        'avg_odds_2': None
    }

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

    # Extract odds for '1', 'X', '2'
    for outcome_key in ['1', 'X', '2']:
        outcome_odds = ft_result_odds.get(outcome_key, {})
        if outcome_odds:
            if isinstance(outcome_odds, dict) and outcome_odds:
                bookmaker_odds = []
                for bookmaker, odd_str in outcome_odds.items():
                    try:
                        odd = float(odd_str)
                        bookmaker_odds.append((odd, bookmaker))
                    except ValueError:
                        pass  # Ignore invalid odds
                if bookmaker_odds:
                    # Get best odds
                    best_odd, best_bookmaker = max(bookmaker_odds, key=lambda x: x[0])
                    odds[f'best_odds_{outcome_key}'] = best_odd
                    odds[f'best_organisation_{outcome_key}'] = best_bookmaker
                    # Calculate average odds
                    avg_odd = sum([odd for odd, _ in bookmaker_odds]) / len(bookmaker_odds)
                    odds[f'avg_odds_{outcome_key}'] = avg_odd
    return pd.Series(odds)

def get_predicted_odds(row):
    predicted_team = row['predictedTeam']
    # Use the odds extracted for '1', 'X', '2'

    # Best odds
    best_odds_1 = row['best_odds_1']
    best_odds_X = row['best_odds_X']
    best_odds_2 = row['best_odds_2']
    best_organisation_1 = row['best_organisation_1']
    best_organisation_X = row['best_organisation_X']
    best_organisation_2 = row['best_organisation_2']

    # Average odds
    avg_odds_1 = row['avg_odds_1']
    avg_odds_X = row['avg_odds_X']
    avg_odds_2 = row['avg_odds_2']

    # Determine which odds to use based on predicted outcome
    if predicted_team == row['homeID']:
        best_odd = best_odds_1
        best_org = best_organisation_1
        avg_odd = avg_odds_1
    elif predicted_team == row['awayID']:
        best_odd = best_odds_2
        best_org = best_organisation_2
        avg_odd = avg_odds_2
    elif predicted_team == -1:
        best_odd = best_odds_X
        best_org = best_organisation_X
        avg_odd = avg_odds_X
    else:
        best_odd = 0
        best_org = None
        avg_odd = 0

    return pd.Series({
        'best_predicted_odds': best_odd,
        'best_organisation': best_org,
        'avg_predicted_odds': avg_odd
    })

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
        # Add odds_list for the predicted class
        odds_list = result.get('odds_list', [])
        flat_result['odds_list'] = odds_list
        flattened_results.append(flat_result)

    # Create DataFrame
    df = pd.DataFrame(flattened_results)

    # Expand odds_list into separate rows for each bookmaker
    df = df.explode('odds_list')
    if 'odds_list' in df.columns and df['odds_list'].notna().any():
        df[['bookmaker', 'odd']] = df['odds_list'].apply(pd.Series)
    df.drop('odds_list', axis=1, inplace=True)

    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Results saved to {output_file}")

def process_league(league_name, season, db, MCS_CW, LTAS_CW, TTAS_CW, CONFIDENCE_THRESHOLD):
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

        # Retrieve matches
        matches = get_matches(db, competition_id, season)
        if not matches:
            print(f"No matches to process for league {league_name}")
            return []

        # Get league and team accuracies
        league_threshold_accuracy, team_accuracies = get_league_and_team_accuracies(db, competition_id)
        if league_threshold_accuracy is None or team_accuracies is None:
            print(f"Missing league or team accuracies for league {league_name}")
            return []

        # Process matches
        processed_matches = process_matches(matches, league_threshold_accuracy, team_accuracies, league_name)
        if not processed_matches:
            print(f"No processed matches for league {league_name}")
            return []

        # Filter matches with MCS >= CONFIDENCE_THRESHOLD
        filtered_processed_matches = [match for match in processed_matches if match['MCS'] >= CONFIDENCE_THRESHOLD]
        print(f"Filtered matches with MCS >= {CONFIDENCE_THRESHOLD}: {len(filtered_processed_matches)} out of {len(processed_matches)}")
        if not filtered_processed_matches:
            print(f"No matches with MCS >= {CONFIDENCE_THRESHOLD} for league {league_name}")
            return []

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
    parser = argparse.ArgumentParser(description='Compute ROI and Accuracy for matches.')
    parser.add_argument('--season', type=str, default='2024/2025', help='Season year (e.g., 2024/2025).')
    # Optional: Uncomment the following lines to allow setting confidence threshold via command-line
    parser.add_argument('--confidence_threshold', type=float, default=0.45, help='Minimum confidence threshold for predictions (e.g., 0.45).')
    args = parser.parse_args()

    season = args.season
    # confidence_threshold = args.confidence_threshold  # Uncomment if using command-line argument

    # Define your weights here
    # You can easily change these values as needed
    MCS_CW = 1  # Weight for Match Confidence Score
    TTAS_CW = 0.0  # Weight for Team Threshold Accuracy Score
    LTAS_CW = 0.0  # Weight for League Threshold Accuracy Score

    # Introduce Confidence Threshold
    CONFIDENCE_THRESHOLD = 0.51  # Set your desired confidence threshold here

    # Alternatively, use command-line argument for confidence threshold
    # CONFIDENCE_THRESHOLD = args.confidence_threshold

    print(f"Using weights - MCS_CW: {MCS_CW}, TTAS_CW: {TTAS_CW}, LTAS_CW: {LTAS_CW}")
    print(f"Applying confidence threshold: {CONFIDENCE_THRESHOLD}")

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

    # Initialize a list to store all processed matches
    all_processed_matches = []

    # Process all leagues and collect matches
    for league_name in domestic_league_names:
        print(f"\nProcessing league: {league_name}")

        # Load config
        config = load_config(league_name)
        if not config:
            print(f"Skipping league {league_name} due to missing config.")
            continue

        # Get competition_id
        competition_ids = config.get('competition_ids', {})
        competition_id = competition_ids.get(season)
        if not competition_id:
            print(f"Competition ID not found for season {season} in config file for league {league_name}.")
            continue

        # Retrieve matches
        matches = get_matches(db, competition_id, season)
        if not matches:
            print(f"No matches to process for league {league_name}")
            continue

        # Get league and team accuracies
        league_threshold_accuracy, team_accuracies = get_league_and_team_accuracies(db, competition_id)
        if league_threshold_accuracy is None or team_accuracies is None:
            print(f"Missing league or team accuracies for league {league_name}")
            continue

        # Process matches
        processed_matches = process_matches(matches, league_threshold_accuracy, team_accuracies, league_name)
        if not processed_matches:
            print(f"No processed matches for league {league_name}")
            continue

        # Add processed matches to the list
        all_processed_matches.extend(processed_matches)

    # Close MongoDB connection
    client.close()
    print("\nDisconnected from MongoDB.")

    # Convert all processed matches to a DataFrame
    df_matches = pd.DataFrame(all_processed_matches)
    initial_match_count = len(df_matches)
    print(f"\nTotal matches before applying confidence threshold: {initial_match_count}")

    if df_matches.empty:
        print("No matches to process.")
    else:
        # Apply confidence threshold filter
        df_matches = df_matches[df_matches['MCS'] >= CONFIDENCE_THRESHOLD]
        filtered_match_count = len(df_matches)
        print(f"Total matches after applying confidence threshold of {CONFIDENCE_THRESHOLD}: {filtered_match_count}")

        if df_matches.empty:
            print("No matches meet the confidence threshold.")
        else:
            # Extract odds for '1', 'X', '2' and organisations
            print("\nExtracting odds data...")
            odds_df = df_matches.apply(get_ft_result_odds, axis=1)
            df_matches = pd.concat([df_matches, odds_df], axis=1)

            # Apply get_predicted_odds to each row
            print("Extracting predicted odds...")
            predicted_odds_df = df_matches.apply(get_predicted_odds, axis=1)
            df_matches = pd.concat([df_matches, predicted_odds_df], axis=1)

            # Exclude matches with zero odds
            print("Filtering matches with valid odds...")
            df_matches = df_matches[(df_matches['avg_predicted_odds'] > 0) & (df_matches['best_predicted_odds'] > 0)]

            if df_matches.empty:
                print("No matches with valid odds.")
            else:
                # Calculate average team accuracy
                print("Calculating average team accuracy...")
                df_matches['average_team_accuracy'] = df_matches.apply(
                    lambda row: (row['team_accuracies'][row['homeID']] + row['team_accuracies'][row['awayID']]) / 2,
                    axis=1
                )

                # Calculate betting score for each match using current weights
                print("Calculating betting scores...")
                df_matches['betting_score'] = df_matches.apply(
                    lambda row: compute_betting_score(row, MCS_CW, LTAS_CW, TTAS_CW), axis=1
                )

                # Check if 'betting_score' was computed
                missing_betting_scores = df_matches['betting_score'].isnull().sum()
                if missing_betting_scores > 0:
                    print(f"Warning: {missing_betting_scores} matches have missing betting scores and will be excluded.")
                    df_matches = df_matches[df_matches['betting_score'].notnull()]

                # Filter matches with betting_score >= 0.60 (Adjusted from 0.50 to 0.60 as per your original filter)
                print("Filtering matches with betting score >= 0.60...")
                df_high_betting_score = df_matches[df_matches['betting_score'] >= 0.55]

                if df_high_betting_score.empty:
                    print("No matches with betting score >= 0.60.")
                else:
                    # Calculate stake for each match based on betting_score
                    print("Calculating stakes...")
                    df_high_betting_score['stake'] = df_high_betting_score['betting_score'].apply(compute_stake)

                    # Calculate profit or loss for each match using best odds
                    print("Calculating profit/loss using best odds...")
                    df_high_betting_score['profit_loss_best'] = df_high_betting_score.apply(
                        lambda row: (row['best_predicted_odds'] * row['stake']) - row['stake'] if row['predictedTeam'] == row['winningTeam'] else -row['stake'],
                        axis=1
                    )

                    # Calculate profit or loss for each match using average odds
                    print("Calculating profit/loss using average odds...")
                    df_high_betting_score['profit_loss_avg'] = df_high_betting_score.apply(
                        lambda row: (row['avg_predicted_odds'] * row['stake']) - row['stake'] if row['predictedTeam'] == row['winningTeam'] else -row['stake'],
                        axis=1
                    )

                    # Calculate total stake
                    total_stake = df_high_betting_score['stake'].sum()

                    # Total profit/loss using best odds
                    total_profit_loss_best = df_high_betting_score['profit_loss_best'].sum()
                    ROI_best = (total_profit_loss_best / total_stake) * 100 if total_stake != 0 else 0  # ROI in percentage

                    # Total profit/loss using average odds
                    total_profit_loss_avg = df_high_betting_score['profit_loss_avg'].sum()
                    ROI_avg = (total_profit_loss_avg / total_stake) * 100 if total_stake != 0 else 0  # ROI in percentage

                    # Calculate Accuracy
                    num_correct_predictions = df_high_betting_score[df_high_betting_score['predictedTeam'] == df_high_betting_score['winningTeam']].shape[0]
                    accuracy = (num_correct_predictions / len(df_high_betting_score)) * 100

                    print("\n--- ROI and Accuracy Calculation (Betting Score >= 0.60) ---")
                    print(f"Number of Bets Placed: {len(df_high_betting_score)}")
                    print(f"Total Amount Staked: £{total_stake:.2f}")
                    print(f"Total Profit/Loss using Best Odds: £{total_profit_loss_best:.2f}")
                    print(f"ROI using Best Odds: {ROI_best:.2f}%")
                    print(f"Total Profit/Loss using Average Odds: £{total_profit_loss_avg:.2f}")
                    print(f"ROI using Average Odds: {ROI_avg:.2f}%")
                    print(f"Accuracy: {accuracy:.2f}%")

                    # Reorder columns to include odds, organisations, and average_team_accuracy
                    desired_columns = [
                        'matchID', 'homeID', 'home_name', 'awayID', 'away_name',
                        'winningTeam', 'predictedTeam', 'MCS', 'game_week', 'date',
                        'probabilities',
                        'best_odds_1', 'best_organisation_1', 'avg_odds_1',
                        'best_odds_X', 'best_organisation_X', 'avg_odds_X',
                        'best_odds_2', 'best_organisation_2', 'avg_odds_2',
                        'best_predicted_odds', 'best_organisation', 'avg_predicted_odds',
                        'league_threshold_accuracy', 'team_accuracies', 'league_name',
                        'average_team_accuracy',  # New column added
                        'betting_score', 'stake', 'profit_loss_best', 'profit_loss_avg'
                    ]

                    # Save detailed results to CSV
                    output_file = '/root/barnard/pga_betting_score/new_betting_results_roi.csv'
                    print(f"Saving detailed betting results to {output_file}...")
                    df_high_betting_score.to_csv(output_file, index=False, columns=desired_columns)
                    print(f"Detailed betting results saved to {output_file}")

                    ### Additional Analysis and Plotting ###
                    print("\nPerforming additional analysis and plotting...")

                    # Create bins for betting scores with interval of 0.02
                    bins = np.arange(0.55, df_high_betting_score['betting_score'].max() + 0.02, 0.02)
                    df_high_betting_score['betting_score_bin'] = pd.cut(df_high_betting_score['betting_score'], bins)

                    # Group by betting score bins and calculate ROI and accuracy using average odds
                    grouped = df_high_betting_score.groupby('betting_score_bin')
                    roi_per_bin = grouped.apply(
                        lambda x: (x['profit_loss_avg'].sum() / x['stake'].sum()) * 100 if x['stake'].sum() > 0 else 0
                    )
                    accuracy_per_bin = grouped.apply(
                        lambda x: (x[x['predictedTeam'] == x['winningTeam']].shape[0] / x.shape[0]) * 100 if x.shape[0] > 0 else 0
                    )
                    avg_odds_per_bin = grouped['avg_predicted_odds'].mean()

                    # Calculate the number of games per bin
                    number_of_games_per_bin = grouped.size()

                    # Calculate the mid-point of each bin for plotting
                    bin_midpoints = [interval.left + 0.01 for interval in grouped.groups.keys()]

                    # Plot ROI vs. Betting Score (using average odds)
                    plt.figure(figsize=(10, 6))
                    plt.plot(bin_midpoints, roi_per_bin.values, marker='o', linestyle='-')
                    plt.title('ROI vs. Betting Score (Using Average Odds)')
                    plt.xlabel('Betting Score')
                    plt.ylabel('ROI (%)')
                    plt.grid(True)
                    roi_plot_path = '/root/barnard/pga_betting_score/roi_vs_betting_score.png'
                    plt.savefig(roi_plot_path)
                    plt.show()
                    print(f"ROI vs. Betting Score plot saved to {roi_plot_path}")

                    # Plot Accuracy vs. Betting Score
                    plt.figure(figsize=(10, 6))
                    plt.plot(bin_midpoints, accuracy_per_bin.values, marker='o', linestyle='-', color='orange')
                    plt.title('Accuracy vs. Betting Score')
                    plt.xlabel('Betting Score')
                    plt.ylabel('Accuracy (%)')
                    plt.grid(True)
                    accuracy_plot_path = '/root/barnard/pga_betting_score/accuracy_vs_betting_score.png'
                    plt.savefig(accuracy_plot_path)
                    plt.show()
                    print(f"Accuracy vs. Betting Score plot saved to {accuracy_plot_path}")

                    # Plot Number of Games vs. Betting Score
                    plt.figure(figsize=(10, 6))
                    plt.bar(bin_midpoints, number_of_games_per_bin.values, width=0.015, align='center', edgecolor='black')
                    plt.title('Number of Games vs. Betting Score')
                    plt.xlabel('Betting Score')
                    plt.ylabel('Number of Games')
                    plt.grid(True)
                    games_plot_path = '/root/barnard/pga_betting_score/number_of_games_vs_betting_score.png'
                    plt.savefig(games_plot_path)
                    plt.show()
                    print(f"Number of Games vs. Betting Score plot saved to {games_plot_path}")

                    # --- New Graph: Average ROI and Average Odds ---
                    print("Creating Average ROI and Average Odds graph...")

                    plt.figure(figsize=(10, 6))
                    ax1 = plt.gca()
                    ax1.plot(bin_midpoints, roi_per_bin.values, marker='o', linestyle='-', color='blue', label='Average ROI (%)')
                    ax1.set_xlabel('Betting Score')
                    ax1.set_ylabel('Average ROI (%)', color='blue')
                    ax1.tick_params(axis='y', labelcolor='blue')

                    ax2 = ax1.twinx()
                    ax2.plot(bin_midpoints, avg_odds_per_bin.values, marker='x', linestyle='--', color='green', label='Average Odds')
                    ax2.set_ylabel('Average Odds', color='green')
                    ax2.tick_params(axis='y', labelcolor='green')

                    plt.title('Average ROI and Average Odds vs. Betting Score')
                    fig = plt.gcf()
                    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

                    plt.grid(True)
                    roi_avg_odds_plot_path = '/root/barnard/pga_betting_score/roi_and_avg_odds_vs_betting_score.png'
                    plt.savefig(roi_avg_odds_plot_path)
                    plt.show()
                    print(f"Average ROI and Average Odds vs. Betting Score plot saved to {roi_avg_odds_plot_path}")

                    print("\nAll plots have been saved to the specified directory.")