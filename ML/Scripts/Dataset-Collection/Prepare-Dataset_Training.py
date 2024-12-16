import csv
from pymongo import MongoClient
import joblib  # For serialization
import os      # For file path handling
import logging  # For logging
import argparse  # For command-line arguments
import yaml      # For YAML configuration

def parse_arguments():
    """
    Parse command-line arguments for league and dataset type.
    """
    parser = argparse.ArgumentParser(description='ELO Rating Updater and Dataset Generator')
    parser.add_argument('league', type=str, help='Name of the league (e.g., premier_league, la-3liga)')
    parser.add_argument('dataset_type', type=str, choices=['Match_Result', 'BTTS_FTS', 'Clean_Sheet'],
                        help='Type of dataset to create')
    args = parser.parse_args()
    return args.league, args.dataset_type

def load_config(league):
    """
    Load the YAML configuration file for the specified league.
    """
    config_path = os.path.join('/root/barnard/ML/Configs', f'{league}.yaml')  # Path to config files
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file for league '{league}' not found at {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(log_file_path):
    """
    Configure the logging module.
    """
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Configure the logging module
    logging.basicConfig(
        level=logging.INFO,  # Set to DEBUG for more granular logs
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )

def get_elo(RA, RB, home_advantage=0):
    RA_adj = RA + home_advantage
    EA = 1 / (1 + 10 ** ((RB - RA_adj) / 500))
    EB = 1 - EA
    return EA, EB

def new_elo(RA, RB, EA, EB, K, SA, SB):
    RA_new = RA + K * (SA - EA)
    RB_new = RB + K * (SB - EB)
    return RA_new, RB_new

def determine_scores(team_goals, opponent_goals):
    if team_goals > opponent_goals:
        return 1, "win"
    elif team_goals < opponent_goals:
        return 0, "loss"
    else:
        return 0.5, "draw"

def main():
    # === Parse Command-Line Arguments ===
    league, dataset_type = parse_arguments()
    
    # === Load Configuration ===
    try:
        config = load_config(league)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)
    
    # === Extract Paths and Configurations from YAML ===
    try:
        predictions_pathways = config['predictions_pathways']
        training_dataset_pathway = config['training_dataset_pathway']
        competition_ids_dict = config['competition_ids']
        
        # Extract seasons and sort them
        seasons = sorted(competition_ids_dict.keys())
        competition_ids = [competition_ids_dict[season] for season in seasons]
        
        # Define current_year_id and first_competition_id
        current_year_id = competition_ids[-1]  # Last ID is current_year_id
        first_competition_id = competition_ids[0]  # First ID is to be skipped for CSV
    except KeyError as e:
        print(f"Missing configuration key: {e}")
        exit(1)
    
    # Validate dataset_type exists in the config
    if dataset_type not in predictions_pathways:
        print(f"Dataset type '{dataset_type}' not found in predictions_pathways.")
        exit(1)
    if dataset_type not in training_dataset_pathway:
        print(f"Dataset type '{dataset_type}' not found in training_dataset_pathway.")
        exit(1)
    
    # Rest of your code...

    
    # === Define Paths Based on Configuration ===
    training_dataset = training_dataset_pathway[dataset_type]
    prediction_dataset = predictions_pathways[dataset_type]
    
    # Define other paths (can also be moved to YAML if needed)
    LOG_FILE_PATH = "/root/barnard/logs/elo_update.log"
    elo_ratings_file = "/root/barnard/machine-learning/tmp/elo_ratings.joblib"
    
    # === Configure Logging ===
    setup_logging(LOG_FILE_PATH)
    logging.info(f"Starting ELO update script for league '{league}' and dataset type '{dataset_type}'.")
    
    # === Ensure Directories for Datasets Exist ===
    os.makedirs(os.path.dirname(training_dataset), exist_ok=True)
    os.makedirs(os.path.dirname(prediction_dataset), exist_ok=True)
    
    # === MongoDB Connection Details ===
    MONGO_URI = "mongodb://localhost:27017/"
    DATABASE_NAME = "footballDB"
    MATCHES_COLLECTION_NAME = "matches"
    
    # === Connect to MongoDB ===
    try:
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        matches_collection = db[MATCHES_COLLECTION_NAME]
        logging.info(f"Connected to MongoDB at {MONGO_URI}, database: {DATABASE_NAME}, collection: {MATCHES_COLLECTION_NAME}.")
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        raise  # Exit the script if connection fails
    
    # === Initialize ELO Ratings Dictionary ===
    if os.path.exists(elo_ratings_file):
        try:
            elo_ratings = joblib.load(elo_ratings_file)
            logging.info(f"ELO ratings loaded from {elo_ratings_file}.")
        except Exception as e:
            logging.error(f"Failed to load ELO ratings from {elo_ratings_file}: {e}")
            elo_ratings = {}
    else:
        logging.info("No existing ELO ratings found. Starting fresh.")
        elo_ratings = {}
    
    # === Initialize Team Mapping Dictionary ===
    team_mapping = {}
    
    # === ELO Parameters ===
    INITIAL_ELO = 1500
    K_FACTOR = 20
    HOME_ADVANTAGE = 100  # Home advantage value (adjusted based on typical football statistics)
    
    # === Define Acceptable Ranges for Numerical Fields to Detect Outliers ===
    GOALS_RANGE = (0, 15)
    ELO_RANGE = (1000, 3000)
    ODDS_RANGE = (0.0, 1000.0)
    REST_DAYS_RANGE = (0, 30)
    
    # === Define Thresholds for Expected Goals ===
    team_thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
    total_thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    
    # === Prepare Per-Match CSV Files ===
    try:
        with open(training_dataset, mode='w', newline='', encoding='utf-8') as training_csvfile, \
             open(prediction_dataset, mode='w', newline='', encoding='utf-8') as prediction_csvfile:

            per_match_fieldnames = [
                'match_id', 'competition_id', 'season', 'game_week', 'match_status',
                'team_id', 'team_name', 'opponent_id', 'opponent_name', 'is_home',
                'team_goal_count', 'opponent_goal_count', 'team_ELO_before', 
                'opponent_ELO_before', 'result',
                'team_h2h_win_percent', 'opponent_h2h_win_percent',
                'odds_team_win', 'odds_draw', 'odds_opponent_win',
                'winning_team',
                'team_rest_days', 'opponent_rest_days', 'team_ppg', 'opponent_ppg', 'opponent_ppg_mc', 'team_ppg_mc',
                'pre_match_home_ppg',
                'pre_match_away_ppg',
                'pre_match_home_xg',
                'pre_match_away_xg',
                'team_home_advantage',
                'opponent_home_advantage',
                'team_away_advantage',
                'opponent_away_advantage',
                'BTTS',
                'h2h_btts',
                'team_btts_l5',
                'team_btts_l10',
                'team_btts_l20',
                'opponent_btts_l5',
                'opponent_btts_l10',
                'opponent_btts_l20',
                'team_mc_btts_l5',
                'team_mc_btts_l10',
                'team_mc_btts_l20',
                'opponent_mc_btts_l5',
                'opponent_mc_btts_l10',
                'opponent_mc_btts_l20',
                'team_conceded_l5', 
                'team_conceded_l10',
                'team_conceded_l20',
                'opponent_conceded_l5',   
                'opponent_conceded_l10',  
                'opponent_conceded_l20',  
                'team_scoring_streak',    
                'opponent_scoring_streak',
                'team_conceded_streak',   
                'opponent_conceded_streak',
                'betting_h2h_betting_odds',
                'betting_h2h_betting_percentage'
            ]

            # Add binary features for team-specific expected goals thresholds
            for t in team_thresholds:
                per_match_fieldnames.append(f'team_over_{t}')
                per_match_fieldnames.append(f'opponent_over_{t}')

            # Add binary features for total goals over thresholds
            for t in total_thresholds:
                per_match_fieldnames.append(f'total_over_{t}')

            # Initialize CSV writers
            training_writer = csv.DictWriter(training_csvfile, fieldnames=per_match_fieldnames)
            prediction_writer = csv.DictWriter(prediction_csvfile, fieldnames=per_match_fieldnames)

            # Write headers to both CSV files
            training_writer.writeheader()
            prediction_writer.writeheader()

            logging.info(f"CSV files '{training_dataset}' and '{prediction_dataset}' initialized with headers.")

            # === Initialize Variables to Track Seasons ===
            prior_season_final_elo = None
            prior_season_teams = set()

            # === Iterate Through Competition IDs in Chronological Order ===
            for idx, competition_id in enumerate(competition_ids):
                logging.info(f"Processing Competition ID: {competition_id} (Season {idx + 1})")

                # Define the query for this competition
                query = {"competition_id": competition_id}

                # Count the number of matches for logging purposes
                try:
                    matches_count = matches_collection.count_documents(query)
                    logging.info(f"Retrieved {matches_count} matches for Competition ID {competition_id}.")
                except Exception as e:
                    logging.error(f"Failed to retrieve matches for Competition ID {competition_id}: {e}")
                    continue  # Skip to the next competition_id

                # Retrieve all matches for this competition_id with status 'complete', sorted by date_unix ascending
                try:
                    matches_cursor = matches_collection.find(query).sort("date_unix", 1)
                except Exception as e:
                    logging.error(f"Failed to retrieve matches for Competition ID {competition_id}: {e}")
                    continue  # Skip to the next competition_id

                # Collect all team_ids in this season
                current_season_team_ids = set()
                for match in matches_cursor:
                    home_id = match.get("homeID")
                    away_id = match.get("awayID")
                    if home_id:
                        current_season_team_ids.add(home_id)
                    if away_id:
                        current_season_team_ids.add(away_id)
                matches_cursor.rewind()  # Reset cursor after iteration

                # Identify new teams (not in prior_season_teams)
                new_teams = current_season_team_ids - prior_season_teams

                # If there are new teams, adjust their ELO ratings
                if new_teams:
                    if prior_season_final_elo:
                        # Sort prior season's final ELOs ascending and take bottom 3
                        sorted_prior_elo = sorted(prior_season_final_elo.items(), key=lambda x: x[1])
                        bottom_3 = sorted_prior_elo[:3]
                        mean_bottom_3 = sum([elo for _, elo in bottom_3]) / len(bottom_3)
                        logging.info(f"Mean ELO of bottom 3 teams from prior season: {mean_bottom_3:.2f}")
                    else:
                        # For the first season, use INITIAL_ELO as the mean
                        mean_bottom_3 = INITIAL_ELO
                        logging.info(f"No prior season data. Using INITIAL_ELO: {mean_bottom_3:.2f} for new teams.")

                    # Adjust ELO for new teams
                    for team_id in new_teams:
                        elo_ratings[team_id] = mean_bottom_3
                        logging.info(f"Set ELO for new team {team_id} to {mean_bottom_3:.2f}")

                # Reset the cursor to iterate through matches again
                try:
                    matches_cursor = matches_collection.find(query).sort("date_unix", 1)
                except Exception as e:
                    logging.error(f"Failed to reset cursor for Competition ID {competition_id}: {e}")
                    continue  # Skip to the next competition_id

                # === Process Each Match in the Current Season ===
                for match in matches_cursor:
                    match_id = match.get("id")

                    if match_id is None:
                        logging.warning(f"Match is missing 'id' field. Skipping.")
                        continue

                    # Ensure match_id is an integer
                    try:
                        match_id_int = int(match_id)
                    except (ValueError, TypeError):
                        logging.warning(f"Match ID {match_id} is not a valid integer. Skipping.")
                        continue

                    # Extract match fields
                    season = match.get("season")        # Season field
                    game_week = match.get("game_week") # Game week field
                    h2h_stats = match.get("h2h", {})
                    betting_stats = h2h_stats.get("betting_stats", {})
                    #odds_comparison = match.get("odds_comparison", {})
                    #btts_odds = odds_comparison.get("Both Teams To Score", {})
                    #btts_odds_yes = btts_odds.get("Yes", {})
                    #btts_odds_no = btts_odds.get("No", {})
                    previous_matches = h2h_stats.get("previous_matches_results", {})
                    btts = match.get("BTTS", {})
                    btts_home = btts.get("home", {})
                    btts_home_at_home = btts.get("home", {})
                    btts_away = btts.get("away", {})
                    btts_away_at_away = btts.get("away", {})
                    
                    home_id = match.get("homeID")
                    away_id = match.get("awayID")
                    home_name = match.get("home_name")  # Home team name    
                    away_name = match.get("away_name")  # Away team name
                    home_goals = match.get("homeGoalCount")
                    away_goals = match.get("awayGoalCount")
                    team_ppg = match.get("home_ppg")
                    opponent_ppg = match.get("away_ppg")
                    team_ppg_mc = match.get("pre_match_home_ppg")
                    pre_match_home_ppg = match.get("pre_match_teamA_overall_ppg")
                    pre_match_away_ppg = match.get("pre_match_teamB_overall_ppg")
                    opponent_ppg_mc = match.get("pre_match_away_ppg")
                    h2h_team_a_win_percent = previous_matches.get("team_a_win_percent")
                    h2h_team_b_win_percent = previous_matches.get("team_b_win_percent")
                    match_status = match.get("status")
                    odds_home_win = match.get("odds_ft_1")
                    odds_draw = match.get("odds_ft_x")
                    odds_away_win = match.get("odds_ft_2")
                    winning_team = match.get("winningTeam")
                    home_rest_days = match.get("team_a_rest_days")
                    away_rest_days = match.get("team_b_rest_days")
                    pre_match_home_xg = match.get("team_a_xg_prematch")
                    pre_match_away_xg = match.get("team_b_xg_prematch")
                    team_home_advantage = match.get("home_advantage_home_team")
                    opponent_away_advantage = match.get("away_team_advantage_away_team")
                    #btts_odds_yes_bet365 = btts_odds_yes.get("bet365")
                    #btts_odds_no_bet365 = btts_odds_no.get("bet365")
                    h2h_btts = btts.get("h2h")
                    betting_h2h_betting_odds = betting_stats.get("btts")
                    betting_h2h_betting_percentage = betting_stats.get("bttsPercentage")
                    
                    team_btts_l5 = btts_home.get("last_5")
                    team_btts_l10 = btts_home.get("last_10")
                    team_btts_l20 = btts_home.get("last_20")
                    team_mc_btts_l5 = btts_home_at_home.get("last_5")
                    team_mc_btts_l10 = btts_home_at_home.get("last_10")
                    team_mc_btts_l20 = btts_home_at_home.get("last_20")
                    
                    opponent_btts_l5 = btts_away.get("last_5")
                    opponent_btts_l10 = btts_away.get("last_10")
                    opponent_btts_l20 = btts_away.get("last_20")
                    opponent_mc_btts_l5 = btts_away_at_away.get("last_5")
                    opponent_mc_btts_l10 = btts_away_at_away.get("last_10")
                    opponent_mc_btts_l20 = btts_away_at_away.get("last_20")
                    
                    team_conceded_l5 = btts.get("home_conceded_last_5")
                    team_conceded_l10 = btts.get("home_conceded_last_10")
                    team_conceded_l20 = btts.get("home_conceded_last_20")
                    opponent_conceded_l5 = btts.get("away_conceded_last_5")
                    opponent_conceded_l10 = btts.get("away_conceded_last_10")
                    opponent_conceded_l20 = btts.get("away_conceded_last_20")
                    
                    team_scoring_streak = btts.get("home_scoring_streak")
                    opponent_scoring_streak = btts.get("away_scoring_streak")
                    
                    team_conceded_streak = btts.get("consecutive_conceded_home")
                    opponent_conceded_streak = btts.get("consecutive_conceded_away")
                    

                    # === Determine BTTS (Both Teams to Score) ===
                    try:
                        home_goals_int = int(home_goals)
                        away_goals_int = int(away_goals)
                        btts = 1 if home_goals_int > 0 and away_goals_int > 0 else 0
                    except (ValueError, TypeError):
                        btts = 0
                        logging.warning(f"Invalid goal counts for Match ID {match_id}. Setting BTTS to 0.")

                    # === Define Required Fields for ELO and CSV ===
                    ELO_required_fields = ["homeID", "awayID", "homeGoalCount", "awayGoalCount", "winningTeam"]
                    CSV_required_fields = [
                        "season", "game_week", "home_name", "away_name",
                        "home_ppg", "away_ppg",
                        "team_a_win_percent", "team_b_win_percent",
                        "odds_ft_1", "odds_ft_x", "odds_ft_2",
                        "team_a_rest_days", "team_b_rest_days"
                    ]

                    # Validate ELO required fields
                    missing_elo_fields = [field for field in ELO_required_fields if match.get(field) is None]

                    # Validate CSV required fields
                    missing_csv_fields = [field for field in CSV_required_fields if match.get(field) is None]

                    # Continue processing ELO updates regardless of CSV missing fields
                    if missing_elo_fields:
                        logging.warning(f"Match ID {match_id} is missing ELO-required information: {', '.join(missing_elo_fields)}. Skipping ELO update.")
                        perform_elo_update = False
                    else:
                        perform_elo_update = True

                    # Check if CSV can be exported
                    can_export_csv = not missing_csv_fields

                    # Initialize team mapping
                    if home_id and home_name:
                        team_mapping[home_id] = home_name
                    if away_id and away_name:
                        team_mapping[away_id] = away_name

                    # Initialize ELO ratings if teams are encountered for the first time
                    if home_id not in elo_ratings:
                        elo_ratings[home_id] = INITIAL_ELO
                        logging.info(f"Initialized ELO for team {home_id} to {INITIAL_ELO}.")
                    if away_id not in elo_ratings:
                        elo_ratings[away_id] = INITIAL_ELO
                        logging.info(f"Initialized ELO for team {away_id} to {INITIAL_ELO}.")

                    # Compute winning_team_value based on Method 2 logic
                    # 2: Home Win, 1: Away Win, 0: Draw
                    if winning_team == home_id:
                        winning_team_value = 2  # Home win
                    elif winning_team == away_id:
                        winning_team_value = 1  # Away win
                    elif winning_team is None:
                        winning_team_value = 0  # Draw
                    else:
                        # It's a draw or undefined
                        winning_team_value = 0

                    # === Validate Numerical Fields and Check for Outliers ===
                    outlier_fields = []

                    # Convert and validate home_goals
                    try:
                        home_goals_int = int(home_goals)
                        if not (GOALS_RANGE[0] <= home_goals_int <= GOALS_RANGE[1]):
                            outlier_fields.append(f"homeGoalCount ({home_goals_int})")
                    except (ValueError, TypeError):
                        outlier_fields.append("homeGoalCount (invalid value)")

                    # Convert and validate away_goals
                    try:
                        away_goals_int = int(away_goals)
                        if not (GOALS_RANGE[0] <= away_goals_int <= GOALS_RANGE[1]):
                            outlier_fields.append(f"awayGoalCount ({away_goals_int})")
                    except (ValueError, TypeError):
                        outlier_fields.append("awayGoalCount (invalid value)")

                    # Convert and validate odds_home_win
                    try:
                        odds_home_win_float = float(odds_home_win)
                        if not (ODDS_RANGE[0] <= odds_home_win_float <= ODDS_RANGE[1]):
                            outlier_fields.append(f"odds_ft_1 ({odds_home_win_float})")
                    except (ValueError, TypeError):
                        outlier_fields.append("odds_ft_1 (invalid value)")

                    # Convert and validate odds_draw
                    try:
                        odds_draw_float = float(odds_draw)
                        if not (ODDS_RANGE[0] <= odds_draw_float <= ODDS_RANGE[1]):
                            outlier_fields.append(f"odds_ft_x ({odds_draw_float})")
                    except (ValueError, TypeError):
                        outlier_fields.append("odds_ft_x (invalid value)")

                    # Convert and validate odds_away_win
                    try:
                        odds_away_win_float = float(odds_away_win)
                        if not (ODDS_RANGE[0] <= odds_away_win_float <= ODDS_RANGE[1]):
                            outlier_fields.append(f"odds_ft_2 ({odds_away_win_float})")
                    except (ValueError, TypeError):
                        outlier_fields.append("odds_ft_2 (invalid value)")

                    # Convert and validate home_rest_days
                    try:
                        home_rest_days_float = float(home_rest_days)
                        if not (REST_DAYS_RANGE[0] <= home_rest_days_float <= REST_DAYS_RANGE[1]):
                            outlier_fields.append(f"team_a_rest_days ({home_rest_days_float})")
                    except (ValueError, TypeError):
                        outlier_fields.append("team_a_rest_days (invalid value)")

                    # Convert and validate away_rest_days
                    try:
                        away_rest_days_float = float(away_rest_days)
                        if not (REST_DAYS_RANGE[0] <= away_rest_days_float <= REST_DAYS_RANGE[1]):
                            outlier_fields.append(f"team_b_rest_days ({away_rest_days_float})")
                    except (ValueError, TypeError):
                        outlier_fields.append("team_b_rest_days (invalid value)")

                    # ELO ratings before the match
                    RA_before = elo_ratings.get(home_id, INITIAL_ELO)
                    RB_before = elo_ratings.get(away_id, INITIAL_ELO)

                    # Validate ELO ratings
                    if not (ELO_RANGE[0] <= RA_before <= ELO_RANGE[1]):
                        outlier_fields.append(f"team_a_ELO_before ({RA_before})")
                    if not (ELO_RANGE[0] <= RB_before <= ELO_RANGE[1]):
                        outlier_fields.append(f"team_b_ELO_before ({RB_before})")

                    # Log outlier fields if any
                    if outlier_fields:
                        logging.warning(f"Match ID {match_id} has outlier fields: {', '.join(outlier_fields)}.")

                    # === Calculate ELO Updates Only if All Critical Fields Are Present and No Outliers ===
                    if perform_elo_update and not outlier_fields:
                        # Compute total goals and binary indicators
                        total_goals = home_goals_int + away_goals_int
                        total_over = {f'total_over_{t}': int(total_goals > t) for t in total_thresholds}

                        # Determine actual scores
                        SA_home, result_home = determine_scores(home_goals_int, away_goals_int)
                        SA_away, result_away = determine_scores(away_goals_int, home_goals_int)

                        # Calculate expected scores with home advantage
                        EA_home, EB_away = get_elo(RA_before, RB_before, home_advantage=HOME_ADVANTAGE)

                        # Update ELO ratings
                        RA_after, RB_after = new_elo(RA_before, RB_before, EA_home, EB_away, K_FACTOR, SA_home, SA_away)

                        # Calculate ELO change per team
                        RA_change = RA_after - RA_before
                        RB_change = RB_after - RB_before

                        # Update the dictionary with new ELO ratings
                        elo_ratings[home_id] = RA_after
                        elo_ratings[away_id] = RB_after

                        # Log ELO changes
                        logging.info(f"Match ID {match_id}: ELO updated for {home_id} from {RA_before:.2f} to {RA_after:.2f} (Change: {RA_change:.2f}).")
                        logging.info(f"Match ID {match_id}: ELO updated for {away_id} from {RB_before:.2f} to {RB_after:.2f} (Change: {RB_change:.2f}).")

                        # Determine which CSV to write to based on competition_id
                        if competition_id == current_year_id:
                            target_writer = prediction_writer
                            target_csv = 'predictions_dataset'
                        else:
                            target_writer = training_writer
                            target_csv = 'training_dataset'

                        # Only write to CSV if valid for CSV and not the first competition_id
                        if idx != 0:
                            # Prepare data for both home and away teams
                            teams = [
                                {
                                    'team_id': home_id,
                                    'team_name': home_name,
                                    'opponent_id': away_id,
                                    'opponent_name': away_name,
                                    'is_home': 1,
                                    'team_goal_count': home_goals_int,
                                    'opponent_goal_count': away_goals_int,
                                    'team_ELO_before': RA_before,
                                    'opponent_ELO_before': RB_before,
                                    'result': 2 if winning_team_value == 2 else (1 if winning_team_value == 1 else 0),  # From home team's perspective
                                    'team_h2h_win_percent': h2h_team_a_win_percent,
                                    'opponent_h2h_win_percent': h2h_team_b_win_percent,
                                    'odds_team_win': odds_home_win_float,
                                    'odds_draw': odds_draw_float,
                                    'odds_opponent_win': odds_away_win_float,
                                    'winning_team': winning_team_value,  # Method 2: 2=Home Win,1=Away Win,0=Draw
                                    'team_rest_days': home_rest_days_float,
                                    'opponent_rest_days': away_rest_days_float,
                                    'team_ppg': team_ppg if team_ppg is not None else '',
                                    'opponent_ppg': opponent_ppg if opponent_ppg is not None else '',
                                    'team_ppg_mc': team_ppg_mc if team_ppg_mc is not None else '',
                                    'opponent_ppg_mc': opponent_ppg_mc if opponent_ppg_mc is not None else '',
                                    'pre_match_home_ppg': pre_match_home_ppg if pre_match_home_ppg is not None else '',
                                    'pre_match_away_ppg': pre_match_away_ppg if pre_match_away_ppg is not None else '',
                                    'pre_match_home_xg': pre_match_home_xg if pre_match_home_xg is not None else '',
                                    'pre_match_away_xg': pre_match_away_xg if pre_match_away_xg is not None else '',
                                    'team_home_advantage': team_home_advantage if team_home_advantage is not None else '',
                                    'opponent_home_advantage': 0,
                                    'opponent_away_advantage': opponent_away_advantage if opponent_away_advantage is not None else '',
                                    'team_away_advantage': 0,
                                    'BTTS': btts,  # Add BTTS for home team perspective
                                    'h2h_btts': h2h_btts if h2h_btts is not None else '',
                                    'team_btts_l5': team_btts_l5,
                                    'team_btts_l10': team_btts_l10,
                                    'team_btts_l20': team_btts_l20,
                                    'opponent_btts_l5': opponent_btts_l5,
                                    'opponent_btts_l10': opponent_btts_l10,
                                    'opponent_btts_l20': opponent_btts_l20,
                                    'team_mc_btts_l5': team_mc_btts_l5,
                                    'team_mc_btts_l10': team_mc_btts_l10,
                                    'team_mc_btts_l20': team_mc_btts_l20,
                                    'opponent_mc_btts_l5': opponent_mc_btts_l5,
                                    'opponent_mc_btts_l10': opponent_mc_btts_l10,
                                    'opponent_mc_btts_l20': opponent_mc_btts_l20,
                                    'team_conceded_l5':team_conceded_l5,
                                    'team_conceded_l10': team_conceded_l10,
                                    'team_conceded_l20': team_conceded_l20,
                                    'opponent_conceded_l5':opponent_conceded_l5,
                                    'opponent_conceded_l10':opponent_conceded_l10,
                                    'opponent_conceded_l20': opponent_conceded_l20,
                                    'team_scoring_streak': team_scoring_streak,
                                    'opponent_scoring_streak': opponent_scoring_streak,
                                    'team_conceded_streak': team_conceded_streak,
                                    'opponent_conceded_streak': opponent_conceded_streak,
                                    'betting_h2h_betting_odds': betting_h2h_betting_odds,
                                    'betting_h2h_betting_percentage': betting_h2h_betting_percentage,
                                    
                                
                                    
                                },
                                {
                                    'team_id': away_id,
                                    'team_name': away_name,
                                    'opponent_id': home_id,
                                    'opponent_name': home_name,
                                    'is_home': 0,
                                    'team_goal_count': away_goals_int,
                                    'opponent_goal_count': home_goals_int,
                                    'team_ELO_before': RB_before,
                                    'opponent_ELO_before': RA_before,
                                    'result': 1 if winning_team_value == 1 else (2 if winning_team_value == 2 else 0),  # From away team's perspective
                                    'team_h2h_win_percent': h2h_team_b_win_percent,
                                    'opponent_h2h_win_percent': h2h_team_a_win_percent,
                                    'odds_team_win': odds_away_win_float,
                                    'odds_draw': odds_draw_float,
                                    'odds_opponent_win': odds_home_win_float,
                                    'winning_team': winning_team_value,  # Method 2: 2=Home Win,1=Away Win,0=Draw
                                    'team_rest_days': away_rest_days_float,
                                    'opponent_rest_days': home_rest_days_float,
                                    'team_ppg': opponent_ppg if opponent_ppg is not None else '',
                                    'opponent_ppg': team_ppg if team_ppg is not None else '',
                                    'team_ppg_mc': opponent_ppg_mc if opponent_ppg_mc is not None else '',
                                    'opponent_ppg_mc': team_ppg_mc if team_ppg_mc is not None else '',
                                    'pre_match_home_ppg': pre_match_away_ppg if pre_match_away_ppg is not None else '',
                                    'pre_match_away_ppg': pre_match_home_ppg if pre_match_home_ppg is not None else '',
                                    'pre_match_home_xg': pre_match_away_xg if pre_match_away_xg is not None else '',
                                    'pre_match_away_xg': pre_match_home_xg if pre_match_home_xg is not None else '',
                                    'team_home_advantage': 0,
                                    'opponent_home_advantage': team_home_advantage if team_home_advantage is not None else '',
                                    'opponent_away_advantage': 0,
                                    'team_away_advantage': opponent_away_advantage if opponent_away_advantage is not None else '',
                                    'BTTS': btts,  # Add BTTS for away team perspective
                                    'h2h_btts': h2h_btts if h2h_btts is not None else '',
                                    'team_btts_l5': opponent_btts_l5,
                                    'team_btts_l10': opponent_btts_l10,
                                    'team_btts_l20': opponent_btts_l20,
                                    'opponent_btts_l5': team_btts_l5,
                                    'opponent_btts_l10': team_btts_l10,
                                    'opponent_btts_l20': team_btts_l20,
                                    'team_mc_btts_l5': opponent_mc_btts_l5,
                                    'team_mc_btts_l10': opponent_mc_btts_l10,
                                    'team_mc_btts_l20': opponent_mc_btts_l20,
                                    'opponent_mc_btts_l5': team_mc_btts_l5,
                                    'opponent_mc_btts_l10': team_mc_btts_l10,
                                    'opponent_mc_btts_l20': team_mc_btts_l20,
                                    'team_conceded_l5':opponent_conceded_l5,
                                    'team_conceded_l10':opponent_conceded_l10,
                                    'team_conceded_l20':opponent_conceded_l20,
                                    'opponent_conceded_l5':team_conceded_l5,
                                    'opponent_conceded_l10':team_conceded_l10,
                                    'opponent_conceded_l20':team_conceded_l20,
                                    'team_scoring_streak':opponent_scoring_streak,
                                    'opponent_scoring_streak':team_scoring_streak,
                                    'team_conceded_streak':opponent_conceded_streak,
                                    'opponent_conceded_streak':team_conceded_streak,
                                    'betting_h2h_betting_odds': betting_h2h_betting_odds,
                                    'betting_h2h_betting_percentage': betting_h2h_betting_percentage,
                                    
                                }
                            ]

                            for team_data in teams:
                                # Compute binary features for team-specific expected goals thresholds
                                team_over = {}
                                opponent_over = {}
                                for t in team_thresholds:
                                    team_over[f'team_over_{t}'] = int(team_data['team_goal_count'] > t)
                                    opponent_over[f'opponent_over_{t}'] = int(team_data['opponent_goal_count'] > t)

                                # **Include total_over features**
                                # These are the same for both home and away team rows
                                # They have already been computed as total_over dictionary

                                # Prepare the row data
                                row_data = {
                                    'match_id': match_id_int,
                                    'competition_id': competition_id,
                                    'season': season,
                                    'game_week': game_week,
                                    'match_status': match_status,
                                    'team_id': team_data['team_id'],
                                    'team_name': team_data['team_name'],
                                    'opponent_id': team_data['opponent_id'],
                                    'opponent_name': team_data['opponent_name'],
                                    'is_home': team_data['is_home'],
                                    'team_goal_count': team_data['team_goal_count'],
                                    'opponent_goal_count': team_data['opponent_goal_count'],
                                    'team_ELO_before': f"{team_data['team_ELO_before']:.2f}",
                                    'opponent_ELO_before': f"{team_data['opponent_ELO_before']:.2f}",
                                    'result': team_data['result'],  # 0,1,2
                                    'team_h2h_win_percent': team_data['team_h2h_win_percent'],
                                    'opponent_h2h_win_percent': team_data['opponent_h2h_win_percent'],
                                    'odds_team_win': f"{team_data['odds_team_win']:.4f}",
                                    'odds_draw': f"{team_data['odds_draw']:.4f}",
                                    'odds_opponent_win': f"{team_data['odds_opponent_win']:.4f}",
                                    'winning_team': team_data['winning_team'],  # Method 2: 2=Home Win,1=Away Win,0=Draw
                                    'team_rest_days': f"{team_data['team_rest_days']:.2f}",
                                    'opponent_rest_days': f"{team_data['opponent_rest_days']:.2f}",
                                    'team_ppg': f"{team_data['team_ppg']:.2f}" if isinstance(team_data['team_ppg'], (int, float)) else '',
                                    'opponent_ppg': f"{team_data['opponent_ppg']:.2f}" if isinstance(team_data['opponent_ppg'], (int, float)) else '',
                                    'team_ppg_mc': f"{team_data['team_ppg_mc']}" if team_data['team_ppg_mc'] is not None else '',
                                    'opponent_ppg_mc': f"{team_data['opponent_ppg_mc']}" if team_data['opponent_ppg_mc'] is not None else '',
                                    'pre_match_home_ppg': f"{team_data['pre_match_home_ppg']}" if team_data['pre_match_home_ppg'] is not None else '',
                                    'pre_match_away_ppg': f"{team_data['pre_match_away_ppg']}" if team_data['pre_match_away_ppg'] is not None else '',
                                    'pre_match_home_xg': f"{team_data['pre_match_home_xg']}" if team_data['pre_match_home_xg'] is not None else '',
                                    'pre_match_away_xg': f"{team_data['pre_match_away_xg']}" if team_data['pre_match_away_xg'] is not None else '',
                                    'team_home_advantage': f"{team_data['team_home_advantage']}" if team_data['team_home_advantage'] is not None else '',
                                    'opponent_home_advantage': f"{team_data['opponent_home_advantage']}" if team_data['opponent_home_advantage'] is not None else '',
                                    'team_away_advantage': f"{team_data['team_away_advantage']}" if team_data['team_away_advantage'] is not None else '',
                                    'opponent_away_advantage': f"{team_data['opponent_away_advantage']}" if team_data['opponent_away_advantage'] is not None else '',
                                    'BTTS': btts,  # BTTS value for this match
                                    'h2h_btts': f"{team_data['h2h_btts']}",
                                    'team_btts_l5': f"{team_data['team_btts_l5']}",
                                    'team_btts_l10': f"{team_data['team_btts_l10']}",
                                    'team_btts_l20': f"{team_data['team_btts_l20']}",
                                    'opponent_btts_l5': f"{team_data['opponent_btts_l5']}",
                                    'opponent_btts_l10': f"{team_data['opponent_btts_l10']}",
                                    'opponent_btts_l20': f"{team_data['opponent_btts_l20']}",
                                    'team_mc_btts_l5': f"{team_data['team_mc_btts_l5']}",
                                    'team_mc_btts_l10': f"{team_data['team_mc_btts_l10']}",
                                    'team_mc_btts_l20': f"{team_data['team_mc_btts_l20']}",
                                    'opponent_mc_btts_l5': f"{team_data['opponent_mc_btts_l5']}",
                                    'opponent_mc_btts_l10': f"{team_data['opponent_mc_btts_l10']}",
                                    'opponent_mc_btts_l20': f"{team_data['opponent_mc_btts_l20']}",
                                    'team_conceded_l5': f"{team_data['team_conceded_l5']}",
                                    'team_conceded_l10': f"{team_data['team_conceded_l10']}",
                                    'team_conceded_l20': f"{team_data['team_conceded_l20']}",
                                    'opponent_conceded_l5': f"{team_data['opponent_conceded_l5']}",
                                    'opponent_conceded_l10': f"{team_data['opponent_conceded_l10']}",
                                    'opponent_conceded_l20': f"{team_data['opponent_conceded_l20']}",
                                    'team_scoring_streak': f"{team_data['team_scoring_streak']}",
                                    'opponent_scoring_streak': f"{team_data['opponent_scoring_streak']}",
                                    'team_conceded_streak': f"{team_data['team_conceded_streak']}",
                                    'opponent_conceded_streak': f"{team_data['opponent_conceded_streak']}",
                                    'betting_h2h_betting_odds': f"{team_data['betting_h2h_betting_odds']}",
                                    'betting_h2h_betting_percentage': f"{team_data['betting_h2h_betting_percentage']}",
                                    
                                    # Include binary features
                                    **team_over,
                                    **opponent_over,
                                    **total_over,  # **Unpacks the total_over dictionary
                                    
                                }

                                # Write to the appropriate CSV
                                try:
                                    target_writer.writerow(row_data)
                                    logging.info(f"Match ID {match_id}: Data written to '{target_csv}'.")
                                except Exception as e:
                                    logging.error(f"Failed to write Match ID {match_id} to '{target_csv}': {e}")

                # === After Processing All Matches in the Current Season ===
                current_season_final_elo = {team_id: elo for team_id, elo in elo_ratings.items() if team_id in current_season_team_ids}
                logging.info(f"Final ELOs for Season {idx + 1}: {current_season_final_elo}")

                # Update prior_season_final_elo and prior_season_teams for the next season
                prior_season_final_elo = current_season_final_elo
                prior_season_teams = current_season_team_ids

    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")
        raise  # Re-raise the exception after logging
     
    # === After Processing All Competitions and Matches, Save the Updated ELO Ratings ===
    try:
        joblib.dump(elo_ratings, elo_ratings_file)
        logging.info(f"ELO ratings saved to {elo_ratings_file}.")
    except Exception as e:
        logging.error(f"Failed to save ELO ratings to {elo_ratings_file}: {e}")
    
    logging.info(f"ELO ratings per match have been saved to '{training_dataset}' and '{prediction_dataset}'.")

if __name__ == "__main__":
    main()
