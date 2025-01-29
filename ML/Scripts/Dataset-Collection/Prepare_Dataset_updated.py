import csv
from pymongo import MongoClient
import os      # For file path handling
import logging  # For logging
import argparse  # For command-line arguments
import yaml      # For YAML configuration

def parse_arguments():
    """
    Parse command-line arguments for league and dataset type.
    """
    parser = argparse.ArgumentParser(description='Dataset Generator using Precomputed ELO Ratings')
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
    
    # === Define Paths Based on Configuration ===
    training_dataset = training_dataset_pathway[dataset_type]
    prediction_dataset = predictions_pathways[dataset_type]
    
    # Define other paths (can also be moved to YAML if needed)
    LOG_FILE_PATH = "/root/barnard/logs/elo_update.log"
    
    # === Configure Logging ===
    setup_logging(LOG_FILE_PATH)
    logging.info(f"Starting dataset generation script for league '{league}' and dataset type '{dataset_type}'.")
    
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
    
    # === Initialize Team Mapping Dictionary ===
    team_mapping = {}
    
    # === Define Acceptable Ranges for Numerical Fields to Detect Outliers ===
    GOALS_RANGE = (0, 15)
    ELO_RANGE = (0, 4000)
    ODDS_RANGE = (0.1, 1000.0)
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
                "under_05_potential",
                        "under_15_potential",
                        "under_25_potential",
                        "under_35_potential",
                        "over_05_potential",
                        "over_15_potential",
                        "over_25_potential",
                        "over_35_potential"
            ]

            # Initialize CSV writers
            training_writer = csv.DictWriter(training_csvfile, fieldnames=per_match_fieldnames)
            prediction_writer = csv.DictWriter(prediction_csvfile, fieldnames=per_match_fieldnames)

            # Write headers to both CSV files
            training_writer.writeheader()
            prediction_writer.writeheader()

            logging.info(f"CSV files '{training_dataset}' and '{prediction_dataset}' initialized with headers.")

            # === Initialize Variables to Track Seasons ===
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

                # Retrieve all matches for this competition_id (sorted by date_unix ascending)
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

                # Update prior_season_teams for the next season
                prior_season_teams = current_season_team_ids

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
                    season = match.get("season")             # Season field
                    game_week = match.get("game_week")       # Game week field
                    h2h_stats = match.get("h2h", {})
                    betting_stats = h2h_stats.get("betting_stats", {})
                    previous_matches = h2h_stats.get("previous_matches_results", {})
                    btts = match.get("BTTS", {})

                    home_id = match.get("homeID")
                    away_id = match.get("awayID")
                    home_name = match.get("home_name")       # Home team name    
                    away_name = match.get("away_name")       # Away team name
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
                    
                    under_05_potential = match.get("u05_potential")
                    under_15_potential = match.get("u15_potential")
                    under_25_potential = match.get("u25_potential")
                    under_35_potential = match.get("u35_potential")
                    
                    over_05_potential = match.get("o05_potential")
                    over_15_potential = match.get("o15_potential")
                    over_25_potential = match.get("o25_potential")
                    over_35_potential = match.get("o35_potential")
                    

                    # Get ELO ratings from match document
                    home_elo_pre_match_HA = match.get('home_elo_pre_match_HA')
                    away_elo_pre_match_HA = match.get('away_elo_pre_match_HA')

                    # === Define Required Fields for CSV ===
                    CSV_required_fields = [
                        "season", "game_week", "home_name", "away_name",
                        "home_ppg", "away_ppg",
                        "team_a_win_percent", "team_b_win_percent",
                        "odds_ft_1", "odds_ft_x", "odds_ft_2",
                        "team_a_rest_days", "team_b_rest_days",
                        "home_elo_pre_match_HA", "away_elo_pre_match_HA"
                        "under_05_potential",
                        "under_15_potential",
                        "under_25_potential",
                        "under_35_potential",
                        "over_05_potential",
                        "over_15_potential",
                        "over_25_potential",
                        "over_35_potential"
                    ]

                    # Validate CSV required fields
                    missing_csv_fields = [field for field in CSV_required_fields if match.get(field) is None]

                    # Check if CSV can be exported
                    can_export_csv = not missing_csv_fields

                    # Initialize team mapping
                    if home_id and home_name:
                        team_mapping[home_id] = home_name
                    if away_id and away_name:
                        team_mapping[away_id] = away_name

                    # Compute winning_team_value for the entire match (not perspective-specific)
                    # 2: Home Win, 1: Away Win, 0: Draw/None
                    if winning_team == home_id:
                        winning_team_value = 2  # Home win
                    elif winning_team == away_id:
                        winning_team_value = 1  # Away win
                    elif winning_team is None:
                        winning_team_value = 0  # Draw
                    else:
                        winning_team_value = 0  # Fallback for unexpected cases

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
                    try:
                        RA_before = float(home_elo_pre_match_HA)
                        if not (ELO_RANGE[0] <= RA_before <= ELO_RANGE[1]):
                            outlier_fields.append(f"home_elo_pre_match_HA ({RA_before})")
                    except (ValueError, TypeError):
                        outlier_fields.append("home_elo_pre_match_HA (invalid value)")

                    try:
                        RB_before = float(away_elo_pre_match_HA)
                        if not (ELO_RANGE[0] <= RB_before <= ELO_RANGE[1]):
                            outlier_fields.append(f"away_elo_pre_match_HA ({RB_before})")
                    except (ValueError, TypeError):
                        outlier_fields.append("away_elo_pre_match_HA (invalid value)")

                    # Log outlier fields if any
                    if outlier_fields:
                        logging.warning(f"Match ID {match_id} has outlier fields: {', '.join(outlier_fields)}.")

                    # === Only write out the row if no outlier fields ===
                    if not outlier_fields:
                        # Determine which CSV to write to based on competition_id
                        if competition_id == current_year_id:
                            target_writer = prediction_writer
                            target_csv = 'predictions_dataset'
                        else:
                            target_writer = training_writer
                            target_csv = 'training_dataset'

                        # Only write to CSV if valid for CSV and not the first competition_id
                        if idx != 0:
                            # === Create a single row from the Home Team perspective ===
                            home_team_data = {
                                'match_id': match_id_int,
                                'competition_id': competition_id,
                                'season': season,
                                'game_week': game_week,
                                'match_status': match_status,
                                'team_id': home_id,
                                'team_name': home_name,
                                'opponent_id': away_id,
                                'opponent_name': away_name,
                                'is_home': 1,  # Home perspective
                                
                                'team_goal_count': home_goals_int,
                                'opponent_goal_count': away_goals_int,
                                
                                # ELO
                                'team_ELO_before': f"{RA_before:.2f}",
                                'opponent_ELO_before': f"{RB_before:.2f}",
                                
                                # Result from the home team's perspective
                                # winning_team_value=2 => Home Win => 'result' = 2
                                # winning_team_value=1 => Away Win => 'result' = 1
                                # winning_team_value=0 => Draw => 'result' = 0
                                'result': 2 if winning_team_value == 2 else (1 if winning_team_value == 1 else 0),
                                
                                'team_h2h_win_percent': h2h_team_a_win_percent,
                                'opponent_h2h_win_percent': h2h_team_b_win_percent,
                                'odds_team_win': f"{odds_home_win_float:.4f}",
                                'odds_draw': f"{odds_draw_float:.4f}",
                                'odds_opponent_win': f"{odds_away_win_float:.4f}",
                                'winning_team': winning_team_value,  # 2=Home,1=Away,0=Draw
                                
                                'team_rest_days': f"{home_rest_days_float:.2f}",
                                'opponent_rest_days': f"{away_rest_days_float:.2f}",
                                
                                'team_ppg': f"{team_ppg:.2f}" if isinstance(team_ppg, (int, float)) else '',
                                'opponent_ppg': f"{opponent_ppg:.2f}" if isinstance(opponent_ppg, (int, float)) else '',
                                'team_ppg_mc': f"{team_ppg_mc}" if team_ppg_mc is not None else '',
                                'opponent_ppg_mc': f"{opponent_ppg_mc}" if opponent_ppg_mc is not None else '',
                                
                                'pre_match_home_ppg': f"{pre_match_home_ppg}" if pre_match_home_ppg is not None else '',
                                'pre_match_away_ppg': f"{pre_match_away_ppg}" if pre_match_away_ppg is not None else '',
                                
                                'pre_match_home_xg': f"{pre_match_home_xg}" if pre_match_home_xg is not None else '',
                                'pre_match_away_xg': f"{pre_match_away_xg}" if pre_match_away_xg is not None else '',
                                
                                'team_home_advantage': f"{team_home_advantage}" if team_home_advantage is not None else '',
                                'opponent_home_advantage': '',  # 0 from home perspective or no value
                                'team_away_advantage': '',      # 0 from home perspective or no value
                                'opponent_away_advantage': f"{opponent_away_advantage}" if opponent_away_advantage is not None else '',
                                'under_05_potential':   under_05_potential,
                                'under_15_potential': under_15_potential,
                                'under_25_potential': under_25_potential,
                                'under_35_potential': under_35_potential,
                                'over_05_potential': over_05_potential,
                    
                                'over_15_potential': over_15_potential,
                                'over_25_potential': over_25_potential,
                                'over_35_potential': over_35_potential,
                            }

                            try:
                                target_writer.writerow(home_team_data)
                                logging.info(f"Match ID {match_id}: Data written to '{target_csv}' (Home perspective only).")
                            except Exception as e:
                                logging.error(f"Failed to write Match ID {match_id} to '{target_csv}': {e}")

    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")
        raise  # Re-raise the exception after logging
     
    logging.info(f"Dataset generation completed. Files saved to '{training_dataset}' and '{prediction_dataset}'.")

if __name__ == "__main__":
    main()
