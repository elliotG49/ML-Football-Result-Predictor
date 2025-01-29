import pandas as pd
import joblib
import os
import yaml
import argparse
import sys
from colorama import init, Fore, Style
from pymongo import MongoClient, UpdateOne
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from datetime import datetime

# Initialize colorama
init(autoreset=True)

# MongoDB Configuration (Hard-coded)
MONGO_URI = 'mongodb://localhost:27017'  # MongoDB URI
DB_NAME = 'footballDB'                   # Database name
MATCHES_COLLECTION_NAME = 'matches'      # Matches collection name

# Define your feature list as used during training (odds are excluded)
features = [
    'team_id', 'opponent_id',
    'team_ELO_before', 'opponent_ELO_before',
    'odds_team_win',
    'odds_draw',
    'odds_opponent_win',
    'opponent_rest_days', 'team_rest_days',
    'team_h2h_win_percent', 'opponent_h2h_win_percent',
    #'team_ppg', 'opponent_ppg',
    #'team_ppg_mc', 'opponent_ppg_mc',
    'pre_match_home_ppg', 'pre_match_away_ppg',
    #'pre_match_home_xg', 'pre_match_away_xg',
    'team_home_advantage', 'opponent_home_advantage',
    #'team_away_advantage', 'opponent_away_advantage'
]

def load_trained_model(model_path):
    """
    Loads a trained model from the specified path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file '{model_path}' does not exist.")
    try:
        model = joblib.load(model_path)
        print(f"{Fore.GREEN}Model loaded successfully from '{model_path}'.{Style.RESET_ALL}")
        return model
    except Exception as e:
        print(f"{Fore.RED}Error loading model: {e}{Style.RESET_ALL}")
        raise e

def load_label_encoders(encoders_info):
    """
    Loads label encoders from the specified directory.
    """
    label_encoders = {}
    for encoder_name, encoder_path in encoders_info.items():
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"The encoder file '{encoder_path}' does not exist.")
        try:
            encoder = joblib.load(encoder_path)
            label_encoders[encoder_name] = encoder
            print(f"{Fore.GREEN}Label encoder '{encoder_name}' loaded successfully.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error loading encoder '{encoder_name}': {e}{Style.RESET_ALL}")
            raise e
    return label_encoders

def connect_to_mongodb(mongo_uri, db_name):
    """
    Connects to MongoDB and returns the database object.
    """
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        print(f"{Fore.GREEN}Connected to MongoDB database '{db_name}' successfully.{Style.RESET_ALL}")
        return db
    except Exception as e:
        print(f"{Fore.RED}Failed to connect to MongoDB: {e}{Style.RESET_ALL}")
        sys.exit(1)

def load_prediction_dataset(dataset_path):
    """
    Loads the prediction dataset from the specified CSV file.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset file '{dataset_path}' does not exist.")
    try:
        df = pd.read_csv(dataset_path)
        print(f"{Fore.GREEN}Prediction dataset loaded successfully from '{dataset_path}'.{Style.RESET_ALL}")
        return df
    except Exception as e:
        print(f"{Fore.RED}Error loading prediction dataset: {e}{Style.RESET_ALL}")
        sys.exit(1)

def verify_columns(df, required_columns):
    """
    Verifies that the required columns are present in the DataFrame.
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise KeyError(f"The following required columns are missing: {missing}")

def preprocess_dataset(df, features, label_encoders=None):
    """
    Preprocesses the dataset for prediction.
    """
    # Ensure all required features are present
    missing_features = set(features) - set(df.columns)
    if missing_features:
        print(f"{Fore.RED}Missing features in the dataset: {missing_features}{Style.RESET_ALL}")
        sys.exit(1)

    X_new = df[features].copy()

    # Handle missing values
    X_new = X_new.fillna(0)
    print(f"{Fore.YELLOW}Handled missing values in the dataset.{Style.RESET_ALL}")

    # Apply label encoders if any
    if label_encoders:
        for column, encoder in label_encoders.items():
            try:
                X_new[column] = encoder.transform(X_new[column])
                print(f"{Fore.YELLOW}Applied encoder to column '{column}'.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error encoding column '{column}': {e}{Style.RESET_ALL}")
                sys.exit(1)

    return X_new, df

def make_predictions(model, X_new):
    """
    Makes predictions and computes prediction probabilities.
    """
    try:
        predictions = model.predict(X_new)
        prediction_proba = model.predict_proba(X_new)
        print(f"{Fore.GREEN}Predictions made successfully.{Style.RESET_ALL}")
        return predictions, prediction_proba
    except Exception as e:
        print(f"{Fore.RED}Error during prediction: {e}{Style.RESET_ALL}")
        sys.exit(1)

def combine_predictions_per_match(df_new, predictions, prediction_proba):
    """
    Combines predictions by averaging class probabilities per match and determining the final prediction.
    """
    # Add predictions and probabilities to the DataFrame
    df_new = df_new.copy()
    df_new['Predicted_Class'] = predictions
    # Assuming the classes are ordered as ['Draw', 'Away Win', 'Home Win']
    df_new[['Prob_Draw', 'Prob_Away_Win', 'Prob_Home_Win']] = prediction_proba

    # Group by 'match_id' and compute average probabilities
    aggregated = df_new.groupby('match_id').agg({
        'Prob_Draw': 'mean',
        'Prob_Away_Win': 'mean',
        'Prob_Home_Win': 'mean',
        'winning_team': 'first',       # Assuming both rows have the same winning_team
        'team_name': 'first',          # Retain team names
        'opponent_name': 'first',
        'game_week': 'first',
        'match_status': 'first'
    }).reset_index()

    # Determine final prediction based on highest average probability
    aggregated['Final_Predicted_Class'] = aggregated[['Prob_Draw', 'Prob_Away_Win', 'Prob_Home_Win']].idxmax(axis=1)

    # Map probability column names to class labels
    probability_mapping = {
        'Prob_Draw': 'Draw',       
        'Prob_Away_Win': 'Away Win',   
        'Prob_Home_Win': 'Home Win'    
    }
    aggregated['Final_Predicted_Class'] = aggregated['Final_Predicted_Class'].map(probability_mapping)

    print(f"{Fore.YELLOW}Combined predictions per match by averaging class probabilities.{Style.RESET_ALL}")
    return aggregated

def update_matches_with_predictions(db, collection_name, df_combined):
    """
    Updates match documents in MongoDB with the combined prediction probabilities under the 'Predictions' field.
    """
    matches_collection = db[collection_name]
    bulk_operations = []
    for index, row in df_combined.iterrows():
        match_id = row['match_id']  # Assuming 'match_id' is the unique identifier
        # Get the averaged probabilities
        prob_draw = float(row['Prob_Draw'])
        prob_away_win = float(row['Prob_Away_Win'])
        prob_home_win = float(row['Prob_Home_Win'])
        final_prediction = row['Final_Predicted_Class']
        # Prepare the update document
        update_doc = {
            'Predictions.Match_Result': {
                'Prob_Draw': prob_draw,
                'Prob_Away_Win': prob_away_win,
                'Prob_Home_Win': prob_home_win,
                'Final_Predicted_Class': final_prediction
            },
            'Predictions.Last_Updated': datetime.utcnow()
        }
        # Prepare the bulk update operation
        bulk_operations.append(
            UpdateOne(
                {'id': match_id},  # Adjust the field name if necessary
                {'$set': update_doc}
            )
        )
    # Execute bulk write operation
    if bulk_operations:
        try:
            result = matches_collection.bulk_write(bulk_operations)
            print(f"{Fore.GREEN}Updated {result.modified_count} matches with predictions.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error updating matches with predictions: {e}{Style.RESET_ALL}")
            sys.exit(1)
    else:
        print(f"{Fore.YELLOW}No matches to update.{Style.RESET_ALL}")

def get_last_completed_game_week(db, collection_name, competition_ids):
    """
    Retrieves the highest game_week number where matches have match_status 'complete' for the specified competitions.
    """
    matches_collection = db[collection_name]
    pipeline = [
        {'$match': {'match_status': {'$regex': '^complete$', '$options': 'i'}, 'competition_id': {'$in': competition_ids}}},
        {'$group': {'_id': None, 'max_game_week': {'$max': '$game_week'}}}
    ]
    result = list(matches_collection.aggregate(pipeline))
    if result:
        return result[0]['max_game_week']
    else:
        return None

def get_next_game_week(db, collection_name, last_completed_game_week, competition_ids):
    """
    Retrieves the next game week after the last completed one.
    """
    matches_collection = db[collection_name]
    pipeline = [
        {'$match': {'game_week': {'$gt': last_completed_game_week}, 'competition_id': {'$in': competition_ids}}},
        {'$group': {'_id': None, 'next_game_week': {'$min': '$game_week'}}}
    ]
    result = list(matches_collection.aggregate(pipeline))
    if result:
        return result[0]['next_game_week']
    else:
        return None

if __name__ == "__main__":
    # Initialize colorama
    init(autoreset=True)

    # Define league name and optional game_week
    parser = argparse.ArgumentParser(description='Predict Match Results and save predictions to MongoDB.')
    parser.add_argument('league_name', type=str, help='Name of the league (e.g., premier_league).')
    parser.add_argument('--game_week', type=int, required=True, help='Game week number to perform predictions for.')
    args = parser.parse_args()

    league_name = args.league_name
    specified_game_week = args.game_week  # New argument for specifying the game_week

    # Path to the YAML configuration files
    config_base_path = '/root/project-barnard/ML/Configs/'
    config_file = os.path.join(config_base_path, f'{league_name}.yaml')

    # Load YAML configuration
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"{Fore.GREEN}Loaded configuration from '{config_file}'.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error loading configuration file '{config_file}': {e}{Style.RESET_ALL}")
        sys.exit(1)

    # Extract configurations from YAML
    try:
        prediction_model_path = config['prediction_model_pathway']['Match_Result']
        competition_ids_dict = config['competition_ids']
        competition_ids = list(competition_ids_dict.values())  # Extract IDs into a list
        dataset_path = config['predictions_pathways']['Match_Result']
    except KeyError as e:
        print(f"{Fore.RED}Missing key in configuration file: {e}{Style.RESET_ALL}")
        sys.exit(1)


    # Load the trained model
    trained_model = load_trained_model(prediction_model_path)

    # Load label encoders if any (update encoders_info accordingly)
    # For this example, let's assume no label encoders are used
    encoders_info = {}  # Dictionary of encoder names and their paths
    label_encoders = load_label_encoders(encoders_info) if encoders_info else None

    # Load the prediction dataset from CSV
    df_prediction = load_prediction_dataset(dataset_path)

    # Verify that all required columns are present
    required_columns = features + [
        'match_id', 'team_name', 'opponent_name', 'winning_team',
        'game_week', 'match_status', 'competition_id'
    ]
    try:
        verify_columns(df_prediction, required_columns)
    except KeyError as e:
        print(f"{Fore.RED}Data verification failed: {e}{Style.RESET_ALL}")
        sys.exit(1)

    # Connect to MongoDB
    db = connect_to_mongodb(MONGO_URI, DB_NAME)

    # Optional: If you still want to get last_completed_game_week and next_game_week
    # but since we're specifying the game_week, we might not need them.
    # However, to ensure the specified_game_week is valid, you can perform checks.

    # Check if the specified_game_week exists in the dataset
    if specified_game_week not in df_prediction['game_week'].unique():
        print(f"{Fore.RED}The specified game_week '{specified_game_week}' does not exist in the dataset.{Style.RESET_ALL}")
        sys.exit(1)

    # Filter the dataset to include only matches from the specified_game_week
    df_filtered = df_prediction[
        (df_prediction['game_week'] == specified_game_week) &
        (df_prediction['competition_id'].isin(competition_ids))
    ]

    if df_filtered.empty:
        print(f"{Fore.YELLOW}No incomplete matches found for game_week '{specified_game_week}'. Nothing to predict.{Style.RESET_ALL}")
        sys.exit(0)

    print(f"{Fore.GREEN}Number of matches to predict for game_week '{specified_game_week}': {len(df_filtered)}{Style.RESET_ALL}")

    # Preprocess the dataset
    X_new, df_matches = preprocess_dataset(df_filtered, features, label_encoders=label_encoders)

    # Make predictions
    predictions, prediction_proba = make_predictions(trained_model, X_new)

    # Combine predictions per match
    df_combined = combine_predictions_per_match(df_matches, predictions, prediction_proba)

    # Update matches in MongoDB with predictions
    update_matches_with_predictions(db, MATCHES_COLLECTION_NAME, df_combined)

    print(f"{Fore.GREEN}Predictions for game_week '{specified_game_week}' have been saved to MongoDB successfully.{Style.RESET_ALL}")
