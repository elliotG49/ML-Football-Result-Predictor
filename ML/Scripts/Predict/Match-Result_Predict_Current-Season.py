import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from colorama import init, Fore, Style
import argparse
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import numpy as np
from pymongo import MongoClient, errors
import datetime  # For handling timestamps

# Initialize colorama
init(autoreset=True)

# Define your feature list as used during training (odds are excluded)
features = [
    'team_id', 'opponent_id',
    'team_ELO_before', 'opponent_ELO_before',
    #'odds_team_win',
    'odds_draw',
    #'odds_opponent_win',
    'opponent_rest_days', 'team_rest_days',
    #'is_home',
    'team_h2h_win_percent', 'opponent_h2h_win_percent',
    #'team_ppg', 'opponent_ppg',
    #'team_ppg_mc', 'opponent_ppg_mc',
    'pre_match_home_ppg', 'pre_match_away_ppg',
    #'pre_match_home_xg', 'pre_match_away_xg',
    'team_home_advantage', 'opponent_home_advantage',
    #'team_away_advantage', 'opponent_away_advantage'
]

def convert_numpy_types(obj):
    """
    Recursively convert numpy data types to native Python types in a data structure.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def load_trained_model(model_name, model_dir='/root/barnard/machine-learning/trained-models'):
    """
    Loads a trained model without applying any calibration.
    """
    if not model_name.endswith('.joblib'):
        model_filename = os.path.join(model_dir, f"{model_name}.joblib")
    else:
        model_filename = os.path.join(model_dir, model_name)
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"The model file '{model_filename}' does not exist.")
    try:
        # Load the model
        model = joblib.load(model_filename)
        print(f"{Fore.GREEN}Model '{model_name}' loaded successfully from '{model_filename}'.{Style.RESET_ALL}")
        return model
    except Exception as e:
        print(f"{Fore.RED}Error loading model '{model_name}': {e}{Style.RESET_ALL}")
        raise e




def load_new_data(csv_path):
    """
    Loads new data from a CSV file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The CSV file '{csv_path}' does not exist.")
    try:
        df_new = pd.read_csv(csv_path)
        print(f"{Fore.GREEN}New data loaded successfully from '{csv_path}'.{Style.RESET_ALL}")
        return df_new
    except Exception as e:
        print(f"{Fore.RED}Error loading data from '{csv_path}': {e}{Style.RESET_ALL}")
        raise e

def verify_columns(df, required_columns):
    """
    Verifies that the required columns are present in the DataFrame.
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise KeyError(f"The following required columns are missing: {missing}")

def preprocess_new_data(df_new, features):
    """
    Preprocesses the new data for prediction.
    """
    try:
        X_new = df_new[features].copy()
        print(f"{Fore.GREEN}Selected required features for prediction.{Style.RESET_ALL}")
    except KeyError as e:
        missing_features = list(set(features) - set(df_new.columns))
        print(f"{Fore.RED}Missing features in the new data: {missing_features}{Style.RESET_ALL}")
        raise KeyError(f"Missing features: {missing_features}")
    
    # Handle missing values
    X_new = X_new.fillna(0)
    print(f"{Fore.GREEN}Handled missing values in new data.{Style.RESET_ALL}")
    
    return X_new

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
        raise e

def combine_predictions_per_match(df_new, predictions, prediction_proba):
    """
    Combines predictions by averaging class probabilities per match and determining the final prediction.
    """
    # Add predictions and probabilities to the DataFrame
    df_new = df_new.copy()
    df_new['Predicted_Winning_Team'] = predictions
    df_new[['Prob_Draw', 'Prob_Away_Win', 'Prob_Home_Win']] = prediction_proba
    
    # Group by 'match_id' and compute average probabilities
    aggregated = df_new.groupby('match_id').agg({
        'Prob_Draw': 'mean',
        'Prob_Away_Win': 'mean',
        'Prob_Home_Win': 'mean',
        'winning_team': 'first',       # Assuming all rows for a match have the same winning_team
        'team_name': 'first',          # Retain team names
        'opponent_name': 'first',
        'odds_draw': 'first',          # Retain odds columns
        'odds_opponent_win': 'first',
        'odds_team_win': 'first',
        'competition_id': 'first'       # Retain competition_id
    }).reset_index()
    
    # Determine final prediction based on highest average probability
    aggregated['Final_Predicted_Winning_Team'] = aggregated[['Prob_Draw', 'Prob_Away_Win', 'Prob_Home_Win']].idxmax(axis=1)
    
    # Map probability column names to class labels
    probability_mapping = {
        'Prob_Draw': 0,       # 0: Draw
        'Prob_Away_Win': 1,   # 1: Away Win
        'Prob_Home_Win': 2    # 2: Home Win
    }
    aggregated['Final_Predicted_Winning_Team'] = aggregated['Final_Predicted_Winning_Team'].map(probability_mapping)
    
    print(f"{Fore.GREEN}Combined predictions per match by averaging class probabilities.{Style.RESET_ALL}")
    return aggregated

def evaluate_model_performance_combined(df_combined):
    """
    Evaluates and returns the base accuracy of the model on combined match data.
    """
    try:
        true_labels = df_combined['winning_team']
        predictions = df_combined['Final_Predicted_Winning_Team']
        accuracy = accuracy_score(true_labels, predictions)
        print(f"{Fore.GREEN}Base Accuracy: {accuracy*100:.2f}%{Style.RESET_ALL}")
        return accuracy
    except KeyError:
        print(f"{Fore.RED}The combined data does not contain the required columns for evaluation.{Style.RESET_ALL}")
        raise KeyError("Missing required columns in combined data.")

def evaluate_threshold_accuracy_combined(df_combined, threshold=0.51):
    """
    Evaluates and returns the threshold accuracy of predictions above a specified confidence.
    """
    try:
        # Determine the probability column corresponding to each prediction
        def get_predicted_probability(row):
            predicted_class = row['Final_Predicted_Winning_Team']
            if predicted_class == 0:
                return row['Prob_Draw']
            elif predicted_class == 1:
                return row['Prob_Away_Win']
            elif predicted_class == 2:
                return row['Prob_Home_Win']
            else:
                return 0  # Unknown class

        df_combined['Predicted_Confidence'] = df_combined.apply(get_predicted_probability, axis=1)

        # Filter predictions based on the threshold
        df_threshold = df_combined[df_combined['Predicted_Confidence'] >= threshold]

        if df_threshold.empty:
            print(f"{Fore.YELLOW}No predictions with confidence >= {threshold*100:.0f}%.{Style.RESET_ALL}")
            threshold_accuracy = None
        else:
            # Calculate accuracy
            correct_predictions = df_threshold['Final_Predicted_Winning_Team'] == df_threshold['winning_team']
            threshold_accuracy = correct_predictions.mean()
            print(f"{Fore.GREEN}Threshold Accuracy (Confidence >= {threshold*100:.0f}%): {threshold_accuracy*100:.2f}%{Style.RESET_ALL}")

        return threshold_accuracy
    except KeyError:
        print(f"{Fore.RED}The combined data does not contain the required columns for threshold evaluation.{Style.RESET_ALL}")
        raise KeyError("Missing required columns in combined data.")

def compute_betting_returns(df_combined, class_to_odds_column, confidence_thresholds, stake_amounts):
    """
    Computes the return/loss for betting based on the model's predicted outcome and confidence thresholds.
    """
    try:
        # Ensure that the lengths match
        if len(confidence_thresholds) != len(stake_amounts):
            raise ValueError("Length of confidence_thresholds and stake_amounts must be equal.")

        # Define a function to determine stake based on prediction confidence
        def determine_stake(row):
            # Confidence is the probability of the predicted class
            predicted_class = row['Final_Predicted_Winning_Team']
            prob_column = ''
            if predicted_class == 0:
                prob_column = 'Prob_Draw'
            elif predicted_class == 1:
                prob_column = 'Prob_Away_Win'
            elif predicted_class == 2:
                prob_column = 'Prob_Home_Win'
            else:
                return 0

            confidence = row[prob_column]
            # Iterate through thresholds to determine stake
            for threshold, stake in zip(confidence_thresholds, stake_amounts):
                if confidence >= threshold:
                    return stake
            return 0  # No bet if confidence is below all thresholds

        # Apply the function to determine stake for each match
        df_combined['Stake_Amount'] = df_combined.apply(determine_stake, axis=1)

        # Define a function to calculate return per row
        def calculate_return(row):
            predicted_class = row['Final_Predicted_Winning_Team']
            actual_class = row['winning_team']
            odds_column = class_to_odds_column.get(predicted_class)
            if not odds_column:
                return 0
            odds = row.get(odds_column, 0)
            if pd.isna(odds):
                return -row['Stake_Amount']  # Lose the stake if odds not available
            if predicted_class == actual_class:
                return (odds * row['Stake_Amount']) - row['Stake_Amount']  # Profit
            else:
                return -row['Stake_Amount']  # Loss

        # Apply the function to calculate bet return
        df_combined['Bet_Return'] = df_combined.apply(calculate_return, axis=1)

        # Compute total profit/loss
        total_return = df_combined['Bet_Return'].sum()

        return df_combined, total_return
    except Exception as e:
        print(f"{Fore.RED}Failed to compute betting returns: {e}{Style.RESET_ALL}")
        raise e

def calculate_total_stake_and_roi(df_combined):
    """
    Calculates and returns the ROI.
    """
    try:
        # Calculate total stake
        total_stake = df_combined['Stake_Amount'].sum()
        # Calculate total profit/loss
        total_profit_loss = df_combined['Bet_Return'].sum()
        # Calculate ROI
        roi = (total_profit_loss / total_stake) * 100 if total_stake != 0 else 0

        print(f"{Fore.GREEN}Return on Investment (ROI): {roi:.2f}%{Style.RESET_ALL}")

        # Return the ROI
        return roi
    except Exception as e:
        print(f"{Fore.RED}Failed to calculate Total Stake and ROI: {e}{Style.RESET_ALL}")
        raise e

def save_metrics_to_db(competition_id, accuracy, threshold_accuracy, roi, league_name_pretty, mongo_uri='mongodb://localhost:27017/', db_name='footballDB', collection_name='leagues'):
    """
    Saves the metrics into the MongoDB 'leagues' collection. If a document with the same competition_id exists, it updates it.
    
    Includes 'league_name_pretty' in the document.
    """
    try:
        # Establish MongoDB connection
        client = MongoClient(mongo_uri)
        db = client[db_name]
        leagues_collection = db[collection_name]

        # Ensure a unique index on competition_id
        leagues_collection.create_index('competition_id', unique=True)
        print(f"{Fore.GREEN}Connected to MongoDB and ensured unique index on 'competition_id'.{Style.RESET_ALL}")

        # Prepare the document with native Python types
        document = {
            'competition_id': int(competition_id),
            'accuracy': float(accuracy),
            'threshold_accuracy': float(threshold_accuracy) if threshold_accuracy is not None else None,
            'roi': float(roi),
            'league_name_pretty': league_name_pretty,  # Add this field
            'timestamp': datetime.datetime.utcnow()  # Use native datetime
        }

        # Upsert the document
        result = leagues_collection.update_one(
            {'competition_id': document['competition_id']},
            {'$set': document},
            upsert=True
        )

        if result.matched_count > 0:
            print(f"{Fore.GREEN}Updated metrics for competition_id {document['competition_id']} in '{collection_name}' collection.{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}Inserted new metrics for competition_id {document['competition_id']} into '{collection_name}' collection.{Style.RESET_ALL}")

    except errors.PyMongoError as e:
        print(f"{Fore.RED}MongoDB Error: {e}{Style.RESET_ALL}")
        raise e
    finally:
        client.close()
        print(f"{Fore.GREEN}Disconnected from MongoDB.{Style.RESET_ALL}")

def plot_class_distribution(df, column, class_names, title, plot_path):
    """
    Plots and saves the distribution of classes in a specified column.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - column (str): Column name to plot the distribution for.
    - class_names (dict): Dictionary mapping class labels to class names.
    - title (str): Title of the plot.
    - plot_path (str): Path to save the plot image.
    """
    try:
        # Map class labels to class names for readability
        df_mapped = df[column].map(class_names)
        
        # Count the occurrences of each class
        class_counts = df_mapped.value_counts().sort_index()
        
        # Plot the distribution
        plt.figure(figsize=(8, 6))
        sns.barplot(x=class_counts.index, y=class_counts.values, palette='pastel')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"{Fore.GREEN}{title} plot saved to '{plot_path}'.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error generating {title} plot: {e}{Style.RESET_ALL}")
        raise e

def compute_and_plot_distributions_combined(df_combined, class_labels, class_names, metrics_dir):
    """
    Computes and plots the distributions of actual and predicted classes for combined match data.
    
    Parameters:
    - df_combined (pd.DataFrame): DataFrame with combined predictions per match.
    - class_labels (list): List of class labels.
    - class_names (dict): Dictionary mapping class labels to class names.
    - metrics_dir (str): Directory to save the plots.
    """
    try:
        # Create a DataFrame for actual and predicted classes
        df_distributions = pd.DataFrame({
            'Actual': df_combined['winning_team'],
            'Predicted': df_combined['Final_Predicted_Winning_Team']
        })
        
        # Define plot paths
        actual_distribution_plot = os.path.join(metrics_dir, 'actual_class_distribution.png')
        predicted_distribution_plot = os.path.join(metrics_dir, 'predicted_class_distribution.png')
        
        # Plot Actual Class Distribution
        plot_class_distribution(
            df=df_distributions,
            column='Actual',
            class_names=class_names,
            title='Actual Class Distribution',
            plot_path=actual_distribution_plot
        )
        
        # Plot Predicted Class Distribution
        plot_class_distribution(
            df=df_distributions,
            column='Predicted',
            class_names=class_names,
            title='Predicted Class Distribution',
            plot_path=predicted_distribution_plot
        )
    except Exception as e:
        print(f"{Fore.RED}Failed to compute and plot class distributions: {e}{Style.RESET_ALL}")
        raise e

if __name__ == "__main__":
    # Define paths and model name

    parser = argparse.ArgumentParser(description='Evaluate model for Match Result prediction.')
    parser.add_argument('config_name', type=str, help='Name of the YAML config file (without .yaml extension).')
    args = parser.parse_args()

    config_name = args.config_name

    config_base_path = '/root/barnard/ML/Configs/'
    config_path = os.path.join(config_base_path, f'{config_name}.yaml')

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        step = f"Load config file from '{config_path}'"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = f"Load config file from '{config_path}'"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        exit(1)

    # Extract paths from the config
    prediction_dataset = config['predictions_pathways']['Match_Result']
    prediction_model = config['prediction_model_pathway']['Match_Result']
    prediction_metrics_dir = config['prediction_metrics_pathway']['Match_Result']

    # Extract 'league_name_pretty' from the config
    league_name_pretty = config.get('league_name_pretty', 'Unknown League')

    # Define class labels and names for distribution plots
    class_labels = [0, 1, 2]  # 0: Draw, 1: Away Win, 2: Home Win
    class_names = {0: 'Draw', 1: 'Away Win', 2: 'Home Win'}

    # Define mapping from class labels to odds columns
    class_to_odds_column = {
        0: 'odds_draw',
        1: 'odds_opponent_win',
        2: 'odds_team_win'
    }

    # Define confidence thresholds and corresponding stake amounts
    confidence_thresholds = [0.51]  # 51%
    stake_amounts = [5]  # Stakes corresponding to thresholds

    # Load the trained model
    trained_model = load_trained_model(prediction_model)

    # Load and preprocess the new data
    new_data = load_new_data(prediction_dataset)

    # Verify that all required columns are present, including 'match_status' and 'competition_id'
    required_columns = features + [
        'match_id', 'team_name', 'opponent_name', 'winning_team',
        'odds_draw', 'odds_opponent_win', 'odds_team_win', 'match_status', 'competition_id'
    ]
    try:
        verify_columns(new_data, required_columns)
    except KeyError as e:
        print(f"{Fore.RED}Data verification failed: {e}{Style.RESET_ALL}")
        exit(1)

    # Extract 'league_name_pretty' has already been done above

    # Filter data to only include matches where 'match_status' is 'complete'
    new_data = new_data[new_data['match_status'] == 'complete']
    if new_data.empty:
        print(f"{Fore.YELLOW}No matches with 'match_status' == 'complete' found in the data.{Style.RESET_ALL}")
        exit(1)

    preprocessed_new_data = preprocess_new_data(new_data, features)

    # Make predictions
    predictions, prediction_proba = make_predictions(trained_model, preprocessed_new_data)

    # Combine predictions per match
    try:
        df_combined = combine_predictions_per_match(new_data, predictions, prediction_proba)
    except Exception as e:
        print(f"{Fore.RED}Failed to combine predictions per match: {e}{Style.RESET_ALL}")
        exit(1)

    # Compute betting returns with variable staking
    try:
        df_combined, total_return = compute_betting_returns(
            df_combined,
            class_to_odds_column,
            confidence_thresholds=confidence_thresholds,
            stake_amounts=stake_amounts
        )
    except Exception as e:
        print(f"{Fore.RED}Failed to compute betting returns: {e}{Style.RESET_ALL}")
        exit(1)

    # Evaluate model performance on combined match data
    try:
        accuracy = evaluate_model_performance_combined(df_combined)
    except Exception as e:
        print(f"{Fore.RED}Failed to evaluate model performance: {e}{Style.RESET_ALL}")

    # Calculate and print Total Stake and ROI
    try:
        roi = calculate_total_stake_and_roi(df_combined)
    except Exception as e:
        print(f"{Fore.RED}Failed to calculate Total Stake and ROI: {e}{Style.RESET_ALL}")

    # Evaluate and print accuracy for predictions over the 51% threshold
    try:
        threshold_accuracy = evaluate_threshold_accuracy_combined(df_combined, threshold=0.51)
    except Exception as e:
        print(f"{Fore.RED}Failed to evaluate threshold accuracy: {e}{Style.RESET_ALL}")

    # Extract unique competition_ids from the combined data
    try:
        competition_ids = df_combined['competition_id'].unique()
        if len(competition_ids) == 0:
            print(f"{Fore.YELLOW}No competition_id found in the data.{Style.RESET_ALL}")
            exit(1)
    except Exception as e:
        print(f"{Fore.RED}Failed to extract competition_id: {e}{Style.RESET_ALL}")
        exit(1)

    # For each competition_id, filter the data and compute metrics
    metrics_list = []
    for competition_id in competition_ids:
        df_competition = df_combined[df_combined['competition_id'] == competition_id]
        if df_competition.empty:
            print(f"{Fore.YELLOW}No data found for competition_id {competition_id}.{Style.RESET_ALL}")
            continue

        # Compute metrics
        try:
            comp_accuracy = accuracy_score(df_competition['winning_team'], df_competition['Final_Predicted_Winning_Team'])
            print(f"{Fore.GREEN}Competition ID {competition_id} - Base Accuracy: {comp_accuracy*100:.2f}%{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Failed to compute accuracy for competition_id {competition_id}: {e}{Style.RESET_ALL}")
            continue

        try:
            # Calculate threshold accuracy
            def get_predicted_probability(row):
                predicted_class = row['Final_Predicted_Winning_Team']
                if predicted_class == 0:
                    return row['Prob_Draw']
                elif predicted_class == 1:
                    return row['Prob_Away_Win']
                elif predicted_class == 2:
                    return row['Prob_Home_Win']
                else:
                    return 0  # Unknown class

            df_competition['Predicted_Confidence'] = df_competition.apply(get_predicted_probability, axis=1)
            df_threshold = df_competition[df_competition['Predicted_Confidence'] >= 0.51]

            if df_threshold.empty:
                print(f"{Fore.YELLOW}No predictions with confidence >= 51% for competition_id {competition_id}.{Style.RESET_ALL}")
                comp_threshold_accuracy = None
            else:
                correct_predictions = df_threshold['Final_Predicted_Winning_Team'] == df_threshold['winning_team']
                comp_threshold_accuracy = correct_predictions.mean()
                print(f"{Fore.GREEN}Competition ID {competition_id} - Threshold Accuracy: {comp_threshold_accuracy*100:.2f}%{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Failed to compute threshold accuracy for competition_id {competition_id}: {e}{Style.RESET_ALL}")
            comp_threshold_accuracy = None

        try:
            # Calculate ROI
            total_stake = df_competition['Stake_Amount'].sum()
            total_profit_loss = df_competition['Bet_Return'].sum()
            roi = (total_profit_loss / total_stake) * 100 if total_stake != 0 else 0
            print(f"{Fore.GREEN}Competition ID {competition_id} - ROI: {roi:.2f}%{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Failed to calculate ROI for competition_id {competition_id}: {e}{Style.RESET_ALL}")
            roi = 0

        # Append metrics to the list, including 'league_name_pretty'
        metrics = {
            'competition_id': int(competition_id),  # Ensure native int
            'accuracy': float(comp_accuracy),
            'threshold_accuracy': float(comp_threshold_accuracy) if comp_threshold_accuracy is not None else None,
            'roi': float(roi),
            'league_name_pretty': league_name_pretty  # Add this field
        }
        metrics_list.append(metrics)

    # Save the key metrics to MongoDB
    try:
        for metrics in metrics_list:
            competition_id = metrics['competition_id']
            accuracy = metrics['accuracy']
            threshold_accuracy = metrics['threshold_accuracy']
            roi = metrics['roi']
            league_name_pretty_value = metrics['league_name_pretty']
            save_metrics_to_db(
                competition_id=competition_id,
                accuracy=accuracy,
                threshold_accuracy=threshold_accuracy,
                roi=roi,
                league_name_pretty=league_name_pretty_value,  # Pass the new field
                mongo_uri='mongodb://localhost:27017/',    # Update if your MongoDB URI is different
                db_name='footballDB',                      # Update if your DB name is different
                collection_name='leagues'
            )
    except Exception as e:
        print(f"{Fore.RED}Failed to save metrics to MongoDB: {e}{Style.RESET_ALL}")
        exit(1)

    print(f"{Fore.GREEN}All metrics have been successfully saved to the 'leagues' collection in MongoDB.{Style.RESET_ALL}")
