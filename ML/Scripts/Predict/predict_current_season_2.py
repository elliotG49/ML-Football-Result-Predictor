import pandas as pd
import joblib
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
    # 'odds_team_win',
    # 'odds_draw', 
    # 'odds_opponent_win',
    'opponent_rest_days', 'team_rest_days',
    'is_home',
    'team_h2h_win_percent', 'opponent_h2h_win_percent',
    'team_ppg', 'opponent_ppg',
    'team_ppg_mc', 'opponent_ppg_mc',
    # 'pre_match_home_ppg', 'pre_match_away_ppg',
    'team_home_advantage', 'opponent_home_advantage',
    # 'team_away_advantage', 'opponent_away_advantage'
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
    Loads a trained model from the specified directory.
    """
    # Check if the model_name already ends with '.joblib'
    if not model_name.endswith('.joblib'):
        model_filename = os.path.join(model_dir, f"{model_name}.joblib")
    else:
        model_filename = os.path.join(model_dir, model_name)
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"The model file '{model_filename}' does not exist.")
    try:
        model = joblib.load(model_filename)
        print(f"{Fore.GREEN}Model '{model_name}' loaded successfully from '{model_filename}'.{Style.RESET_ALL}")
        return model
    except Exception as e:
        print(f"{Fore.RED}Error loading model '{model_name}': {e}{Style.RESET_ALL}")
        raise e

def load_label_encoder(encoder_name, model_dir='/root/barnard/machine-learning/trained-models'):
    """
    Loads a label encoder from the specified directory.
    """
    encoder_path = os.path.join(model_dir, encoder_name)
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"The encoder file '{encoder_path}' does not exist.")
    try:
        encoder = joblib.load(encoder_path)
        print(f"{Fore.GREEN}Label encoder '{encoder_name}' loaded successfully.{Style.RESET_ALL}")
        return encoder
    except Exception as e:
        print(f"{Fore.RED}Error loading encoder '{encoder_name}': {e}{Style.RESET_ALL}")
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

def preprocess_new_data(df_new, features, label_encoders=None, random_state=None):
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
    
    # Apply label encoders if provided
    if label_encoders:
        for column, encoder in label_encoders.items():
            try:
                X_new[column] = encoder.transform(X_new[column])
                print(f"{Fore.GREEN}Applied encoder to column '{column}'.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error encoding column '{column}': {e}{Style.RESET_ALL}")
                raise e
    
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
    
    Parameters:
    - df_new (pd.DataFrame): Original DataFrame with match data.
    - predictions (np.array): Array of predictions per row.
    - prediction_proba (np.array): Array of prediction probabilities per row.
    
    Returns:
    - pd.DataFrame: DataFrame with one row per match, including averaged probabilities, odds, and final prediction.
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
        'odds_team_win': 'first'
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

def save_combined_predictions(df_combined, output_csv_path):
    """
    Saves the combined predictions to a CSV file.
    
    Parameters:
    - df_combined (pd.DataFrame): DataFrame with combined predictions per match.
    - output_csv_path (str): Path to save the combined predictions CSV.
    """
    try:
        # Define the columns to include: relevant features + 'team_name' + 'opponent_name' + final prediction + odds + betting return
        columns_to_include = [
            'match_id', 'team_name', 'opponent_name', 'Final_Predicted_Winning_Team',
            'Prob_Draw', 'Prob_Away_Win', 'Prob_Home_Win',
            'odds_draw', 'odds_opponent_win', 'odds_team_win',
            'winning_team', 'Stake_Amount', 'Bet_Return'  # Include 'Stake_Amount' and 'Bet_Return'
        ]

        # Check if 'Stake_Amount' and 'Bet_Return' exist, else exclude them
        if 'Stake_Amount' not in df_combined.columns:
            columns_to_include.remove('Stake_Amount')
        if 'Bet_Return' not in df_combined.columns:
            columns_to_include.remove('Bet_Return')

        # Ensure that all required columns are present
        missing_cols = set(columns_to_include) - set(df_combined.columns)
        if missing_cols:
            raise KeyError(f"The following required columns are missing in the combined data: {missing_cols}")

        # Select only the relevant columns
        df_output = df_combined[columns_to_include].copy()

        # Save to CSV
        df_output.to_csv(output_csv_path, index=False)
        print(f"{Fore.GREEN}Combined predictions saved successfully to '{output_csv_path}'.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error saving combined predictions to '{output_csv_path}': {e}{Style.RESET_ALL}")
        raise e

def save_detailed_bet_returns(df_combined, output_detailed_csv_path):
    """
    Saves detailed bet returns per match to a CSV file.
    
    Parameters:
    - df_combined (pd.DataFrame): DataFrame with combined predictions per match.
    - output_detailed_csv_path (str): Path to save the detailed bet returns CSV.
    """
    try:
        # Define the columns to include
        detailed_columns = [
            'match_id', 'team_name', 'opponent_name',
            'Final_Predicted_Winning_Team', 'winning_team',
            'Prob_Draw', 'Prob_Away_Win', 'Prob_Home_Win',
            'odds_draw', 'odds_opponent_win', 'odds_team_win',
            'Stake_Amount', 'Bet_Return'
        ]

        # Check for missing columns
        missing = set(detailed_columns) - set(df_combined.columns)
        if missing:
            raise KeyError(f"Missing columns for detailed bet returns: {missing}")

        # Extract the necessary columns
        df_detailed = df_combined[detailed_columns].copy()

        # Optional: Add 'ROI_Per_Match' if Stake_Amount > 0
        df_detailed['ROI_Per_Match'] = np.where(
            df_detailed['Stake_Amount'] > 0,
            (df_detailed['Bet_Return'] / df_detailed['Stake_Amount']) * 100,
            0
        )

        # Save to CSV
        df_detailed.to_csv(output_detailed_csv_path, index=False)
        print(f"{Fore.GREEN}Detailed bet returns saved successfully to '{output_detailed_csv_path}'.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error saving detailed bet returns to '{output_detailed_csv_path}': {e}{Style.RESET_ALL}")
        raise e

def display_combined_predictions(df_combined, num_samples=5):
    """
    Displays the first few combined predictions.
    """
    try:
        print(df_combined.head(num_samples))
    except Exception as e:
        print(f"{Fore.RED}Error displaying combined predictions: {e}{Style.RESET_ALL}")
        raise e

def evaluate_model_performance_combined(df_combined, output_metrics_path, model_name):
    """
    Evaluates and saves the performance metrics of the model on combined match data.
    """
    try:
        true_labels = df_combined['winning_team']
        predictions = df_combined['Final_Predicted_Winning_Team']
    except KeyError:
        print(f"{Fore.RED}The combined data does not contain the required columns for evaluation.{Style.RESET_ALL}")
        raise KeyError("Missing required columns in combined data.")
    
    # Compute evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
    class_report = classification_report(true_labels, predictions, zero_division=0, output_dict=True)
    cm = confusion_matrix(true_labels, predictions)
    
    # Compile metrics
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_score_macro": f1,
        "classification_report": class_report,
        "confusion_matrix": cm.tolist()  # Convert to list for JSON serialization
    }
    
    # Save metrics to JSON
    try:
        with open(output_metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"{Fore.GREEN}Evaluation metrics saved successfully to '{output_metrics_path}'.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error saving metrics to '{output_metrics_path}': {e}{Style.RESET_ALL}")
        raise e
    
    # Display metrics
    print("\n--- Model Evaluation Metrics ---")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision (Macro): {precision:.2f}")
    print(f"Recall (Macro): {recall:.2f}")
    print(f"F1-Score (Macro): {f1:.2f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, zero_division=0))
    
    # Plot and save confusion matrix
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Draw', 'Away Win', 'Home Win'], 
                    yticklabels=['Draw', 'Away Win', 'Home Win'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        cm_plot_path = os.path.join(os.path.dirname(output_metrics_path), 'confusion_matrix.png')
        plt.savefig(cm_plot_path)
        plt.close()
        print(f"{Fore.GREEN}Confusion matrix plot saved to '{cm_plot_path}'.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error generating confusion matrix plot: {e}{Style.RESET_ALL}")
        raise e
    
    return metrics

def evaluate_per_team_accuracy_combined(df_combined, output_per_team_accuracy_csv):
    """
    Evaluates and saves the accuracy metrics for each team based on combined match predictions.
    
    Parameters:
    - df_combined (pd.DataFrame): DataFrame with combined predictions per match.
    - output_per_team_accuracy_csv (str): Path to save per-team accuracy CSV.
    """
    try:
        # Check required columns
        required_columns = ['team_name', 'Final_Predicted_Winning_Team', 'winning_team']
        missing_columns = set(required_columns) - set(df_combined.columns)
        if missing_columns:
            raise KeyError(f"The following required columns are missing in the combined data: {missing_columns}")
        
        # Calculate correctness
        df_combined['Correct_Prediction'] = df_combined['Final_Predicted_Winning_Team'] == df_combined['winning_team']
        
        # Group by team_name
        per_team_group = df_combined.groupby('team_name')
        
        # Calculate accuracy per team
        per_team_accuracy = per_team_group['Correct_Prediction'].mean().reset_index()
        per_team_accuracy.rename(columns={'Correct_Prediction': 'Accuracy'}, inplace=True)
        
        # Convert Accuracy to percentage
        per_team_accuracy['Accuracy'] = per_team_accuracy['Accuracy'] * 100
        
        # Sort by Accuracy descending
        per_team_accuracy.sort_values(by='Accuracy', ascending=False, inplace=True)
        
        # Save to CSV
        try:
            per_team_accuracy.to_csv(output_per_team_accuracy_csv, index=False)
            print(f"{Fore.GREEN}Per-team accuracy saved successfully to '{output_per_team_accuracy_csv}'.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error saving per-team accuracy to '{output_per_team_accuracy_csv}': {e}{Style.RESET_ALL}")
            raise e
        
        # Optionally, display the top and bottom performing teams
        print("\n--- Per-Team Accuracy (Top 10 Teams) ---")
        print(per_team_accuracy.head(10))
        print("\n--- Per-Team Accuracy (Bottom 10 Teams) ---")
        print(per_team_accuracy.tail(10))
        
        return per_team_accuracy
    except Exception as e:
        print(f"{Fore.RED}Failed to evaluate per-team accuracy: {e}{Style.RESET_ALL}")
        raise e

def plot_per_team_accuracy(per_team_accuracy_df, output_plot_path):
    """
    Plots and saves a bar chart of per-team accuracy.
    
    Parameters:
    - per_team_accuracy_df (pd.DataFrame): DataFrame with per-team accuracy.
    - output_plot_path (str): Path to save the plot image.
    """
    try:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Accuracy', y='team_name', data=per_team_accuracy_df, palette='viridis')
        plt.xlabel('Accuracy (%)')
        plt.ylabel('Team Name')
        plt.title('Per-Team Prediction Accuracy')
        plt.tight_layout()
        plt.savefig(output_plot_path)
        plt.close()
        print(f"{Fore.GREEN}Per-team accuracy plot saved to '{output_plot_path}'.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error generating per-team accuracy plot: {e}{Style.RESET_ALL}")
        raise e

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

def compute_betting_returns(df_combined, class_to_odds_column, confidence_thresholds, stake_amounts):
    """
    Computes the return/loss for betting based on the model's predicted outcome and confidence thresholds.

    Parameters:
    - df_combined (pd.DataFrame): DataFrame with combined predictions per match.
    - class_to_odds_column (dict): Mapping from class labels to odds column names.
    - confidence_thresholds (list of floats): List of confidence thresholds in descending order.
    - stake_amounts (list of floats): Corresponding stake amounts for each threshold.

    Returns:
    - pd.DataFrame: DataFrame with added 'Stake_Amount' and 'Bet_Return' columns.
    - float: Total profit/loss from all bets.
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
                print(f"{Fore.RED}Unknown predicted class {predicted_class} in match_id {row['match_id']}.{Style.RESET_ALL}")
                return 0

            confidence = row[prob_column]
            # Iterate through thresholds to determine stake
            for threshold, stake in zip(confidence_thresholds, stake_amounts):
                if confidence >= threshold:
                    return stake
            return 0  # No bet if confidence is below all thresholds

        # Apply the function to determine stake for each match
        df_combined['Stake_Amount'] = df_combined.apply(determine_stake, axis=1)
        print(f"{Fore.GREEN}Stake amounts determined based on prediction confidence.{Style.RESET_ALL}")

        # Define a function to calculate return per row
        def calculate_return(row):
            predicted_class = row['Final_Predicted_Winning_Team']
            actual_class = row['winning_team']
            odds_column = class_to_odds_column.get(predicted_class)
            if not odds_column:
                print(f"{Fore.RED}No odds column mapping for class {predicted_class} in match_id {row['match_id']}.{Style.RESET_ALL}")
                return 0
            odds = row.get(odds_column, 0)
            if pd.isna(odds):
                print(f"{Fore.RED}Odds not available for class {predicted_class} in match_id {row['match_id']}.{Style.RESET_ALL}")
                return -row['Stake_Amount']  # Lose the stake if odds not available
            if predicted_class == actual_class:
                return (odds * row['Stake_Amount']) - row['Stake_Amount']  # Profit
            else:
                return -row['Stake_Amount']  # Loss

        # Apply the function to calculate bet return
        df_combined['Bet_Return'] = df_combined.apply(calculate_return, axis=1)
        print(f"{Fore.GREEN}Bet returns calculated based on stake amounts and prediction accuracy.{Style.RESET_ALL}")

        # Compute total profit/loss
        total_return = df_combined['Bet_Return'].sum()
        print(f"{Fore.GREEN}Total Profit/Loss from variable staking bets: £{total_return:.2f}{Style.RESET_ALL}")

        return df_combined, total_return
    except Exception as e:
        print(f"{Fore.RED}Failed to compute betting returns: {e}{Style.RESET_ALL}")
        raise e

def print_bet_returns_per_threshold(df_combined, stake_amounts):
    """
    Prints the total profit/loss per stake amount.

    Parameters:
    - df_combined (pd.DataFrame): DataFrame with combined predictions per match.
    - stake_amounts (list of floats): List of stake amounts corresponding to thresholds.
    """
    try:
        # Group by 'Stake_Amount' and sum 'Bet_Return'
        bet_returns = df_combined.groupby('Stake_Amount')['Bet_Return'].sum().reset_index()

        print("\n--- Profit/Loss per Stake Amount (£) ---")
        for _, row in bet_returns.iterrows():
            stake = row['Stake_Amount']
            profit_loss = row['Bet_Return']
            status = "Profit" if profit_loss > 0 else "Loss"
            print(f"{Fore.BLUE}Stake £{stake}: {status} of £{profit_loss:.2f}{Style.RESET_ALL}")

        # Ensure all stake amounts are reported, even if zero profit/loss
        for stake in stake_amounts:
            if stake not in bet_returns['Stake_Amount'].values:
                print(f"{Fore.BLUE}Stake £{stake}: Profit/Loss of £0.00{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Failed to print bet returns per threshold: {e}{Style.RESET_ALL}")
        raise e

def calculate_total_stake_and_roi(df_combined):
    """
    Calculates and prints the total stake and ROI.

    Returns:
        dict: A dictionary containing total_stake, total_profit_loss, and roi.
    """
    try:
        # Calculate total stake
        total_stake = df_combined['Stake_Amount'].sum()
        # Calculate total profit/loss
        total_profit_loss = df_combined['Bet_Return'].sum()
        # Calculate ROI
        roi = (total_profit_loss / total_stake) * 100 if total_stake != 0 else 0

        print(f"\n{Fore.YELLOW}--- Total Stake and ROI ---{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Total Stake: £{total_stake:.2f}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Total Profit/Loss: £{total_profit_loss:.2f}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Return on Investment (ROI): {roi:.2f}%{Style.RESET_ALL}")

        # Return the metrics
        return {
            'total_stake': total_stake,
            'total_profit_loss': total_profit_loss,
            'roi': roi
        }
    except Exception as e:
        print(f"{Fore.RED}Failed to calculate Total Stake and ROI: {e}{Style.RESET_ALL}")
        raise e

def evaluate_threshold_accuracy_combined(df_combined, threshold=0.45):
    """
    Evaluates and prints the accuracy of predictions that have a confidence above the specified threshold.

    Returns:
        dict: A dictionary containing threshold, number_of_predictions, correct_predictions, and accuracy.
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
            print(f"\n{Fore.YELLOW}No predictions with confidence >= {threshold*100:.0f}%.{Style.RESET_ALL}")
            return {
                'threshold': threshold,
                'number_of_predictions': 0,
                'correct_predictions': 0,
                'threshold_accuracy': None
            }

        # Calculate accuracy
        correct_predictions = df_threshold['Final_Predicted_Winning_Team'] == df_threshold['winning_team']
        accuracy = correct_predictions.mean()

        print(f"\n{Fore.MAGENTA}--- Accuracy for Predictions with Confidence >= {threshold*100:.0f}% ---{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Number of Predictions: {len(df_threshold)}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Correct Predictions: {correct_predictions.sum()}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Accuracy: {accuracy*100:.2f}%{Style.RESET_ALL}")

        # Return the metrics
        return {
            'threshold': threshold,
            'number_of_predictions': len(df_threshold),
            'correct_predictions': correct_predictions.sum(),
            'threshold_accuracy': accuracy
        }
    except Exception as e:
        print(f"{Fore.RED}Failed to evaluate threshold accuracy: {e}{Style.RESET_ALL}")
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

def log_high_return_matches(df_combined, roi_threshold=100, log_csv_path=None):
    """
    Logs matches that contribute to a high ROI for further inspection.
    
    Parameters:
    - df_combined (pd.DataFrame): DataFrame with combined predictions per match.
    - roi_threshold (float): Threshold ROI to flag high-return matches.
    - log_csv_path (str): Path to save the high-return matches CSV.
    """
    try:
        # Calculate ROI per match
        df_combined['ROI_Per_Match'] = np.where(
            df_combined['Stake_Amount'] > 0,
            (df_combined['Bet_Return'] / df_combined['Stake_Amount']) * 100,
            0
        )

        # Filter matches with ROI >= threshold
        high_roi_matches = df_combined[df_combined['ROI_Per_Match'] >= roi_threshold]

        if high_roi_matches.empty:
            print(f"{Fore.YELLOW}No matches with ROI >= {roi_threshold}% found.{Style.RESET_ALL}")
            return

        # Define columns to log
        log_columns = [
            'match_id', 'team_name', 'opponent_name',
            'Final_Predicted_Winning_Team', 'winning_team',
            'Prob_Draw', 'Prob_Away_Win', 'Prob_Home_Win',
            'odds_draw', 'odds_opponent_win', 'odds_team_win',
            'Stake_Amount', 'Bet_Return', 'ROI_Per_Match'
        ]

        # Ensure all columns are present
        missing = set(log_columns) - set(high_roi_matches.columns)
        if missing:
            raise KeyError(f"Missing columns for logging high ROI matches: {missing}")

        # Extract and save
        df_high_roi = high_roi_matches[log_columns].copy()
        if log_csv_path:
            df_high_roi.to_csv(log_csv_path, index=False)
            print(f"{Fore.GREEN}High ROI matches logged to '{log_csv_path}'.{Style.RESET_ALL}")
        else:
            print("\n--- High ROI Matches ---")
            print(df_high_roi)
    except Exception as e:
        print(f"{Fore.RED}Failed to log high ROI matches: {e}{Style.RESET_ALL}")
        raise e

def save_detailed_bet_returns(df_combined, output_detailed_csv_path):
    """
    Saves detailed bet returns per match to a CSV file.
    
    Parameters:
    - df_combined (pd.DataFrame): DataFrame with combined predictions per match.
    - output_detailed_csv_path (str): Path to save the detailed bet returns CSV.
    """
    try:
        # Define the columns to include
        detailed_columns = [
            'match_id', 'team_name', 'opponent_name',
            'Final_Predicted_Winning_Team', 'winning_team',
            'Prob_Draw', 'Prob_Away_Win', 'Prob_Home_Win',
            'odds_draw', 'odds_opponent_win', 'odds_team_win',
            'Stake_Amount', 'Bet_Return'
        ]

        # Check for missing columns
        missing = set(detailed_columns) - set(df_combined.columns)
        if missing:
            raise KeyError(f"Missing columns for detailed bet returns: {missing}")

        # Extract the necessary columns
        df_detailed = df_combined[detailed_columns].copy()

        # Optional: Add 'ROI_Per_Match' if Stake_Amount > 0
        df_detailed['ROI_Per_Match'] = np.where(
            df_detailed['Stake_Amount'] > 0,
            (df_detailed['Bet_Return'] / df_detailed['Stake_Amount']) * 100,
            0
        )

        # Save to CSV
        df_detailed.to_csv(output_detailed_csv_path, index=False)
        print(f"{Fore.GREEN}Detailed bet returns saved successfully to '{output_detailed_csv_path}'.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error saving detailed bet returns to '{output_detailed_csv_path}': {e}{Style.RESET_ALL}")
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

    # Define output paths
    output_predictions_csv = os.path.join(prediction_metrics_dir, 'combined_predictions.csv')
    output_detailed_bet_returns_csv = os.path.join(prediction_metrics_dir, 'bet_returns_per_match.csv')  # New CSV
    output_metrics_json = os.path.join(prediction_metrics_dir, 'evaluation_metrics.json')
    output_per_team_accuracy_csv = os.path.join(prediction_metrics_dir, 'per_team_accuracy.csv')
    output_per_team_accuracy_plot = os.path.join(prediction_metrics_dir, 'per_team_accuracy.png')
    output_high_roi_csv = os.path.join(prediction_metrics_dir, 'high_roi_matches.csv')  # For high ROI matches

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
    confidence_thresholds = [0.45]  # 45%
    stake_amounts = [5]  # Stakes corresponding to thresholds

    # Load the trained model
    trained_model = load_trained_model(prediction_model)

    # Load any necessary encoders (if used during training)
    label_encoders = None  # Update if you have label encoders

    # Load and preprocess the new data
    new_data = load_new_data(prediction_dataset)

    # Verify that all required columns are present, including 'match_status'
    required_columns = features + [
        'match_id', 'team_name', 'opponent_name', 'winning_team',
        'odds_draw', 'odds_opponent_win', 'odds_team_win', 'match_status'
    ]
    try:
        verify_columns(new_data, required_columns)
    except KeyError as e:
        print(f"{Fore.RED}Data verification failed: {e}{Style.RESET_ALL}")
        exit(1)

    # Filter data to only include matches where 'match_status' is 'complete'
    new_data = new_data[new_data['match_status'] == 'complete']
    if new_data.empty:
        print(f"{Fore.YELLOW}No matches with 'match_status' == 'complete' found in the data.{Style.RESET_ALL}")
        exit(1)

    preprocessed_new_data = preprocess_new_data(new_data, features, label_encoders=label_encoders)

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

    # Save combined predictions to a CSV file
    try:
        save_combined_predictions(df_combined, output_predictions_csv)
    except Exception as e:
        print(f"{Fore.RED}Failed to save combined predictions: {e}{Style.RESET_ALL}")
        exit(1)

    # Save detailed bet returns to a separate CSV for debugging
    try:
        save_detailed_bet_returns(df_combined, output_detailed_bet_returns_csv)
    except Exception as e:
        print(f"{Fore.RED}Failed to save detailed bet returns: {e}{Style.RESET_ALL}")
        exit(1)

    # Log high ROI matches for further inspection
    try:
        log_high_return_matches(
            df_combined,
            roi_threshold=100,  # Adjust threshold as needed
            log_csv_path=output_high_roi_csv
        )
    except Exception as e:
        print(f"{Fore.RED}Failed to log high ROI matches: {e}{Style.RESET_ALL}")
        exit(1)

    # Optionally, display the first few combined predictions
    try:
        display_combined_predictions(df_combined, num_samples=5)
    except Exception as e:
        print(f"{Fore.RED}Failed to display combined predictions: {e}{Style.RESET_ALL}")

    # Compute and plot class distributions
    try:
        compute_and_plot_distributions_combined(
            df_combined=df_combined,
            class_labels=class_labels,
            class_names=class_names,
            metrics_dir=os.path.dirname(output_metrics_json)
        )
    except Exception as e:
        print(f"{Fore.RED}Failed to compute and plot class distributions: {e}{Style.RESET_ALL}")

    # Initialize metrics dictionary
    metrics = {}

    # Evaluate model performance on combined match data
    try:
        performance_metrics = evaluate_model_performance_combined(df_combined, output_metrics_json, prediction_model)
        metrics.update(performance_metrics)
    except Exception as e:
        print(f"{Fore.RED}Failed to evaluate model performance: {e}{Style.RESET_ALL}")

    # Evaluate and save per-team accuracy
    try:
        per_team_accuracy = evaluate_per_team_accuracy_combined(df_combined, output_per_team_accuracy_csv)
    except Exception as e:
        print(f"{Fore.RED}Failed to evaluate per-team accuracy: {e}{Style.RESET_ALL}")

    # Plot per-team accuracy
    try:
        plot_per_team_accuracy(per_team_accuracy, output_per_team_accuracy_plot)
    except Exception as e:
        print(f"{Fore.RED}Failed to plot per-team accuracy: {e}{Style.RESET_ALL}")

    # Print profit/loss per stake threshold
    try:
        print_bet_returns_per_threshold(df_combined, stake_amounts)
    except Exception as e:
        print(f"{Fore.RED}Failed to print bet returns per threshold: {e}{Style.RESET_ALL}")

    # Calculate and print Total Stake and ROI
    try:
        roi_metrics = calculate_total_stake_and_roi(df_combined)
        # Add ROI metrics to metrics dictionary
        metrics.update(roi_metrics)
    except Exception as e:
        print(f"{Fore.RED}Failed to calculate Total Stake and ROI: {e}{Style.RESET_ALL}")

    # Evaluate and print accuracy for predictions over the 45% threshold
    try:
        threshold_metrics = evaluate_threshold_accuracy_combined(df_combined, threshold=0.45)
        # Add threshold metrics to metrics dictionary
        metrics.update(threshold_metrics)
    except Exception as e:
        print(f"{Fore.RED}Failed to evaluate threshold accuracy: {e}{Style.RESET_ALL}")

    # Save updated metrics to JSON
    try:
        # Convert numpy types to native Python types
        metrics_serializable = convert_numpy_types(metrics)
        with open(output_metrics_json, 'w') as f:
            json.dump(metrics_serializable, f, indent=4)
        print(f"{Fore.GREEN}Updated evaluation metrics saved successfully to '{output_metrics_json}'.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error saving updated metrics to '{output_metrics_json}': {e}{Style.RESET_ALL}")
        raise e
