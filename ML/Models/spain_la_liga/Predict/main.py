import pandas as pd
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Define your feature list as used during training (odds are excluded)
features = [
    'team_id', 'opponent_id',
    'team_ELO_before', 'opponent_ELO_before',
    # 'odds_team_win',
    # 'odds_draw', 'odds_opponent_win',
    'opponent_rest_days', 'team_rest_days',
    'is_home',
    'team_h2h_win_percent', 'opponent_h2h_win_percent',
    'team_ppg', 'opponent_ppg',
    'team_ppg_mc', 'opponent_ppg_mc',
    # 'pre_match_home_ppg', 'pre_match_away_ppg',
    'team_home_advantage', 'opponent_home_advantage',
    # 'team_away_advantage', 'opponent_away_advantage'
]

def load_trained_model(model_name, model_dir='/root/barnard/machine-learning/trained-models'):
    """
    Loads a trained model from the specified directory.
    """
    model_filename = os.path.join(model_dir, f"{model_name}.joblib")
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"The model file '{model_filename}' does not exist.")
    try:
        model = joblib.load(model_filename)
        print(f"Model '{model_name}' loaded successfully from '{model_filename}'.")
        return model
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
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
        print(f"Label encoder '{encoder_name}' loaded successfully.")
        return encoder
    except Exception as e:
        print(f"Error loading encoder '{encoder_name}': {e}")
        raise e

def load_new_data(csv_path):
    """
    Loads new data from a CSV file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The CSV file '{csv_path}' does not exist.")
    try:
        df_new = pd.read_csv(csv_path)
        print(f"New data loaded successfully from '{csv_path}'.")
        return df_new
    except Exception as e:
        print(f"Error loading data from '{csv_path}': {e}")
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
        print("Selected required features for prediction.")
    except KeyError as e:
        missing_features = list(set(features) - set(df_new.columns))
        print(f"Missing features in the new data: {missing_features}")
        raise KeyError(f"Missing features: {missing_features}")
    
    # Handle missing values
    X_new = X_new.fillna(0)
    print("Handled missing values in new data.")
    
    # Apply label encoders if provided
    if label_encoders:
        for column, encoder in label_encoders.items():
            try:
                X_new[column] = encoder.transform(X_new[column])
                print(f"Applied encoder to column '{column}'.")
            except Exception as e:
                print(f"Error encoding column '{column}': {e}")
                raise e
    
    return X_new

def make_predictions(model, X_new):
    """
    Makes predictions and computes prediction probabilities.
    """
    try:
        predictions = model.predict(X_new)
        prediction_proba = model.predict_proba(X_new)
        print("Predictions made successfully.")
        return predictions, prediction_proba
    except Exception as e:
        print(f"Error during prediction: {e}")
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
    
    # Group by existing 'match_id' and compute average probabilities and retain odds
    aggregated = df_new.groupby('match_id').agg({
        'Prob_Draw': 'mean',
        'Prob_Away_Win': 'mean',
        'Prob_Home_Win': 'mean',
        'winning_team': 'first',       # Assuming both rows have the same winning_team
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
    
    print("Combined predictions per match by averaging class probabilities.")
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
        print(f"Combined predictions saved successfully to '{output_csv_path}'.")
    except Exception as e:
        print(f"Error saving combined predictions to '{output_csv_path}': {e}")
        raise e

def display_combined_predictions(df_combined, num_samples=5):
    """
    Displays the first few combined predictions.
    """
    try:
        print(df_combined.head(num_samples))
    except Exception as e:
        print(f"Error displaying combined predictions: {e}")
        raise e

def evaluate_model_performance_combined(df_combined, output_metrics_path, model_name):
    """
    Evaluates and saves the performance metrics of the model on combined match data.
    """
    try:
        true_labels = df_combined['winning_team']
        predictions = df_combined['Final_Predicted_Winning_Team']
    except KeyError:
        print("The combined data does not contain the required columns for evaluation.")
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
        print(f"Evaluation metrics saved successfully to '{output_metrics_path}'.")
    except Exception as e:
        print(f"Error saving metrics to '{output_metrics_path}': {e}")
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
        print(f"Confusion matrix plot saved to '{cm_plot_path}'.")
    except Exception as e:
        print(f"Error generating confusion matrix plot: {e}")
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
            print(f"Per-team accuracy saved successfully to '{output_per_team_accuracy_csv}'.")
        except Exception as e:
            print(f"Error saving per-team accuracy to '{output_per_team_accuracy_csv}': {e}")
            raise e
        
        # Optionally, display the top and bottom performing teams
        print("\n--- Per-Team Accuracy (Top 10 Teams) ---")
        print(per_team_accuracy.head(10))
        print("\n--- Per-Team Accuracy (Bottom 10 Teams) ---")
        print(per_team_accuracy.tail(10))
        
        return per_team_accuracy
    except Exception as e:
        print(f"Failed to evaluate per-team accuracy: {e}")
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
        print(f"Per-team accuracy plot saved to '{output_plot_path}'.")
    except Exception as e:
        print(f"Error generating per-team accuracy plot: {e}")
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
        print(f"{title} plot saved to '{plot_path}'.")
    except Exception as e:
        print(f"Error generating {title} plot: {e}")
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
        print(f"Failed to compute and plot class distributions: {e}")
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
                print(f"Unknown predicted class {predicted_class} in match_id {row['match_id']}")
                return 0

            confidence = row[prob_column]
            for threshold, stake in zip(confidence_thresholds, stake_amounts):
                if confidence >= threshold:
                    return stake
            return 1  # Default stake

        # Apply the function to determine stake for each match
        df_combined['Stake_Amount'] = df_combined.apply(determine_stake, axis=1)
        print("Stake amounts determined based on prediction confidence.")

        # Define a function to calculate return per row
        def calculate_return(row):
            predicted_class = row['Final_Predicted_Winning_Team']
            actual_class = row['winning_team']
            odds_column = class_to_odds_column.get(predicted_class)
            if not odds_column:
                print(f"No odds column mapping for class {predicted_class} in match_id {row['match_id']}")
                return 0
            odds = row.get(odds_column, 0)
            if pd.isna(odds):
                print(f"Odds not available for class {predicted_class} in match_id {row['match_id']}")
                return -row['Stake_Amount']  # Lose the stake if odds not available
            if predicted_class == actual_class:
                return (odds * row['Stake_Amount']) - row['Stake_Amount']  # Profit
            else:
                return -row['Stake_Amount']  # Loss

        # Apply the function to calculate bet return
        df_combined['Bet_Return'] = df_combined.apply(calculate_return, axis=1)
        print("Bet returns calculated based on stake amounts and prediction accuracy.")

        # Compute total profit/loss
        total_return = df_combined['Bet_Return'].sum()
        print(f"Total Profit/Loss from variable staking bets: £{total_return:.2f}")

        return df_combined, total_return
    except Exception as e:
        print(f"Failed to compute betting returns: {e}")
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
            print(f"Stake £{stake}: {status} of £{profit_loss:.2f}")

        # Ensure all stake amounts are reported, even if zero profit/loss
        for stake in stake_amounts:
            if stake not in bet_returns['Stake_Amount'].values:
                print(f"Stake £{stake}: Profit/Loss of £0.00")
    except Exception as e:
        print(f"Failed to print bet returns per threshold: {e}")
        raise e

def calculate_total_stake_and_roi(df_combined):
    """
    Calculates and prints the total stake and ROI.

    Parameters:
    - df_combined (pd.DataFrame): DataFrame with combined predictions per match.
    """
    try:
        # Calculate total stake
        total_stake = df_combined['Stake_Amount'].sum()
        # Calculate total profit/loss
        total_profit_loss = df_combined['Bet_Return'].sum()
        # Calculate ROI
        roi = (total_profit_loss / total_stake) * 100 if total_stake != 0 else 0

        print("\n--- Total Stake and ROI ---")
        print(f"Total Stake: £{total_stake:.2f}")
        print(f"Total Profit/Loss: £{total_profit_loss:.2f}")
        print(f"Return on Investment (ROI): {roi:.2f}%")
    except Exception as e:
        print(f"Failed to calculate Total Stake and ROI: {e}")
        raise e

if __name__ == "__main__":
    # Define paths and model name
    
    model_name = 'v6-lucent-mock-rs42'  # Replace with your actual model name
    model_dir = f'/root/barnard/machine-learning/trained-models/la-liga'
    new_data_csv = f'/root/barnard/machine-learning/data-sets/la-liga/predictions.csv'  # Replace with your new data CSV path
    output_predictions_csv = '/root/barnard/machine-learning/models/metrics/la-liga/combined_predictions.csv'  # Replace with desired output path
    output_metrics_json = '/root/barnard/machine-learning/models/metrics/la-liga/evaluation_metrics.json'  # Path to save evaluation metrics
    output_per_team_accuracy_csv = '/root/barnard/machine-learning/models/metrics/la-liga/per_team_accuracy.csv'  # Path to save per-team accuracy
    output_per_team_accuracy_plot = '/root/barnard/machine-learning/models/metrics/la-liga/per_team_accuracy.png'  # Path to save the plot
    
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
    confidence_thresholds = [0.45]  # 45%, else
    stake_amounts = [5]  # Stakes corresponding to thresholds
    
    # Load the trained model
    trained_model = load_trained_model(model_name, model_dir=model_dir)

    # Load any necessary encoders (if used during training)
    # For example, if 'team_id' was label encoded:
    # label_encoder_team = load_label_encoder('label_encoder_team.joblib', model_dir=model_dir)
    # label_encoders = {'team_id': label_encoder_team, 'opponent_id': label_encoder_opponent}
    
    # If no encoders were used, set label_encoders to None
    label_encoders = None  # or {'team_id': label_encoder_team, 'opponent_id': label_encoder_opponent}

    # Load and preprocess the new data
    new_data = load_new_data(new_data_csv)
    
    # Verify that all required columns are present
    # Include odds columns in required_columns to ensure they're loaded
    required_columns = features + [
        'match_id', 'team_name', 'opponent_name', 'winning_team',
        'odds_draw', 'odds_opponent_win', 'odds_team_win'
    ]
    try:
        verify_columns(new_data, required_columns)
    except KeyError as e:
        print(f"Data verification failed: {e}")
        exit(1)
    
    preprocessed_new_data = preprocess_new_data(new_data, features, label_encoders=label_encoders)
    
    # Make predictions
    predictions, prediction_proba = make_predictions(trained_model, preprocessed_new_data)
    
    # Combine predictions per match
    try:
        df_combined = combine_predictions_per_match(new_data, predictions, prediction_proba)
    except Exception as e:
        print(f"Failed to combine predictions per match: {e}")
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
        print(f"Failed to compute betting returns: {e}")
        exit(1)
    
    # Save combined predictions to a CSV file
    try:
        save_combined_predictions(df_combined, output_predictions_csv)
    except Exception as e:
        print(f"Failed to save combined predictions: {e}")
        exit(1)
    
    # Optionally, display the first few combined predictions
    try:
        display_combined_predictions(df_combined, num_samples=5)
    except Exception as e:
        print(f"Failed to display combined predictions: {e}")
    
    # Compute and plot class distributions
    try:
        compute_and_plot_distributions_combined(
            df_combined=df_combined,
            class_labels=class_labels,
            class_names=class_names,
            metrics_dir=os.path.dirname(output_metrics_json)
        )
    except Exception as e:
        print(f"Failed to compute and plot class distributions: {e}")
    
    # Evaluate model performance on combined match data
    try:
        metrics = evaluate_model_performance_combined(df_combined, output_metrics_json, model_name)
    except Exception as e:
        print(f"Failed to evaluate model performance: {e}")
    
    # Evaluate and save per-team accuracy
    try:
        per_team_accuracy = evaluate_per_team_accuracy_combined(df_combined, output_per_team_accuracy_csv)
    except Exception as e:
        print(f"Failed to evaluate per-team accuracy: {e}")
    
    # Plot per-team accuracy
    try:
        plot_per_team_accuracy(per_team_accuracy, output_per_team_accuracy_plot)
    except Exception as e:
        print(f"Failed to plot per-team accuracy: {e}")
    
    # Print profit/loss per stake threshold
    try:
        print_bet_returns_per_threshold(df_combined, stake_amounts)
    except Exception as e:
        print(f"Failed to print bet returns per threshold: {e}")
    
    # Calculate and print Total Stake and ROI
    try:
        calculate_total_stake_and_roi(df_combined)
    except Exception as e:
        print(f"Failed to calculate Total Stake and ROI: {e}")


