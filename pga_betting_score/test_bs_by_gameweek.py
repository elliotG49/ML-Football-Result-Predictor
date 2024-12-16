import argparse
import pandas as pd
from pymongo import MongoClient, errors
import sys
import numpy as np
from datetime import datetime
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt  # For optional plotting

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Verify match predictions against MongoDB and calculate accuracy per calendar week.')
    parser.add_argument('--csv', required=True, help='Path to the CSV file containing predictions.')
    parser.add_argument('--mongo_uri', default='mongodb://localhost:27017/', help='MongoDB connection URI.')
    parser.add_argument('--db', default='footballDB', help='MongoDB database name.')
    parser.add_argument('--collection', default='matches', help='MongoDB collection name.')
    parser.add_argument('--output_csv', help='Path to save the detailed verification results (optional).')
    parser.add_argument('--plot', action='store_true', help='Generate plots for accuracy over weeks.')
    return parser.parse_args()

def connect_to_mongo(uri, db_name, collection_name):
    """
    Establish a connection to MongoDB.

    Args:
        uri (str): MongoDB URI.
        db_name (str): Database name.
        collection_name (str): Collection name.

    Returns:
        tuple: (MongoClient, Collection)
    """
    try:
        client = MongoClient(uri)
        db = client[db_name]
        collection = db[collection_name]
        print(f"Connected to MongoDB database '{db_name}', collection '{collection_name}'.")
        return client, collection
    except errors.PyMongoError as e:
        print(f"Error connecting to MongoDB: {e}")
        sys.exit(1)

def read_csv(csv_path):
    """
    Read the CSV file containing predictions.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: DataFrame containing CSV data.
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully read CSV file: {csv_path}")
        return df
    except FileNotFoundError:
        print(f"CSV file not found: {csv_path}")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        sys.exit(1)

def assign_calendar_week(date_str):
    """
    Assign calendar week and year based on the match date.

    Args:
        date_str (str): Date string from the CSV.

    Returns:
        tuple: (year, week_number)
    """
    try:
        date_obj = pd.to_datetime(date_str)
        iso_calendar = date_obj.isocalendar()
        return (iso_calendar.year, iso_calendar.week)
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
        return (None, None)

def verify_predictions(df, collection):
    """
    Verify predictions against MongoDB matches and calculate accuracy per calendar week.

    Args:
        df (pandas.DataFrame): DataFrame containing predictions.
        collection (pymongo.collection.Collection): MongoDB collection object.

    Returns:
        tuple: (total_predictions, correct_predictions, week_stats, detailed_results, y_true, y_pred)
    """
    total_predictions = 0
    correct_predictions = 0
    week_correct = {}
    week_total = {}
    detailed_results = []
    
    # For recall score calculation
    y_true = []
    y_pred = []
    
    # Iterate through each prediction
    for index, row in df.iterrows():
        match_id = row['matchID']
        predicted_team = row['predictedTeam']
        date_str = row['date']  # Assuming 'date' column exists and is in a parseable format

        # Assign calendar week and year
        year, week_number = assign_calendar_week(date_str)
        if year is None or week_number is None:
            print(f"Skipping matchID '{match_id}' due to invalid date.")
            continue

        # Query MongoDB for the match with the given _id and status 'complete'
        try:
            match = collection.find_one({'_id': match_id, 'status': 'complete'})
        except errors.PyMongoError as e:
            print(f"Error querying MongoDB for matchID '{match_id}': {e}")
            continue

        if not match:
            print(f"MatchID '{match_id}' not found or not complete. Skipping.")
            continue

        winning_team = match.get('winningTeam')

        if winning_team is None:
            print(f"MatchID '{match_id}' has no 'winningTeam' information. Skipping.")
            continue

        # Update total predictions
        total_predictions += 1

        # Initialize week in dictionaries if not present
        week_key = f"{year}-W{week_number}"
        if week_key not in week_total:
            week_total[week_key] = 0
            week_correct[week_key] = 0

        week_total[week_key] += 1

        # Check if prediction is correct
        is_correct = predicted_team == winning_team
        if is_correct:
            correct_predictions += 1
            week_correct[week_key] += 1

        # For recall score
        # Define classes: 0 = Home Win, 1 = Draw, 2 = Away Win
        if predicted_team == match.get('homeID'):
            pred_class = 0
        elif predicted_team == -1:
            pred_class = 1
        elif predicted_team == match.get('awayID'):
            pred_class = 2
        else:
            # Unknown class, skip
            print(f"Invalid predicted_team '{predicted_team}' for matchID '{match_id}'. Skipping.")
            continue

        if winning_team == match.get('homeID'):
            true_class = 0
        elif winning_team == -1:
            true_class = 1
        elif winning_team == match.get('awayID'):
            true_class = 2
        else:
            # Unknown class, skip
            print(f"Invalid winning_team '{winning_team}' for matchID '{match_id}'. Skipping.")
            continue

        y_true.append(true_class)
        y_pred.append(pred_class)

        # Append detailed result
        detailed_results.append({
            'matchID': match_id,
            'predictedTeam': predicted_team,
            'winningTeam': winning_team,
            'is_correct': is_correct,
            'calendar_week': week_key,
            'date': date_str
        })

    # Calculate calendar week statistics
    week_stats = []
    for week in sorted(week_total.keys()):
        total = week_total[week]
        correct = week_correct[week]
        accuracy = (correct / total) * 100 if total > 0 else np.nan  # Use NaN for undefined accuracy
        week_stats.append({
            'calendar_week': week,
            'total_predictions': total,
            'correct_predictions': correct,
            'accuracy_percentage': round(accuracy, 2) if not np.isnan(accuracy) else 'N/A'
        })

    return total_predictions, correct_predictions, week_stats, detailed_results, y_true, y_pred

def calculate_recall(y_true, y_pred):
    """
    Calculate recall score for each class.

    Args:
        y_true (list): True class labels.
        y_pred (list): Predicted class labels.

    Returns:
        dict: Recall scores for each class.
    """
    try:
        recall = recall_score(y_true, y_pred, labels=[0,1,2], average=None, zero_division=0)
        recall_dict = {
            'Home Win': recall[0],
            'Draw': recall[1],
            'Away Win': recall[2]
        }
        return recall_dict
    except Exception as e:
        print(f"Error calculating recall score: {e}")
        return {}

def save_detailed_results(detailed_results, output_path):
    """
    Save detailed verification results to a CSV file.

    Args:
        detailed_results (list of dict): Detailed verification data.
        output_path (str): Path to save the output CSV.
    """
    if not detailed_results:
        print("No detailed results to save.")
        return

    df_details = pd.DataFrame(detailed_results)
    try:
        df_details.to_csv(output_path, index=False)
        print(f"Detailed verification results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving detailed results to CSV: {e}")

def plot_accuracy_over_weeks(week_stats):
    """
    Plot accuracy over calendar weeks.

    Args:
        week_stats (list of dict): List containing statistics per week.
    """
    # Convert to DataFrame for plotting
    df_plot = pd.DataFrame(week_stats)
    df_plot = df_plot[df_plot['accuracy_percentage'] != 'N/A']  # Remove weeks with no data

    # Sort by calendar_week
    df_plot['year'] = df_plot['calendar_week'].apply(lambda x: int(x.split('-W')[0]))
    df_plot['week'] = df_plot['calendar_week'].apply(lambda x: int(x.split('-W')[1]))
    df_plot = df_plot.sort_values(['year', 'week'])

    # Create a combined week label
    df_plot['week_label'] = df_plot.apply(lambda row: f"{row['year']}-W{row['week']:02d}", axis=1)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_plot['week_label'], df_plot['accuracy_percentage'], marker='o', linestyle='-')
    plt.xticks(rotation=90)
    plt.xlabel('Calendar Week')
    plt.ylabel('Accuracy (%)')
    plt.title('Prediction Accuracy Over Calendar Weeks')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Read CSV file
    df_predictions = read_csv(args.csv)

    # Connect to MongoDB
    client, matches_collection = connect_to_mongo(args.mongo_uri, args.db, args.collection)

    # Verify predictions and calculate accuracy per calendar week
    total, correct, week_stats, details, y_true, y_pred = verify_predictions(df_predictions, matches_collection)

    # Close MongoDB connection
    client.close()
    print("Disconnected from MongoDB.")

    # Calculate overall accuracy
    if total == 0:
        print("No predictions were verified.")
    else:
        overall_accuracy = (correct / total) * 100
        print(f"\n--- Overall Accuracy ---")
        print(f"Total Predictions Verified: {total}")
        print(f"Correct Predictions: {correct}")
        print(f"Accuracy: {overall_accuracy:.2f}%")

        # Calculate recall scores
        recall_scores = calculate_recall(y_true, y_pred)
        print(f"\n--- Recall Scores ---")
        for category, score in recall_scores.items():
            print(f"{category}: {score:.2f}")

        # Print accuracy per calendar week
        print(f"\n--- Accuracy per Calendar Week ---")
        header = f"{'Calendar Week':<15} {'Total Predictions':<20} {'Correct Predictions':<20} {'Accuracy (%)':<15}"
        print(header)
        print('-' * len(header))
        for stat in week_stats:
            accuracy_display = f"{stat['accuracy_percentage']}%" if stat['accuracy_percentage'] != 'N/A' else "N/A"
            print(f"{stat['calendar_week']:<15} {stat['total_predictions']:<20} {stat['correct_predictions']:<20} {accuracy_display:<15}")

        # Optional: Plot accuracy over weeks
        if args.plot:
            plot_accuracy_over_weeks(week_stats)

    # Save detailed results if output_csv is provided
    if args.output_csv:
        save_detailed_results(details, args.output_csv)

if __name__ == "__main__":
    main()
