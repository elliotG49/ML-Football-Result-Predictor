import argparse
import pandas as pd
from pymongo import MongoClient, errors
import sys
import numpy as np

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Verify match predictions against MongoDB and calculate accuracy per betting score interval.')
    parser.add_argument('--csv', required=True, help='Path to the CSV file containing predictions.')
    parser.add_argument('--mongo_uri', default='mongodb://localhost:27017/', help='MongoDB connection URI.')
    parser.add_argument('--db', default='footballDB', help='MongoDB database name.')
    parser.add_argument('--collection', default='matches', help='MongoDB collection name.')
    parser.add_argument('--output_csv', help='Path to save the detailed verification results (optional).')
    return parser.parse_args()

def connect_to_mongo(uri, db_name, collection_name):
    """
    Establish a connection to MongoDB.

    Args:
        uri (str): MongoDB URI.
        db_name (str): Database name.
        collection_name (str): Collection name.

    Returns:
        pymongo.collection.Collection: MongoDB collection object.
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

def assign_betting_score_interval(betting_score):
    """
    Assign betting score to a defined interval of 5.

    Args:
        betting_score (float): The betting score.

    Returns:
        str: The interval label.
    """
    if pd.isnull(betting_score):
        return 'Unknown'

    # Ensure betting_score is a float and within 0-100
    try:
        score = float(betting_score)
        if score < 0 or score > 100:
            return 'Unknown'
    except (ValueError, TypeError):
        return 'Unknown'

    # Define bins from 0 to 100 in steps of 5
    bins = list(range(0, 105, 5))  # 0-5, 5-10, ..., 100+
    labels = [f"{i}-{i+5}" for i in bins[:-1]]
    labels[-1] = "100+"  # Adjust the last label to '100+'

    # Use pd.cut to assign intervals
    interval = pd.cut([score], bins=bins, labels=labels, right=False)
    return interval[0] if not pd.isna(interval[0]) else 'Unknown'

def verify_predictions(df, collection):
    """
    Verify predictions against MongoDB matches and calculate accuracy per betting score interval.

    Args:
        df (pandas.DataFrame): DataFrame containing predictions.
        collection (pymongo.collection.Collection): MongoDB collection object.

    Returns:
        tuple: (total_predictions, correct_predictions, interval_stats, detailed_results)
    """
    total_predictions = 0
    correct_predictions = 0
    interval_correct = {}
    interval_total = {}
    detailed_results = []

    # Define betting score intervals
    bins = list(range(0, 105, 5))  # 0-5, 5-10, ..., 100+
    labels = [f"{i}-{i+5}" for i in bins[:-1]]
    labels[-1] = "100+"  # Adjust the last label to '100+'

    # Initialize intervals including 'Unknown'
    for label in labels:
        interval_correct[label] = 0
        interval_total[label] = 0
    interval_correct['Unknown'] = 0
    interval_total['Unknown'] = 0

    # Iterate through each prediction
    for index, row in df.iterrows():
        match_id = row['matchID']
        predicted_team = row['predictedTeam']
        betting_score = row['betting_score']
        betting_score = betting_score * 100
        home_name = row['home_name']
        away_name = row['away_name']
        normalized_TTAS = row['normalized_TTAS']
        normalized_LTAS = row['normalized_LTAS']
        mcs = row['MCS']
        league = row['league_name']
        

        # Assign betting score to interval
        interval_label = assign_betting_score_interval(betting_score)

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
        interval_total[interval_label] += 1

        # Check if prediction is correct
        is_correct = predicted_team == winning_team
        if is_correct:
            correct_predictions += 1
            interval_correct[interval_label] += 1

        # Append detailed result
        detailed_results.append({
            'matchID': match_id,
            'predictedTeam': predicted_team,
            'winningTeam': winning_team,
            'is_correct': is_correct,
            'betting_score': betting_score,
            'betting_score_interval': interval_label,
            'home_name': home_name,
            'away_name': away_name,
            'MCS': mcs,
            'TTAS': normalized_TTAS,
            'LTAS': normalized_LTAS,
            'league': league
        })

    # Calculate interval statistics
    interval_stats = []
    for label in labels + ['Unknown']:
        total = interval_total[label]
        correct = interval_correct[label]
        accuracy = (correct / total) * 100 if total > 0 else np.nan  # Use NaN for undefined accuracy
        interval_stats.append({
            'betting_score_interval': label,
            'total_predictions': total,
            'correct_predictions': correct,
            'accuracy_percentage': round(accuracy, 2) if not np.isnan(accuracy) else 'N/A'
        })

    return total_predictions, correct_predictions, interval_stats, detailed_results

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

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Read CSV file
    df_predictions = read_csv(args.csv)

    # Connect to MongoDB
    client, matches_collection = connect_to_mongo(args.mongo_uri, args.db, args.collection)

    # Verify predictions and calculate accuracy per interval
    total, correct, interval_stats, details = verify_predictions(df_predictions, matches_collection)

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

        # Print accuracy per betting score interval
        print(f"\n--- Accuracy per Betting Score Interval ---")
        header = f"{'Betting Score Interval':<25} {'Total Predictions':<20} {'Correct Predictions':<20} {'Accuracy (%)':<15}"
        print(header)
        print('-' * len(header))
        for stat in interval_stats:
            accuracy_display = f"{stat['accuracy_percentage']}%" if stat['accuracy_percentage'] != 'N/A' else "N/A"
            print(f"{stat['betting_score_interval']:<25} {stat['total_predictions']:<20} {stat['correct_predictions']:<20} {accuracy_display:<15}")

    # Save detailed results if output_csv is provided
    if args.output_csv:
        save_detailed_results(details, args.output_csv)

if __name__ == "__main__":
    main()
