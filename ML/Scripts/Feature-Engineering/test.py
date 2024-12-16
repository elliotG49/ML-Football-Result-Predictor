from pymongo import MongoClient, ASCENDING
from datetime import datetime, timezone
import sys

# Configuration
MONGO_URI = 'mongodb://localhost:27017'  # Replace with your MongoDB URI
DB_NAME = 'footballDB'                   # Replace with your database name
MATCHES_COLLECTION = 'matches'           # Your matches collection name

# Specify the competition_ids to filter by
FILTERED_COMPETITION_IDS = [4123, 4105, 7, 5, 4, 3, 177, 1636, 4392, 4673, 6192, 7664, 9655, 12337, 12316, 12530, 12325, 12529]
# Replace with your competition IDs

def calculate_full_days(previous_timestamp, current_timestamp):
    """
    Calculate the number of full days between two Unix timestamps.

    Args:
        previous_timestamp (int): Unix timestamp of the previous match in seconds.
        current_timestamp (int): Unix timestamp of the current match in seconds.

    Returns:
        int: Number of full days between the two matches.
    """
    previous_date = datetime.fromtimestamp(previous_timestamp, tz=timezone.utc)
    current_date = datetime.fromtimestamp(current_timestamp, tz=timezone.utc)
    delta = current_date - previous_date
    return delta.days

def main():
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        matches_collection = db[MATCHES_COLLECTION]
        print("Connected successfully to MongoDB.")

        # (Optional) Create a unique index on 'id' to prevent duplicates if updating existing documents
        # Uncomment the following lines if you want to ensure uniqueness based on 'id'
        """
        try:
            matches_collection.create_index([("id", ASCENDING)], unique=True)
            print(f"Ensured unique index on 'id' in '{MATCHES_COLLECTION}' collection.")
        except Exception as e:
            print(f"Index creation failed or already exists: {e}")
        """

        # Step 1: Fetch all matches sorted by date_unix ascending
        all_matches_cursor = matches_collection.find().sort("date_unix", ASCENDING)

        # Initialize a dictionary to keep track of the last match date for each team
        last_match_dates = {}

        # Iterate through all matches in chronological order
        for match in all_matches_cursor:
            match_date_unix = match.get('date_unix')
            competition_id = match.get('competition_id')
            home_id = match.get('homeID')
            away_id = match.get('awayID')
            match_id = match.get('id')  # Assuming 'id' is the unique identifier for matches

            # Check if the match is within the specified competitions
            if competition_id in FILTERED_COMPETITION_IDS:
                # Calculate Home Team Rest Days
                if home_id in last_match_dates:
                    home_rest_days = calculate_full_days(last_match_dates[home_id], match_date_unix)
                else:
                    home_rest_days = None  # First match for the team

                # Calculate Away Team Rest Days
                if away_id in last_match_dates:
                    away_rest_days = calculate_full_days(last_match_dates[away_id], match_date_unix)
                else:
                    away_rest_days = None  # First match for the team

                # Prepare the update document with 'team_a_rest_days' and 'team_b_rest_days'
                update_fields = {
                    'team_a_rest_days': home_rest_days,
                    'team_b_rest_days': away_rest_days
                }

                # Update the corresponding match document in the 'matches' collection
                try:
                    result = matches_collection.update_one(
                        {'id': match_id},           # Filter to find the correct match
                        {'$set': update_fields},    # Fields to update
                        upsert=False                 # Do not insert if not found
                    )

                    if result.matched_count > 0:
                        print(f"Updated match_id {match_id} with rest_days.")
                    else:
                        print(f"No document found for match_id {match_id}. Skipping update.")
                        # If you want to insert a new document when not found, uncomment below
                        """
                        new_document = {
                            'id': match_id,
                            'team_a_rest_days': home_rest_days,
                            'team_b_rest_days': away_rest_days
                            # Add other necessary fields here if needed
                        }
                        matches_collection.insert_one(new_document)
                        print(f"Inserted new document for match_id {match_id}.")
                        """
                except Exception as e:
                    print(f"Failed to update match_id {match_id}: {e}")

            # Update the last match date for both teams regardless of competition
            last_match_dates[home_id] = match_date_unix
            last_match_dates[away_id] = match_date_unix

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

    finally:
        # Close the MongoDB connection
        client.close()
        print("\nDisconnected from MongoDB.")

if __name__ == "__main__":
    main()