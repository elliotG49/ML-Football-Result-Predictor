from pymongo import MongoClient, ASCENDING, UpdateOne
import sys

# Configuration
MONGO_URI = 'mongodb://localhost:27017'  # Replace with your MongoDB URI
DB_NAME = 'footballDB'                   # Replace with your database name
MATCHES_COLLECTION = 'matches'           # Your matches collection name

# Define the list of competition IDs to filter matches
FILTERED_COMPETITION_IDS = [
    3119, 246, 12, 11, 10, 9, 161, 1625,
    2012, 4759, 6135, 7704, 
    9660,
    12325
]

# Define the list of X values for conceded goals calculations
CONCEDED_HISTORY_SIZES = [5, 10, 20]  # You can add more values as needed

def main():
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        matches_collection = db[MATCHES_COLLECTION]
        print("Connected successfully to MongoDB.")

        # Ensure indexes for faster queries
        try:
            matches_collection.create_index([("date_unix", ASCENDING)])
            matches_collection.create_index([("homeID", ASCENDING), ("date_unix", ASCENDING)])
            matches_collection.create_index([("awayID", ASCENDING), ("date_unix", ASCENDING)])
            print("Indexes ensured on 'date_unix', 'homeID', and 'awayID'.")
        except Exception as e:
            print(f"Index creation failed or already exists: {e}")

        # Step 1: Fetch all matches for the specified competitions, sorted by date_unix ascending
        all_matches_cursor = matches_collection.find(
            {"competition_id": {"$in": FILTERED_COMPETITION_IDS}}
        ).sort("date_unix", ASCENDING)

        # Initialize team-wise goals conceded history
        team_conceded_goals = {}  # { team_id: [goals_conceded1, goals_conceded2, ...] }

        print("Processing matches and calculating goals conceded in previous matches...")

        # Prepare bulk operations
        bulk_operations = []
        BATCH_SIZE = 1000  # Adjust based on memory and performance considerations

        for match in all_matches_cursor:
            match_id = match.get('id')
            date_unix = match.get('date_unix')
            home_id = match.get('homeID')
            away_id = match.get('awayID')
            home_goal_count = match.get('homeGoalCount')
            away_goal_count = match.get('awayGoalCount')

            # Validate essential fields
            essential_fields = ['id', 'date_unix', 'homeID', 'awayID', 'homeGoalCount', 'awayGoalCount']
            if any(match.get(field) is None for field in essential_fields):
                print(f"Match_id {match_id} is missing essential fields. Skipping.")
                continue

            # Calculate goals conceded for home and away teams in the current match
            # Home Team: Conceded goals = away_goal_count
            # Away Team: Conceded goals = home_goal_count
            home_conceded = away_goal_count
            away_conceded = home_goal_count

            # Initialize or update the goals conceded history for Home Team
            if home_id not in team_conceded_goals:
                team_conceded_goals[home_id] = []
            team_conceded_goals[home_id].append(home_conceded)

            # Initialize or update the goals conceded history for Away Team
            if away_id not in team_conceded_goals:
                team_conceded_goals[away_id] = []
            team_conceded_goals[away_id].append(away_conceded)

            # Prepare the update document with goals conceded in previous X matches
            update_doc = {}

            for X in CONCEDED_HISTORY_SIZES:
                # Calculate the total goals conceded in the last X matches before this match
                # Since the current match has just been appended, we need to exclude it
                home_goals_history = team_conceded_goals[home_id][:-1]  # Exclude current match
                away_goals_history = team_conceded_goals[away_id][:-1]  # Exclude current match

                # Get the last X matches; if fewer than X matches, sum all available
                home_last_X = home_goals_history[-X:] if len(home_goals_history) >= X else home_goals_history
                away_last_X = away_goals_history[-X:] if len(away_goals_history) >= X else away_goals_history

                # Sum the goals conceded in the last X matches
                home_conceded_last_X = sum(home_last_X)
                away_conceded_last_X = sum(away_last_X)

                # Assign to the update document under the BTTS section
                update_doc[f'BTTS.home_conceded_last_{X}'] = home_conceded_last_X
                update_doc[f'BTTS.away_conceded_last_{X}'] = away_conceded_last_X

            # Append the update operation to bulk operations
            bulk_operations.append(
                UpdateOne(
                    {'id': match_id},
                    {'$set': update_doc}
                )
            )

            # Execute bulk operations in batches
            if len(bulk_operations) >= BATCH_SIZE:
                try:
                    matches_collection.bulk_write(bulk_operations)
                    print(f"Executed bulk update for {BATCH_SIZE} matches.")
                    bulk_operations = []
                except Exception as e:
                    print(f"Bulk write failed: {e}")
                    # Optionally, log the error or handle it as needed

        # Execute any remaining bulk operations
        if bulk_operations:
            try:
                matches_collection.bulk_write(bulk_operations)
                print(f"Executed final bulk update for {len(bulk_operations)} matches.")
            except Exception as e:
                print(f"Final bulk write failed: {e}")

        print("\nGoals conceded calculations and updates completed.")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

    finally:
        # Close the MongoDB connection
        client.close()
        print("Disconnected from MongoDB.")

if __name__ == "__main__":
    main()
