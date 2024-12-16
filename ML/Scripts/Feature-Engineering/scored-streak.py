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

        # Initialize team-wise scoring streaks
        team_scoring_streak = {}  # { team_id: consecutive_scoring_count }

        print("Processing matches and calculating consecutive scoring streaks...")

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

            # Assign current scoring streaks to the match's BTTS structure
            home_current_streak = team_scoring_streak.get(home_id, 0)
            away_current_streak = team_scoring_streak.get(away_id, 0)

            # Prepare the update document with scoring streaks
            update_doc = {
                'BTTS.home_scoring_streak': home_current_streak,
                'BTTS.away_scoring_streak': away_current_streak
            }

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

            # Determine if teams scored in the current match
            # Home Team: Scored if home_goal_count >= 1
            # Away Team: Scored if away_goal_count >= 1
            home_scored = 1 if home_goal_count >= 1 else 0
            away_scored = 1 if away_goal_count >= 1 else 0

            # Update team_scoring_streak based on current match's scoring
            if home_scored:
                team_scoring_streak[home_id] = home_current_streak + 1
            else:
                team_scoring_streak[home_id] = 0

            if away_scored:
                team_scoring_streak[away_id] = away_current_streak + 1
            else:
                team_scoring_streak[away_id] = 0

        # Execute any remaining bulk operations
        if bulk_operations:
            try:
                matches_collection.bulk_write(bulk_operations)
                print(f"Executed final bulk update for {len(bulk_operations)} matches.")
            except Exception as e:
                print(f"Final bulk write failed: {e}")

        print("\nConsecutive scoring streak calculations and updates completed.")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

    finally:
        # Close the MongoDB connection
        client.close()
        print("Disconnected from MongoDB.")

if __name__ == "__main__":
    main()
