from pymongo import MongoClient, ASCENDING, UpdateOne
import sys
import logging

# Configure logging
logging.basicConfig(
    filename='consecutive_conceded_streak.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Configuration
MONGO_URI = 'mongodb://localhost:27017'  # Replace with your MongoDB URI
DB_NAME = 'footballDB'                   # Replace with your database name
MATCHES_COLLECTION = 'matches'           # Your matches collection name

# Define the list of competition IDs to filter matches
FILTERED_COMPETITION_IDS = [
    3119, 246, 12, 11, 10, 9, 161, 1625,
    2012, 4759, 6135, 7704, 
    9660, 12325
]

def main():
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        matches_collection = db[MATCHES_COLLECTION]
        logging.info("Connected successfully to MongoDB.")

        # Ensure indexes for faster queries
        try:
            matches_collection.create_index([("date_unix", ASCENDING)])
            matches_collection.create_index([("homeID", ASCENDING), ("date_unix", ASCENDING)])
            matches_collection.create_index([("awayID", ASCENDING), ("date_unix", ASCENDING)])
            logging.info("Indexes ensured on 'date_unix', 'homeID', and 'awayID'.")
        except Exception as e:
            logging.error(f"Index creation failed or already exists: {e}")

        # Step 1: Fetch all matches for the specified competitions, sorted by date_unix ascending
        all_matches_cursor = matches_collection.find(
            {"competition_id": {"$in": FILTERED_COMPETITION_IDS}}
        ).sort("date_unix", ASCENDING)

        # Initialize team-wise match histories
        # This dictionary will store the current streak count for each team
        team_conceded_streak = {}  # { team_id: consecutive_conceded_count }

        logging.info("Processing matches and calculating consecutive conceded goal streaks...")

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
                logging.warning(f"Match_id {match_id} is missing essential fields. Skipping.")
                continue

            # Assign current streaks to the match's BTTS field BEFORE updating streaks based on current match
            # This ensures the streak reflects only prior matches
            home_current_streak = team_conceded_streak.get(home_id, 0)
            away_current_streak = team_conceded_streak.get(away_id, 0)

            # Prepare the update document within the BTTS field
            update_doc = {
                'BTTS.consecutive_conceded_home': home_current_streak,
                'BTTS.consecutive_conceded_away': away_current_streak
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
                    logging.info(f"Executed bulk update for {BATCH_SIZE} matches.")
                    bulk_operations = []
                except Exception as e:
                    logging.error(f"Bulk write failed: {e}")
                    # Optionally, implement retry logic or handle the error as needed

            # Determine if teams conceded in the current match
            # For Home Team: They conceded if Away Team scored >=1
            # For Away Team: They conceded if Home Team scored >=1
            home_conceded = 1 if away_goal_count >= 1 else 0
            away_conceded = 1 if home_goal_count >= 1 else 0

            # Update the consecutive conceded streak for Home Team based on current match
            if home_id in team_conceded_streak:
                if home_conceded:
                    team_conceded_streak[home_id] += 1
                else:
                    team_conceded_streak[home_id] = 0
            else:
                # Initialize streak
                team_conceded_streak[home_id] = 1 if home_conceded else 0

            # Update the consecutive conceded streak for Away Team based on current match
            if away_id in team_conceded_streak:
                if away_conceded:
                    team_conceded_streak[away_id] += 1
                else:
                    team_conceded_streak[away_id] = 0
            else:
                # Initialize streak
                team_conceded_streak[away_id] = 1 if away_conceded else 0

        # Execute any remaining bulk operations
        if bulk_operations:
            try:
                matches_collection.bulk_write(bulk_operations)
                logging.info(f"Executed final bulk update for {len(bulk_operations)} matches.")
            except Exception as e:
                logging.error(f"Final bulk write failed: {e}")

        logging.info("Consecutive conceded goal streak calculations and updates completed.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

    finally:
        # Close the MongoDB connection
        client.close()
        logging.info("Disconnected from MongoDB.")

if __name__ == "__main__":
    main()
