from pymongo import MongoClient, ASCENDING
import sys

# Configuration
MONGO_URI = 'mongodb://localhost:27017'  # Replace with your MongoDB URI
DB_NAME = 'footballDB'                   # Replace with your database name
MATCHES_COLLECTION = 'matches'           # Your matches collection name

# Define the list of X values for BTTS calculations
BTTS_HISTORY_SIZES = [5, 10, 20]  # You can add more values as needed

def calculate_btts_percentage(btts_list):
    """
    Calculate the BTTS percentage based on a list of BTTS results.

    Args:
        btts_list (list): List of past BTTS results (0 or 1).

    Returns:
        float: BTTS percentage (0 to 100). Returns 0 if no games are available.
    """
    if not btts_list:
        return 0
    percentage = (sum(btts_list) / len(btts_list)) * 100
    return round(percentage, 2)

def main():
    FILTERED_COMPETITION_IDS = [3119, 246, 12, 11, 10, 9, 161, 1625, 2012, 4759, 6135, 7704, 9660, 12325]
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

        # Initialize team-wise and H2H-wise match histories
        team_matches = {}       # { team_id: [ btts1, btts2, ... ] }
        h2h_matches = {}        # { (team1_id, team2_id): [ btts1, btts2, ... ] }
        team_home_matches = {}  # { team_id: [ btts1, btts2, ... ] }
        team_away_matches = {}  # { team_id: [ btts1, btts2, ... ] }

        print("Initializing team-wise and H2H-wise match histories...")
        # No pre-fetching; we'll build histories as we process matches

        # Prepare bulk operations
        from pymongo import UpdateOne
        bulk_operations = []

        print("Processing matches and calculating BTTS percentages...")
        for match in all_matches_cursor:
            match_id = match.get('id')
            date_unix = match.get('date_unix')
            home_id = match.get('homeID')
            away_id = match.get('awayID')
            home_goal_count = match.get('homeGoalCount')
            away_goal_count = match.get('awayGoalCount')

            # Validate essential fields
            if None in (match_id, date_unix, home_id, away_id, home_goal_count, away_goal_count):
                print(f"Match_id {match_id} is missing essential fields. Skipping.")
                continue

            # Derive BTTS value for the current match
            current_btts = 1 if (home_goal_count > 0 and away_goal_count > 0) else 0

            # Initialize BTTS percentages structure
            btts_percentages = {
                'BTTS': {
                    'home': {},
                    'away': {},
                    'h2h': None,
                    'home_at_home': {},
                    'away_at_away': {}
                }
            }

            # Function to calculate BTTS percentages for a team based on their history
            def get_team_btts(team_id, X):
                """
                Calculate BTTS percentage for a team based on their last X matches.

                Args:
                    team_id (int): The team ID.
                    X (int): Number of recent matches to consider.

                Returns:
                    float: BTTS percentage or 0 if no matches are available.
                """
                matches = team_matches.get(team_id, [])
                if not matches:
                    return 0
                recent_btts = matches[-X:] if len(matches) >= X else matches
                return calculate_btts_percentage(recent_btts)

            # Function to calculate H2H BTTS percentage based on entire history
            def get_h2h_btts(home_id, away_id):
                """
                Calculate H2H BTTS percentage based on all prior matches between two teams.

                Args:
                    home_id (int): Home team ID.
                    away_id (int): Away team ID.

                Returns:
                    float: H2H BTTS percentage. Returns 0 if no prior H2H matches.
                """
                team_pair = tuple(sorted([home_id, away_id]))
                matches = h2h_matches.get(team_pair, [])
                # Always calculate BTTS percentage, defaulting to 0 if no prior matches
                return calculate_btts_percentage(matches)

            # Function to calculate BTTS percentages for home_at_home
            def get_home_at_home_btts(home_id, X):
                """
                Calculate BTTS percentage for a team based on their last X home matches.

                Args:
                    home_id (int): Home team ID.
                    X (int): Number of recent home matches to consider.

                Returns:
                    float: BTTS percentage or 0 if no home matches are available.
                """
                matches = team_home_matches.get(home_id, [])
                if not matches:
                    return 0
                recent_btts = matches[-X:] if len(matches) >= X else matches
                return calculate_btts_percentage(recent_btts)

            # Function to calculate BTTS percentages for away_at_away
            def get_away_at_away_btts(away_id, X):
                """
                Calculate BTTS percentage for a team based on their last X away matches.

                Args:
                    away_id (int): Away team ID.
                    X (int): Number of recent away matches to consider.

                Returns:
                    float: BTTS percentage or 0 if no away matches are available.
                """
                matches = team_away_matches.get(away_id, [])
                if not matches:
                    return 0
                recent_btts = matches[-X:] if len(matches) >= X else matches
                return calculate_btts_percentage(recent_btts)

            # Calculate BTTS percentages for home team
            for X in BTTS_HISTORY_SIZES:
                pct = get_team_btts(home_id, X)
                btts_percentages['BTTS']['home'][f'last_{X}'] = pct

            # Calculate BTTS percentages for away team
            for X in BTTS_HISTORY_SIZES:
                pct = get_team_btts(away_id, X)
                btts_percentages['BTTS']['away'][f'last_{X}'] = pct

            # Calculate H2H BTTS percentage based on entire history
            h2h_pct = get_h2h_btts(home_id, away_id)
            btts_percentages['BTTS']['h2h'] = h2h_pct

            # Calculate BTTS percentages for home_at_home
            for X in BTTS_HISTORY_SIZES:
                pct = get_home_at_home_btts(home_id, X)
                btts_percentages['BTTS']['home_at_home'][f'last_{X}'] = pct

            # Calculate BTTS percentages for away_at_away
            for X in BTTS_HISTORY_SIZES:
                pct = get_away_at_away_btts(away_id, X)
                btts_percentages['BTTS']['away_at_away'][f'last_{X}'] = pct

            # Prepare the bulk update operation
            bulk_operations.append(
                UpdateOne(
                    {'id': match_id},
                    {'$set': btts_percentages}
                )
            )

            # Execute bulk operations in batches of 1000
            if len(bulk_operations) >= 1000:
                try:
                    matches_collection.bulk_write(bulk_operations)
                    print(f"Executed bulk update for 1000 matches.")
                    bulk_operations = []
                except Exception as e:
                    print(f"Bulk write failed: {e}")
                    # Optionally, handle or log the error further

            # Update team_matches with the current match's BTTS
            if home_id not in team_matches:
                team_matches[home_id] = []
            team_matches[home_id].append(current_btts)

            if away_id not in team_matches:
                team_matches[away_id] = []
            team_matches[away_id].append(current_btts)

            # Update h2h_matches with the current match's BTTS
            team_pair = tuple(sorted([home_id, away_id]))
            if team_pair not in h2h_matches:
                h2h_matches[team_pair] = []
            h2h_matches[team_pair].append(current_btts)

            # Update team_home_matches and team_away_matches
            if home_id not in team_home_matches:
                team_home_matches[home_id] = []
            team_home_matches[home_id].append(current_btts)

            if away_id not in team_away_matches:
                team_away_matches[away_id] = []
            team_away_matches[away_id].append(current_btts)

        # Execute any remaining bulk operations
        if bulk_operations:
            try:
                matches_collection.bulk_write(bulk_operations)
                print(f"Executed final bulk update for {len(bulk_operations)} matches.")
            except Exception as e:
                print(f"Final bulk write failed: {e}")

        print("\nBTTS percentage calculations and updates completed.")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

    finally:
        # Close the MongoDB connection
        client.close()
        print("Disconnected from MongoDB.")

if __name__ == "__main__":
    main()
