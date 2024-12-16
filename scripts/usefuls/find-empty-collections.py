import csv
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('localhost', 27017)  # Adjust host and port if needed
db = client.footballDB  # Replace with your database name

# Collections to check
collections_to_check = {
    'matches': db.matches,
    'players': db.players,
    'teams': db.teams
}

# Path to the CSV file containing competition IDs and seasons
csv_file_path = '/root/barnard/data/betting/usefuls/combined-season-ids.csv'  # Update with your actual file path

# Output file path
output_file_path = '/root/barnard/data/betting/usefuls/empty-collections.txt'  # Update with your desired output file path

# ANSI color codes
RED = "\033[91m"
ENDC = "\033[0m"

# Function to check if a competition ID and season exist in a collection
def check_competition_in_collection(collection, competition_id, season):
    query = {"competition_id": competition_id, "season": season}
    count = collection.count_documents(query)
    return count > 0

# Open the output file for writing
with open(output_file_path, 'w') as output_file:
    # Read the CSV file and check each competition ID and season
    with open(csv_file_path, mode='r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            competition_id = int(row['id'])  # Adjust column names if necessary
            season = row['year']
            league_name = row['league-name']  # Adjust column name if necessary

            output_file.write(f"\nChecking for Competition ID: {competition_id}, Season: {season}, League: {league_name}\n")

            for collection_name, collection in collections_to_check.items():
                if not check_competition_in_collection(collection, competition_id, season):
                    missing_message = f"Missing in {collection_name} collection for Season: {season}, League: {league_name}\n"
                    output_file.write(missing_message)
                    # Print in red for missing entries
                    print(f"{RED}{missing_message}{ENDC}")

print("\nCompleted checking all competition IDs and seasons. Output written to:", output_file_path)
