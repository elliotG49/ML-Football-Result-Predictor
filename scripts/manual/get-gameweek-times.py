import csv
from datetime import datetime
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string
db = client.footballDB  # Replace with your database name

# Collection name
collection = db.matches

# Read competition IDs and corresponding years from CSV file
with open('/root/barnard/data/betting/efl-championship/season-ids.csv', mode='r') as file:  # Replace with your CSV file path
    reader = csv.DictReader(file)
    competitions = [(row['id'], row['year']) for row in reader]

# Loop through each competition ID
for competition_id, year in competitions:
    competition_id = int(competition_id)
    
    # Get distinct game weeks for the competition
    game_weeks = collection.distinct("game_week", {"competition_id": competition_id})

    earliest_dates = []

    # Loop through each game week and find the earliest match date
    for game_week in game_weeks:
        # Retrieve the earliest match in the gameweek, sorted by date_unix in ascending order
        earliest_match = collection.find(
            {"game_week": game_week, "competition_id": competition_id},
            {"date_unix": 1, "id": 1}
        ).sort("date_unix", 1).limit(1)

        # Get the earliest match document
        earliest_match = list(earliest_match)[0]

        # Subtract 1 hour (3600 seconds) from the UNIX timestamp
        adjusted_date_unix = earliest_match["date_unix"] - 3600

        # Convert the adjusted UNIX timestamp to a human-readable date (date only)
        date_normal = datetime.utcfromtimestamp(adjusted_date_unix).strftime('%Y-%m-%d')

        # Store the earliest date with the required format
        earliest_dates.append({
            "game_week": game_week,
            "date_normal": date_normal,
            "date_unix": adjusted_date_unix
        })

    # Specify the output CSV file path
    csv_file_path = f'/root/barnard/data/betting/efl-championship/Gameweek-UNIX-Timestamps/{year.replace("/", "-")}.csv'

    # Write to CSV
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Gameweek", "Date (Normal)", "Date (Unix)"])  # Write the header

        for item in earliest_dates:
            writer.writerow([item["game_week"], item["date_normal"], item["date_unix"]])

    print(f"Data for competition ID {competition_id} saved to {csv_file_path}")
