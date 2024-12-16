import requests
import csv
import os
from pymongo import MongoClient
from api import KEY

# API base URL and API key
base_url = 'https://api.football-data-api.com/league-tables'

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string
db = client.footballDB  # Replace with your database name
collection = db.league_table  # Collection where data will be inserted

# Path to the CSV file containing game week data
csv_filename = '/root/barnard/data/csv/premier-league/Gameweek-UNIX-Timestamps/2017-2018.csv'  # Use the corrected CSV with quotes around dates

# Function to sanitize the filename by removing problematic characters
def sanitize_filename(filename):
    return filename.replace(",", "").replace(" ", "_").replace(":", "-")

# Function to fetch and insert data into MongoDB
def fetch_and_insert_data(season_id, game_week, completion_date, unix_timestamp):
    # Construct the API URL with the required parameters
    url = f'{base_url}?key={KEY}&season_id={season_id}&max_time={unix_timestamp}'
    
    try:
        # Perform the API call
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        
        # Parse the response JSON
        data = response.json()
        
        # Extract the league_table data
        league_table = data.get('data', {}).get('league_table', [])
        
        # Add metadata to each entry in the league table
        for entry in league_table:
            entry['season_id'] = int(season_id)
            entry['game_week'] = int(game_week)
            entry['completion_date'] = completion_date
            entry['unix_timestamp'] = int(unix_timestamp)
            entry['id'] = int(entry['id'])
        
        # Insert the data into the MongoDB collection
        if league_table:
            collection.insert_many(league_table)
        
        print(f"Data for Season {season_id}, Game Week {game_week}, Date {completion_date} inserted into the database.")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching data for Game Week {game_week} on {completion_date}: {e}")

# Read the CSV and perform API calls
def process_csv_and_fetch_data(csv_filename, season_id):
    with open(csv_filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            game_week = row['Gameweek']
            completion_date = row['Date (Normal)']
            unix_timestamp = row['Date (Unix)']
            
            # Perform the API call and insert data into MongoDB
            fetch_and_insert_data(season_id, game_week, completion_date, unix_timestamp)

# Example call for a specific season, change the season_id as needed
season_id = 161  # Example season_id for the 2023/2024 season

# Process the CSV and fetch data for the given season_id
process_csv_and_fetch_data(csv_filename, season_id)
