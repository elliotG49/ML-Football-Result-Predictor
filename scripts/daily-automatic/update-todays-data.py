import requests
import json
import csv
import os
import schedule
import time
import logging
import subprocess
from pymongo import MongoClient
from api import KEY
from datetime import datetime, timezone, timedelta
import threading
import shutil
import pickle
import pytz
import sys
import traceback
from pushover import PKEY, USER_KEY  # Import Pushover API keys

# Set up logging    
logging.basicConfig(
    filename='/root/barnard/logs/todays_matches_update.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Function to send Pushover notifications
def send_pushover_notification(message, title='Script Notification'):
    url = 'https://api.pushover.net/1/messages.json'
    payload = {
        'token': PKEY,
        'user': USER_KEY,
        'message': message,
        'title': title
    }
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        logging.info('Pushover notification sent successfully.')
    except requests.exceptions.RequestException as e:
        logging.error(f'Failed to send Pushover notification: {e}')
        logging.error(f'Response Content: {response.text}')

# MongoDB connection setup
client = MongoClient('mongodb://localhost:27017/')
db = client['footballDB']
players_collection = db['players']

# Directory where files will be saved
json_dir = '/root/barnard/data/betting/todays-matches/'
csv_dir = '/root/barnard/data/betting/todays-matches/'
if not os.path.exists(json_dir):
    os.makedirs(json_dir)

# File to store scheduled tasks
scheduled_tasks_file = '/root/barnard/logs/scheduled_tasks.pkl'

# Global list to store directories to clean up after post-game updates
csv_dirs_to_cleanup = []

# Function to save scheduled tasks to a file
def save_scheduled_tasks():
    with open(scheduled_tasks_file, 'wb') as f:
        pickle.dump(schedule.jobs, f)

# Function to load scheduled tasks from a file
def load_scheduled_tasks():
    if os.path.exists(scheduled_tasks_file):
        with open(scheduled_tasks_file, 'rb') as f:
            saved_jobs = pickle.load(f)
            for job in saved_jobs:
                schedule.jobs.append(job)

# Function to fetch today's matches and save data to CSV files
def fetch_todays_matches():
    url = f"https://api.football-data-api.com/todays-matches?key={KEY}&timezone=BST"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        with open(os.path.join(json_dir, 'todays-matches.json'), 'w') as json_file:
            json.dump(data, json_file, indent=4)
        logging.info("Today's matches data successfully saved to JSON.")
    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred while fetching today's matches: {e}")
        send_pushover_notification(f"Error fetching today's matches: {e}", title="Script Error")
        return

    # Extract IDs and competition_id for teams, players, and matches, and create separate CSV files
    matches_by_competition = {}

    for match in data.get('data', []):
        home_id = match['homeID']
        away_id = match['awayID']
        competition_id = match['competition_id']

        # Add match data (match_id and competition_id) to matches CSV
        match_data = {'match_id': match['id'], 'competition_id': competition_id}

        # Add team data (team_id and competition_id) to teams CSV
        team_data_home = {'team_id': home_id, 'competition_id': competition_id}
        team_data_away = {'team_id': away_id, 'competition_id': competition_id}

        # Retrieve player IDs based on homeID and awayID
        player_ids = get_player_ids(home_id, competition_id) + get_player_ids(away_id, competition_id)
        player_data = [{'player_id': pid, 'competition_id': competition_id} for pid in player_ids]

        if competition_id not in matches_by_competition:
            matches_by_competition[competition_id] = {
                'teams': [],
                'players': [],
                'matches': []
            }

        matches_by_competition[competition_id]['teams'].extend([team_data_home, team_data_away])
        matches_by_competition[competition_id]['matches'].append(match_data)
        matches_by_competition[competition_id]['players'].extend(player_data)

    # Save to separate CSV files
    for competition_id, data in matches_by_competition.items():
        # Create a directory for this competition_id
        specific_csv_dir = os.path.join(csv_dir, str(competition_id))
        os.makedirs(specific_csv_dir, exist_ok=True)

        # Save the data to CSV files
        teams_csv_path = os.path.join(specific_csv_dir, 'teams.csv')
        matches_csv_path = os.path.join(specific_csv_dir, 'matches.csv')
        players_csv_path = os.path.join(specific_csv_dir, 'players.csv')

        save_to_csv(teams_csv_path, ['team_id', 'competition_id'], data['teams'])
        save_to_csv(matches_csv_path, ['match_id', 'competition_id'], data['matches'])
        save_to_csv(players_csv_path, ['player_id', 'competition_id'], data['players'])

        logging.info(f"CSV files created successfully for competition_id {competition_id} at {specific_csv_dir}")

        # Run the update scripts sequentially
        run_scripts_sequentially(specific_csv_dir)

        # Add directory to list for later cleanup
        if specific_csv_dir not in csv_dirs_to_cleanup:
            csv_dirs_to_cleanup.append(specific_csv_dir)

    # Do not clean up CSV directories here; we'll clean up after post-game updates

# Function to get player IDs based on team_id and competition_id
def get_player_ids(team_id, competition_id):
    query = {
        'club_team_id': team_id,
        'competition_id': competition_id
    }
    players = players_collection.find(query, {'id': 1})
    return [player['id'] for player in players]

# Function to save data to a CSV file
def save_to_csv(file_path, fieldnames, data):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    logging.info(f"Data saved to CSV file: {file_path}")

# Function to run the update scripts sequentially
def run_scripts_sequentially(specific_csv_dir):
    scripts = [
        ('/root/barnard/scripts/daily-automatic/update-teams.py', os.path.join(specific_csv_dir, 'teams.csv')),
        ('/root/barnard/scripts/daily-automatic/update-matches.py', os.path.join(specific_csv_dir, 'matches.csv')),
        ('/root/barnard/scripts/daily-automatic/update-recent-form.py', os.path.join(specific_csv_dir, 'teams.csv')),
        ('/root/barnard/scripts/daily-automatic/update-league-table.py', None),  # No CSV required
        ('/root/barnard/scripts/daily-automatic/update-players.py', os.path.join(specific_csv_dir, 'players.csv'))
    ]

    for script, csv_path in scripts:
        if csv_path:
            run_script(script, csv_path)
        else:
            # If no CSV path is provided, just run the script
            run_script(script)

    # Do not clean up CSV directories here; we'll clean up after post-game updates

# Function to run external Python scripts with a CSV file as an argument
def run_script(script_name, csv_path=None):
    try:
        if csv_path:
            logging.info(f"Running script: {script_name} with CSV: {csv_path}")
            command = ['/root/barnard/myenv/bin/python3', script_name, csv_path]
        else:
            logging.info(f"Running script: {script_name}")
            command = ['/root/barnard/myenv/bin/python3', script_name]
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        logging.info(f"Script {script_name} completed successfully")
        logging.info(f"Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running script {script_name}")
        logging.error(f"Exit Status: {e.returncode}")
        logging.error(f"Standard Output:\n{e.stdout}")
        logging.error(f"Standard Error:\n{e.stderr}")
        send_pushover_notification(f"Error running script {script_name}: {e.stderr}", title="Script Error")

# Cleanup of CSV directories
def schedule_cleanup():
    for directory in csv_dirs_to_cleanup:
        try:
            shutil.rmtree(directory)
            logging.info(f"Cleaned up directory: {directory}")
        except Exception as e:
            logging.error(f"Error cleaning up directory {directory}: {e}")
    csv_dirs_to_cleanup.clear()

# Function to run post-game updates
def run_post_game_updates():
    logging.info("Running post-game updates.")
    send_pushover_notification("Post-game updates are starting.", title="Post-game Update")
    # Re-fetch today's matches to get updated data
    fetch_todays_matches()
    # After running the scripts, clean up the directories
    schedule_cleanup()
    send_pushover_notification("Post-game updates completed.", title="Post-game Update")

# Function to determine the end time of the latest match and schedule post-game updates
def schedule_post_game_updates():
    url = f"https://api.football-data-api.com/todays-matches?key={KEY}&timezone=BST"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        latest_match_end_time = None

        for match in data.get('data', []):
            # Assuming match['date'] contains the match start time in ISO format in BST
            match_time_str = match['date']
            # Parse the time and set the timezone to BST
            match_time = datetime.strptime(match_time_str, '%Y-%m-%d %H:%M:%S')
            bst = pytz.timezone('Europe/London')  # BST time zone
            match_time = bst.localize(match_time)
            # Assuming average match duration is 2 hours (including extra time)
            match_end_time = match_time + timedelta(hours=2)

            if not latest_match_end_time or match_end_time > latest_match_end_time:
                latest_match_end_time = match_end_time

        if latest_match_end_time:
            # Convert match end time to local time zone
            local_tz = pytz.timezone('Europe/London')  # Assuming your server is in BST
            scheduled_time = latest_match_end_time + timedelta(minutes=15)  # Add extra buffer time
            scheduled_time = scheduled_time.astimezone(local_tz)

            # Ensure the scheduled time is in the future
            now = datetime.now(local_tz)
            if scheduled_time <= now:
                # If scheduled time is in the past, schedule it for the next day
                scheduled_time += timedelta(days=1)

            scheduled_time_str = scheduled_time.strftime('%H:%M')

            logging.info(f"Scheduling post-game updates at {scheduled_time_str} local time")
            send_pushover_notification(f"Post-game updates scheduled at {scheduled_time_str} local time", title="Scheduling Update")

            # Check if the job is already scheduled
            for job in schedule.get_jobs():
                if job.job_func == run_post_game_updates and job.at_time.strftime('%H:%M') == scheduled_time_str:
                    logging.info(f"Post-game update already scheduled at {scheduled_time_str}, skipping scheduling.")
                    return

            # Schedule the job
            schedule.every().day.at(scheduled_time_str).do(run_post_game_updates)

            # Save scheduled tasks
            save_scheduled_tasks()
        else:
            logging.warning("No matches found to schedule post-game updates.")
    except Exception as e:
        logging.error(f"Error scheduling post-game updates: {e}")
        logging.error(traceback.format_exc())
        send_pushover_notification(f"Error scheduling post-game updates: {e}", title="Scheduling Error")

# Load saved scheduled tasks (if any)
load_scheduled_tasks()

# Run fetch_todays_matches() immediately
fetch_todays_matches()
send_pushover_notification("Pre-game updates have started.", title="Pre-game Update")

# Schedule fetch_todays_matches() to run every day at 00:01
schedule.every().day.at("00:01").do(fetch_todays_matches)
logging.info("Scheduled daily fetch_todays_matches at 00:01")
send_pushover_notification("Scheduled daily pre-game updates at 00:01.", title="Scheduling Update")

# Schedule post-game updates
schedule_post_game_updates()

# Save scheduled tasks after scheduling them
save_scheduled_tasks()

# Main loop to run the scheduler
while True:
    schedule.run_pending()
    time.sleep(1)

# Close the MongoDB connection when done
client.close()
