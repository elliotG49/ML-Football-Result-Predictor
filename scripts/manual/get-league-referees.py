import json
import requests
import os
from api import KEY

# Define all paths here
SEASON_ID = 12325
base_url = f'https://api.football-data-api.com/league-referees?key={KEY}&season_id={SEASON_ID}'
all_referees_stats_path = '/root/barnard/data/2425/referee-stats-raw/all_referees.json'
output_directory = '/root/barnard/data/2425/referee-stats-raw/'
os.makedirs(output_directory, exist_ok=True)

# Initialize an empty list to store all referee data
all_referees = []

# Start with the first page
current_page = 1

while True:
    league_referees_url = f'{base_url}&page={current_page}'
    
    response = requests.get(league_referees_url)
    
    if response.status_code == 200:
        data = response.json()
        
        # Add the referees from the current page to the list
        all_referees.extend(data['data'])
        
        # Check if this is the last page
        if current_page >= data['pager']['max_page']:
            break
        
        # Move to the next page
        current_page += 1
    else:
        print("Error fetching data:", response.status_code)
        break

# Save the combined data to a JSON file
with open(all_referees_stats_path, 'w', encoding='utf-8') as f:
    json.dump({'data': all_referees}, f, indent=4)
    print(f"All referee data has been written to {all_referees_stats_path}")

# Load the JSON data and split into individual referee files
with open(all_referees_stats_path, 'r') as file:
    data = json.load(file)

# Loop through each referee's data and save to a separate JSON file
for referee in data['data']:
    referee_name = referee['shorthand'].replace(" ", "-").lower()
    filename = f"{referee_name}.json"
    filepath = os.path.join(output_directory, filename)
    
    with open(filepath, 'w') as f:
        json.dump(referee, f, indent=4)

    print(f"Saved {filename}")

print("All referees have been saved to separate files.")

# Finally, delete the all_referees.json file
if os.path.exists(all_referees_stats_path):
    os.remove(all_referees_stats_path)
    print(f"{all_referees_stats_path} has been deleted.")
