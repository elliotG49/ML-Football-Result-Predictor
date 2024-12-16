import requests
import json
from api import KEY

# Replace 'YOURKEY' with your actual API key
url = f"https://api.football-data-api.com/league-list?key=dcfab3f4b36acc7031c6ddaa1212e1c35d750da0a8f3257c771c93df954b374c&chosen_leagues_only=true"

try:
    # Make the API request
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses (4XX, 5XX)

    # Parse the JSON response
    data = response.json()

    # Save the data to a JSON file
    with open('/root/barnard/tmp/league_list_2.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print("Data successfully saved.")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
