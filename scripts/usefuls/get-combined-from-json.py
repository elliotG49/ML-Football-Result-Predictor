import json
import csv
import pycountry
import re

# Function to convert year to 'YYYY/YYYY' format
def convert_year(year_value):
    """
    Converts the year field to 'YYYY/YYYY' format.
    Handles both 8-digit years (e.g., 20172018) and single years (e.g., 2024).

    Args:
        year_value (int or str): The year value from JSON.

    Returns:
        str: Formatted season year or 'Invalid Year' if format is unrecognized.
    """
    year_str = str(year_value)
    if len(year_str) == 8 and year_str.isdigit():
        # Format: 20172018 -> 2017/2018
        start_year = year_str[:4]
        end_year = year_str[4:]
        return f"{start_year}/{end_year}"
    elif len(year_str) == 4 and year_str.isdigit():
        # Format: 2024 -> 2024/2025
        start_year = year_str
        try:
            end_year = str(int(start_year) + 1)
            return f"{start_year}/{end_year}"
        except ValueError:
            return "Invalid Year"
    else:
        return "Invalid Year"

# Function to format league name
def format_league_name(name):
    """
    Formats the league name by converting it to lowercase and replacing
    non-alphanumeric characters with hyphens.

    Args:
        name (str): Original league name.

    Returns:
        str: Formatted league name.
    """
    name = name.lower()
    name = re.sub(r'[^a-z0-9]+', '-', name)
    name = name.strip('-')
    return name

# Function to get country code
def get_country_code(country_name):
    """
    Maps the country name to its ISO 3166-1 alpha-3 code.
    Returns 'EUR' for Europe and 'UNK' for unknown countries.

    Args:
        country_name (str): Name of the country.

    Returns:
        str: ISO 3166-1 alpha-3 country code.
    """
    if country_name.lower() == "europe":
        return "EUR"
    try:
        country = pycountry.countries.lookup(country_name)
        return country.alpha_3
    except LookupError:
        return "UNK"  # Unknown

def main():
    # Load JSON data from a file
    input_file = '/root/barnard/tmp/league_list_2.json'
    output_file = 'output.csv'

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON. {e}")
        return

    # Define CSV headers
    csv_headers = [
        "season",
        "competition_id",
        "league_name",
        "domestic_value",
        "league_type",
        "country_code"
    ]

    # Dictionary to hold leagues and their seasons
    leagues = {}

    # Iterate over each league in the data
    for league in json_data.get('data', []):
        league_name_original = league.get('name', '')
        league_name = format_league_name(league_name_original)
        country = league.get('country', '')
        country_code = get_country_code(country)
        league_type = "interleague" if country.lower() == "europe" else "domestic"

        # Initialize the league entry if not already present
        if league_name not in leagues:
            leagues[league_name] = {
                "league_name_original": league_name_original,
                "league_name_formatted": league_name,
                "country": country,
                "country_code": country_code,
                "league_type": league_type,
                "seasons": []
            }

        # Iterate over each season for the league
        for season in league.get('season', []):
            year_value = season.get('year', '')
            season_year = convert_year(year_value)
            competition_id = season.get('id', '')

            # Prepare the row
            row = {
                "season": season_year,
                "competition_id": competition_id,
                "league_name": league_name,
                "domestic_value": 1,  # As per your instruction
                "league_type": league_type,
                "country_code": country_code
            }

            # Log invalid years
            if row["season"] == "Invalid Year":
                print(f"Warning: Invalid year format for competition_id {competition_id} in league '{league_name_original}'")

            # Add the row to the league's seasons
            leagues[league_name]["seasons"].append(row)

    # Function to extract starting year for sorting
    def get_starting_year(season_str):
        """
        Extracts the starting year from a season string formatted as 'YYYY/YYYY'.

        Args:
            season_str (str): The season string.

        Returns:
            int: The starting year as an integer. Returns a large number for 'Invalid Year'.
        """
        try:
            return int(season_str.split('/')[0])
        except (ValueError, AttributeError, IndexError):
            return float('inf')  # Places 'Invalid Year' at the end

    # Open CSV file for writing
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()

            # Iterate over each league in the sorted order (optional: sort leagues alphabetically)
            for league_name in sorted(leagues.keys()):
                league_info = leagues[league_name]
                seasons = league_info["seasons"]

                # Sort seasons within the league
                seasons_sorted = sorted(seasons, key=lambda x: get_starting_year(x['season']))

                # Write each season to the CSV
                for row in seasons_sorted:
                    writer.writerow(row)

        print(f"CSV file '{output_file}' has been created successfully.")
    except IOError as e:
        print(f"Error: Unable to write to CSV file. {e}")

if __name__ == "__main__":
    main()
