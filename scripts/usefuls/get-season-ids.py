import csv
import json

def extract_and_save_seasons(file_path, output_csv_path):
    # Read the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Extract seasons data
    seasons = data['season']

    # Sort seasons by year in ascending order
    sorted_seasons = sorted(seasons, key=lambda x: x['year'])

    # Write sorted data to CSV
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write the header
        writer.writerow(['year', 'id'])
        # Write the sorted data
        for season in sorted_seasons:
            writer.writerow([season['year'], season['id']])

    print(f"Data has been written to {output_csv_path}")

# Example usage
input_json_path = '/root/barnard/tmp/tmp.json'  # Replace with your input JSON file path
output_csv_path = '/root/barnard/useful-csvs/community-shield/season-ids.csv'  # Replace with your desired output CSV file path
extract_and_save_seasons(input_json_path, output_csv_path)
