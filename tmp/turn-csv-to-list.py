import pandas as pd

base = '/root/barnard/data/betting/premier-league/'
csv_file_path = f'{base}season-ids.csv'  # Change this if your CSV file has a different name or path

# Define the output Python file path
output_py_file = f'{base}season-ids.py'

# Read the CSV file into a pandas DataFrame
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: The file '{csv_file_path}' is empty.")
    exit(1)
except pd.errors.ParserError as e:
    print(f"Error parsing CSV: {e}")
    exit(1)

# Check if 'id' column exists
if 'id' not in df.columns:
    print("Error: 'id' column not found in the CSV file.")
    exit(1)

# Extract the 'id' column and convert to a list of integers
try:
    id_list = df['id'].astype(int).tolist()
except ValueError as e:
    print(f"Error converting IDs to integers: {e}")
    # Optionally, handle non-integer IDs here
    # For example, remove non-integer IDs:
    id_list = pd.to_numeric(df['id'], errors='coerce').dropna().astype(int).tolist()
    print("Non-integer IDs have been removed.")

# Create the content to be written to the Python file
python_content = f"competition_ids = {id_list}\n"

# Write the list to the Python file
with open(output_py_file, mode='w', encoding='utf-8') as pyfile:
    pyfile.write(python_content)

print(f"List of IDs has been saved to '{output_py_file}'.")
