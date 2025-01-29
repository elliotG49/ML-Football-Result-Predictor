import pandas as pd
from colorama import init, Fore, Style
# Load the dataset
file_path = "/root/barnard/ML/Models/combined_leagues/Datasets/MR_Dataset_2_copy.csv"  # Replace with your actual file path
data = pd.read_csv(file_path)

# Create a copy of the dataset to work with
cleaned_data = data.copy()

# Step 1: Filter out rows where any of the odds columns have a value of 0
columns_to_check = ['odds_team_win', 'odds_draw', 'odds_opponent_win']
# Using DataFrame.any() to identify rows with any zero in the specified columns
cleaned_data = cleaned_data[~(cleaned_data[columns_to_check] == 0).any(axis=1)]

# Step 2: Keep only the rows where is_home == 1
cleaned_data = cleaned_data[cleaned_data['is_home'] == 1]



# Optional: Reset the index for cleanliness
cleaned_data.reset_index(drop=True, inplace=True)

# Step 4: Save the cleaned dataset to a new file
cleaned_file_path = "/root/barnard/ML/Models/combined_leagues/Datasets/MR_Dataset_Clean_Home.csv"  # Replace with your desired file name
cleaned_data.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset with home team perspective saved to {cleaned_file_path}")
