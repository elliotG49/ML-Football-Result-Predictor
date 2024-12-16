import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib  # Added for serialization
import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import nltk
from nltk.corpus import wordnet as wn
from colorama import init, Fore, Style
import matplotlib.colors as mcolors
import yaml
import argparse  # Added for argument parsing

# Initialize colorama
init(autoreset=True)

# ===========================================
# Function Definitions
# ===========================================

def download_nltk_dependencies():
    """Downloads necessary NLTK data files."""
    nltk_packages = ['wordnet', 'omw-1.4']
    for package in nltk_packages:
        try:
            nltk.download(package, quiet=True)
            step = f"Download {package}"
            status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
            print(f"{step:<50} {status}")
        except Exception as e:
            step = f"Download {package}"
            status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
            print(f"{step:<50} {status}")
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

def get_words(pos, max_words=1000, max_length=5):
    """
    Retrieve a list of simple words for a specific part of speech from WordNet.

    Parameters:
    - pos (str): Part of speech ('a' for adjectives, 'n' for nouns).
    - max_words (int): Maximum number of words to retrieve.
    - max_length (int): Maximum length of the words to ensure simplicity.

    Returns:
    - List of simple words.
    """
    words = set()
    try:
        for synset in wn.all_synsets(pos=pos):
            for lemma in synset.lemmas():
                word = lemma.name().replace('_', ' ').lower()
                if word.isalpha() and 2 < len(word) <= max_length:
                    words.add(word)
                if len(words) >= max_words:
                    break
            if len(words) >= max_words:
                break
        step = f"Retrieve {len(words)} '{pos}' words"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = f"Retrieve '{pos}' words"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    return list(words)

def generate_model_name(prefix='v6', adjectives=[], nouns=[], random_state=None):
    """
    Generates a unique model name in the format 'v6-adj-noun-rs{random_state}'.

    Parameters:
    - prefix (str): Prefix for the model name.
    - adjectives (list): List of adjectives.
    - nouns (list): List of nouns.
    - random_state (int): The random state used.

    Returns:
    - String representing the model name.
    """
    try:
        adjective = random.choice(adjectives)
        noun = random.choice(nouns)
        if random_state is not None:
            model_name = f"{prefix}-{adjective}-{noun}-rs{random_state}"
        else:
            model_name = f"{prefix}-{adjective}-{noun}"
        step = f"Generate name: {model_name}"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        return model_name
    except IndexError:
        step = "Generate name"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: Adjective or noun list is empty.{Style.RESET_ALL}")
        return f"{prefix}-model"

def create_plots_dir(base_dir):
    """
    Creates a directory for saving plots if it doesn't exist.

    Parameters:
    - base_dir (str): Path to the base directory for plots.

    Returns:
    - Path to the plots directory.
    """
    plots_dir = os.path.join(base_dir, 'plots')
    try:
        os.makedirs(plots_dir, exist_ok=True)
        step = f"Create plots dir: {plots_dir}"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = f"Create plots dir: {plots_dir}"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    return plots_dir

def save_plot(fig, filepath):
    """
    Saves a matplotlib figure to the specified filepath.

    Parameters:
    - fig (matplotlib.figure.Figure): The figure to save.
    - filepath (str): Path where the figure will be saved.
    """
    try:
        fig.savefig(filepath)
        plt.close(fig)
        step = f"Save plot: {os.path.basename(filepath)}"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = f"Save plot: {filepath}"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

# ===========================================
# Main Script
# ===========================================

def main(random_state, config):
    # ===========================================
    # Setup and Configuration
    # ===========================================

    league_name = config['league']
    data_path = config['training_dataset_pathway']['BTTS']
    metrics_dir_base = config['training_model_pathways']['BTTS']

    print(f"{Fore.CYAN}--- Start Training Script ---{Style.RESET_ALL}")

    # Download NLTK dependencies
    download_nltk_dependencies()

    # Get lists of adjectives and nouns
    adjectives = get_words('a', max_words=1000, max_length=7)  # Simpler adjectives
    nouns = get_words('n', max_words=1000, max_length=5)       # Simpler nouns

    if not adjectives:
        step = "Adjective retrieval"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: No adjectives found. Exiting.{Style.RESET_ALL}")
        return
    if not nouns:
        step = "Noun retrieval"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: No nouns found. Exiting.{Style.RESET_ALL}")
        return

    # Generate a unique model name
    model_base_name = generate_model_name(prefix='v6', adjectives=adjectives, nouns=nouns, random_state=random_state)

    # Define paths
    metrics_dir = os.path.join(metrics_dir_base, model_base_name)
    model_filename = os.path.join(metrics_dir, f"{model_base_name}.joblib")
    features_filename = os.path.join(metrics_dir, "features.json")
    metrics_filename = os.path.join(metrics_dir, "metrics.json")
    plots_dir = create_plots_dir(metrics_dir)

    # Create metrics directory
    try:
        os.makedirs(metrics_dir, exist_ok=True)
        step = f"Create metrics dir: {metrics_dir}"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = f"Create metrics dir: {metrics_dir}"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return

    # ===========================================
    # Model Notes
    # ===========================================

    model_notes = f"""
    Model Notes:
    ------------
    - This model uses two rows per match, which allows for the presence of a 'is_home' variable, aiming to increase the home_win accuracy percentage.
    - This model uses the dataset specified in the config file.
    - This model predicts the 'BTTS' value (Both Teams to Score).
    - Random state used: {random_state}
    - League: {league_name}
    """

    # ===========================================
    # Save Notes
    # ===========================================

    try:
        # Define the path for notes.txt
        notes_filename = os.path.join(metrics_dir, "notes.txt")
        # Save the model notes to notes.txt
        with open(notes_filename, 'w') as notes_file:
            notes_file.write(model_notes)
        step = f"Save Notes at '{notes_filename}'"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = f"Save Notes at '{notes_filename}'"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return

    # ===========================================
    # Load and Prepare Data
    # ===========================================

    print(f"{Fore.CYAN}--- Load & Prepare Data ---{Style.RESET_ALL}")

    try:
        df = pd.read_csv(data_path)
        step = f"Load dataset from '{data_path}'"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = f"Load dataset from '{data_path}'"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return

    # The rest of the code remains the same, adjusting variables as needed

    # Continue with the existing code, making sure to adjust any hardcoded paths or variables to use values from the config

    # ... [Rest of the main function code] ...

    # For brevity, only significant changes are shown; the rest of your code remains as in the original script 2,
    # but with paths and variables adjusted to use the configuration values.

# ===========================================
# Entry Point
# ===========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model for BTTS prediction.')
    parser.add_argument('config_name', type=str, help='Name of the YAML config file (without .yaml extension).')
    args = parser.parse_args()

    config_name = args.config_name

    config_base_path = '/root/barnard/ML/Configs/'
    config_path = os.path.join(config_base_path, f'{config_name}.yaml')

    # Load the YAML config file
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        step = f"Load config file from '{config_path}'"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = f"Load config file from '{config_path}'"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        exit(1)

    random_states = [42]  # List of different random states
    overall_metrics = []

    for rs in random_states:
        print(f"\n{Fore.BLUE}Running experiment with random_state = {rs}{Style.RESET_ALL}\n")
        metrics = main(random_state=rs, config=config)
        overall_metrics.append(metrics)

    # Aggregate and display results
    if overall_metrics:
        metrics_df = pd.DataFrame(overall_metrics)
        print(f"\n{Fore.CYAN}--- Aggregated Results ---{Style.RESET_ALL}")
        print(metrics_df.describe())

        # Save aggregated metrics
        # Assuming 'model_metrics_pathway' in config for saving aggregated metrics
        model_metrics_path = config.get('model_metrics_pathway', {}).get('BTTS', '/path/to/default/model_metrics')
        aggregated_metrics_filename = os.path.join(model_metrics_path, 'aggregated_metrics.json')

        # Ensure the directory exists
        os.makedirs(model_metrics_path, exist_ok=True)

        metrics_df.to_json(aggregated_metrics_filename, orient='records', indent=4)
        print(f"\nAggregated metrics saved to {aggregated_metrics_filename}")

        # Optional: Plot distribution of overall accuracies
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(metrics_df['overall_accuracy'], bins=10, kde=True, ax=ax, color='skyblue')
            ax.set_title('Distribution of Overall Accuracies Across Random States')
            ax.set_xlabel('Overall Accuracy')
            plot_path = os.path.join(model_metrics_path, 'accuracy_distribution.png')
            fig.savefig(plot_path)
            plt.close(fig)
            print(f"Accuracy distribution plot saved to {plot_path}")
        except Exception as e:
            step = "Plot Accuracy Distribution"
            status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
            print(f"{step:<50} {status}")
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
