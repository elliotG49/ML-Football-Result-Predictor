import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score, 
                             classification_report)
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
import argparse

# Initialize colorama
init(autoreset=True)

# =====================
# Toggle Balancing
# =====================
# Set to True to apply the class balancing strategy.
# Set to False to keep the original distribution of classes.
BALANCE_CLASSES = False

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
    data_path = config['training_dataset_pathway']['Match_Result']
    metrics_dir_base = config['training_model_pathways']['Match_Result']

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
    - Uses min-sample-leaf = 1
    - Using new dataset with 1 row per match
    - {'Balancing classes' if BALANCE_CLASSES else 'Keeping original class distribution'}
    - Added Team_ID's and xG within feature set
    """

    # ===========================================
    # Save Notes
    # ===========================================

    try:
        notes_filename = os.path.join(metrics_dir, "notes.txt")
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

    try:
        # Replace missing values in 'winning_team' with 0 (assuming 0 represents a draw)
        df["winning_team"] = df["winning_team"].fillna(0)
        step = "Fill missing 'winning_team' with 0"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = "Fill missing 'winning_team' with 0"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return

    # ===========================================
    # Either Balance Classes or Keep Original
    # ===========================================
    if BALANCE_CLASSES:
        try:
            # Separate each class
            draws = df[df["winning_team"] == 0]
            home_wins = df[df["winning_team"] == 2]  # 2 = Home Win
            away_wins = df[df["winning_team"] == 1]  # 1 = Away Win
            step = "Separate classes"
            status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
            print(f"{step:<50} {status}")
        except Exception as e:
            step = "Separate classes"
            status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
            print(f"{step:<50} {status}")
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            return

        try:
            # Ensure there are enough samples
            min_samples = min(len(draws), len(home_wins), len(away_wins))
            step = f"Min samples for balancing: {min_samples}"
            status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
            print(f"{step:<50} {status}")
        except Exception as e:
            step = "Determine min samples for balancing"
            status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
            print(f"{step:<50} {status}")
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            return

        try:
            # Resample to balance the dataset
            home_wins_downsampled = home_wins.sample(n=min_samples, random_state=random_state)
            draws_upsampled = draws.sample(n=min_samples, replace=True, random_state=random_state)
            away_wins_downsampled = away_wins.sample(n=min_samples, replace=True, random_state=random_state)

            df_balanced = pd.concat([home_wins_downsampled, away_wins_downsampled, draws_upsampled])
            df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

            step = f"Classes balanced (draws={len(draws_upsampled)}, home_wins={len(home_wins_downsampled)}, away_wins={len(away_wins_downsampled)})"
            status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
            print(f"{step:<50} {status}")

        except Exception as e:
            step = "Resample to balance classes"
            status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
            print(f"{step:<50} {status}")
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            return

    else:
        # Use original distribution
        step = f"Keep original class distribution"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        df_balanced = df.copy()

    # ===========================================
    # Define Features and Target
    # ===========================================
    try:
        features = [
            'team_id', 'opponent_id',
            'team_ELO_before', 'opponent_ELO_before',
            'odds_team_win', 'odds_draw', 'odds_opponent_win',
            'opponent_rest_days', 'team_rest_days',
            'team_h2h_win_percent', 'opponent_h2h_win_percent',
            'pre_match_home_ppg', 
            'pre_match_away_ppg',
            'opponent_ppg_mc', 'team_ppg_mc', 
            'team_home_advantage', 
            'opponent_away_advantage', 
            'pre_match_home_xg',
            'pre_match_away_xg'
        ]

        X = df_balanced[features]
        y = df_balanced["winning_team"]

        match_details = df_balanced[["team_id", "opponent_id", "match_id"]]

        step = "Define features and target"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = "Define features and target"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return

    # ===========================================
    # Initialize the Random Forest Classifier
    # ===========================================
    MIN_SAMPLE_LEAF = 1
    N_ESTIMATORS = 165

    try:
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=random_state,
            min_samples_leaf=MIN_SAMPLE_LEAF,
            n_jobs=-1
        )
        step = "Init Random Forest Classifier"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = "Init Random Forest Classifier"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return

    # ===========================================
    # Perform Stratified K-Fold Cross-Validation
    # ===========================================
    print(f"{Fore.CYAN}--- Stratified K-Fold Cross-Validation ---{Style.RESET_ALL}")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    cv_accuracies = []
    cv_precisions = []
    cv_recalls = []
    cv_f1s = []
    feature_importances = np.zeros(X.shape[1])

    # Lists to store predictions, confidence scores, and match details
    predictions = []
    confidences = []
    actuals = []
    home_ids = []
    away_ids = []
    match_ids = []

    class_labels = [0, 1, 2]  # 0: Draw, 1: Away, 2: Home
    class_names = ['Draw', 'Away Win', 'Home Win']

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        try:
            step = f"Process Fold {fold}"
            status = f"{Fore.YELLOW}In Progress{Style.RESET_ALL}"
            print(f"{step:<50} {status}")

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Train the model
            model.fit(X_train, y_train)

            # Predict probabilities and determine confidence
            probas = model.predict_proba(X_test)
            preds = model.predict(X_test)

            # Calculate metrics for this fold
            acc = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds, average='macro')
            recall = recall_score(y_test, preds, average='macro')
            f1 = f1_score(y_test, preds, average='macro')

            cv_accuracies.append(acc)
            cv_precisions.append(precision)
            cv_recalls.append(recall)
            cv_f1s.append(f1)

            # Store the predictions, confidences, and actual values
            predictions.extend(preds)
            actuals.extend(y_test)
            confidences.extend(np.max(probas, axis=1))  # Max probability as confidence

            # Store match details
            home_ids.extend(match_details.iloc[test_index]["team_id"])
            away_ids.extend(match_details.iloc[test_index]["opponent_id"])
            match_ids.extend(match_details.iloc[test_index]["match_id"])

            # Capture feature importances
            feature_importances += model.feature_importances_

            # Confusion Matrix and Classification Report
            cm = confusion_matrix(y_test, preds, labels=class_labels)
            report = classification_report(y_test, preds, labels=class_labels, 
                                           target_names=class_names, zero_division=0)

            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            print(f"{Fore.MAGENTA}Classification Report for Fold {fold}:{Style.RESET_ALL}")
            print(report)
            print(f"{Fore.MAGENTA}Confusion Matrix for Fold {fold}:{Style.RESET_ALL}")
            print(cm_df)
            print("\n" + "-"*60 + "\n")

            status = (f"{Fore.GREEN}✔ Successful "
                      f"(Acc: {acc:.2f}, Prec: {precision:.2f}, Rec: {recall:.2f}, F1: {f1:.2f}){Style.RESET_ALL}")
            print(f"{step:<50} {status}")
        except Exception as e:
            status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
            print(f"{step:<50} {status}")
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

    # Average feature importance
    feature_importances /= skf.get_n_splits()
    step = "Calculate feature importances"
    status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
    print(f"{step:<50} {status}")

    # ===========================================
    # Compile Results
    # ===========================================
    print(f"{Fore.CYAN}--- Compile Results ---{Style.RESET_ALL}")

    try:
        results_df = pd.DataFrame({
            'Actual': actuals,
            'Predicted': predictions,
            'Confidence': confidences,
            'HomeID': home_ids,
            'AwayID': away_ids,
            'MatchID': match_ids
        })
        step = "Compile DataFrame"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = "Compile DataFrame"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return

    try:
        overall_accuracy = (results_df['Actual'] == results_df['Predicted']).mean()
        step = "Overall Accuracy"
        status = f"{Fore.GREEN}{overall_accuracy:.2f}{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = "Calculate Overall Accuracy"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return

    try:
        report = classification_report(results_df['Actual'], results_df['Predicted'], 
                                       output_dict=True, zero_division=0)
        overall_precision = report['macro avg']['precision']
        overall_recall = report['macro avg']['recall']
        overall_f1 = report['macro avg']['f1-score']
        step = "Overall Precision, Recall, F1-Score"
        status = (f"{Fore.GREEN}Prec: {overall_precision:.2f}, Rec: {overall_recall:.2f}, "
                  f"F1: {overall_f1:.2f}{Style.RESET_ALL}")
        print(f"{step:<50} {status}")
    except Exception as e:
        step = "Calculate Overall Precision, Recall, F1-Score"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return

    try:
        cm = confusion_matrix(results_df['Actual'], results_df['Predicted'], labels=class_labels)
        step = "Calculate Confusion Matrix"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = "Calculate Confusion Matrix"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return

    try:
        high_confidence_threshold = 0.51
        high_confidence_df = results_df[results_df['Confidence'] >= high_confidence_threshold]
        step = f"Filter High-Confidence (>= {int(high_confidence_threshold*100)}%)"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = "Filter High-Confidence Predictions"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return

    try:
        if len(high_confidence_df) > 0:
            high_confidence_accuracy = (high_confidence_df['Actual'] == high_confidence_df['Predicted']).mean()
            high_conf_report = classification_report(high_confidence_df['Actual'], 
                                                     high_confidence_df['Predicted'], 
                                                     output_dict=True, zero_division=0)
            high_conf_precision = high_conf_report['macro avg']['precision']
            high_conf_recall = high_conf_report['macro avg']['recall']
            high_conf_f1 = high_conf_report['macro avg']['f1-score']
            step = "High-Confidence Metrics"
            status = (f"{Fore.GREEN}Acc: {high_confidence_accuracy:.2f}, Prec: {high_conf_precision:.2f}, "
                      f"Rec: {high_conf_recall:.2f}, F1: {high_conf_f1:.2f}{Style.RESET_ALL}")
            print(f"{step:<50} {status}")
        else:
            high_confidence_accuracy = 0
            high_conf_precision = 0
            high_conf_recall = 0
            high_conf_f1 = 0
            step = "High-Confidence Metrics"
            status = f"{Fore.RED}✖ No High-Confidence Predictions{Style.RESET_ALL}"
            print(f"{step:<50} {status}")
    except Exception as e:
        step = "Calculate High-Confidence Metrics"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return

    # ===========================================
    # Print Summary Statistics
    # ===========================================
    try:
        mean_cv_accuracy = np.mean(cv_accuracies)
        std_cv_accuracy = np.std(cv_accuracies)
        mean_cv_precision = np.mean(cv_precisions)
        std_cv_precision = np.std(cv_precisions)
        mean_cv_recall = np.mean(cv_recalls)
        std_cv_recall = np.std(cv_recalls)
        mean_cv_f1 = np.mean(cv_f1s)
        std_cv_f1 = np.std(cv_f1s)

        step = "Mean CV Accuracy"
        status = f"{Fore.GREEN}{mean_cv_accuracy:.2f}{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        step = "CV Accuracy Std Dev"
        status = f"{Fore.GREEN}{std_cv_accuracy:.2f}{Style.RESET_ALL}"
        print(f"{step:<50} {status}")

        step = "Mean CV Precision"
        status = f"{Fore.GREEN}{mean_cv_precision:.2f}{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        step = "CV Precision Std Dev"
        status = f"{Fore.GREEN}{std_cv_precision:.2f}{Style.RESET_ALL}"
        print(f"{step:<50} {status}")

        step = "Mean CV Recall"
        status = f"{Fore.GREEN}{mean_cv_recall:.2f}{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        step = "CV Recall Std Dev"
        status = f"{Fore.GREEN}{std_cv_recall:.2f}{Style.RESET_ALL}"
        print(f"{step:<50} {status}")

        step = "Mean CV F1-Score"
        status = f"{Fore.GREEN}{mean_cv_f1:.2f}{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        step = "CV F1-Score Std Dev"
        status = f"{Fore.GREEN}{std_cv_f1:.2f}{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = "Calculate CV Statistics"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

    # ===========================================
    # Retrain the Model on the Entire Dataset
    # ===========================================
    print(f"{Fore.CYAN}--- Retrain Model on Entire Dataset ---{Style.RESET_ALL}")

    try:
        final_model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=random_state,
            min_samples_leaf=MIN_SAMPLE_LEAF
        )
        final_model.fit(X, y)
        step = "Retrain model on entire dataset"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = "Retrain model on entire dataset"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return

    # ===========================================
    # Save the Trained Model and Features
    # ===========================================
    try:
        joblib.dump(final_model, model_filename)
        step = f"Save trained model to '{model_filename}'"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = f"Save trained model to '{model_filename}'"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return

    try:
        feature_importances_dict = dict(zip(features, feature_importances))
        with open(os.path.join(metrics_dir, "feature_importances.json"), 'w') as f_imp:
            json.dump(feature_importances_dict, f_imp, indent=4)
        step = "Save feature importances"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = "Save feature importances"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

    # ===========================================
    # Generate and Save Plots
    # ===========================================
    print(f"{Fore.CYAN}--- Generate and Save Plots ---{Style.RESET_ALL}")

    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        cm_plot_path = os.path.join(plots_dir, 'confusion_matrix.png')
        save_plot(plt.gcf(), cm_plot_path)
    except Exception as e:
        step = "Plot Confusion Matrix"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

    try:
        plt.figure(figsize=(10, 8))
        sorted_features = sorted(feature_importances_dict.items(), key=lambda x: x[1], reverse=True)
        features_sorted, importances_sorted = zip(*sorted_features)
        sns.barplot(x=importances_sorted, y=features_sorted, palette='viridis')
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances')
        fi_plot_path = os.path.join(plots_dir, 'feature_importances.png')
        save_plot(plt.gcf(), fi_plot_path)
    except Exception as e:
        step = "Plot Feature Importances"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

    # ===========================================
    # Save Model Metrics
    # ===========================================
    try:
        overall_cm = cm.tolist()
        step = "Calculate Overall Confusion Matrix"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = "Calculate Overall Confusion Matrix"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return

    try:
        metrics = {
            "balance_classes": BALANCE_CLASSES,
            "n_estimators": N_ESTIMATORS,
            "min_sample_leaf": MIN_SAMPLE_LEAF,
            "random_state": random_state,
            "overall_accuracy": overall_accuracy,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "cv_accuracies": cv_accuracies,
            "mean_cv_accuracy": float(np.mean(cv_accuracies)),
            "std_cv_accuracy": float(np.std(cv_accuracies)),
            "cv_precisions": cv_precisions,
            "mean_cv_precision": float(np.mean(cv_precisions)),
            "std_cv_precision": float(np.std(cv_precisions)),
            "cv_recalls": cv_recalls,
            "mean_cv_recall": float(np.mean(cv_recalls)),
            "std_cv_recall": float(np.std(cv_recalls)),
            "cv_f1s": cv_f1s,
            "mean_cv_f1": float(np.mean(cv_f1s)),
            "std_cv_f1": float(np.std(cv_f1s)),
            "feature_importances": feature_importances_dict,
            "high_confidence_threshold": high_confidence_threshold,
            "high_confidence_accuracy": high_confidence_accuracy,
            "high_confidence_precision": high_conf_precision,
            "high_confidence_recall": high_conf_recall,
            "high_confidence_f1": high_conf_f1,
            "number_high_confidence_predictions": len(high_confidence_df),
            "overall_confusion_matrix": overall_cm
        }

        with open(metrics_filename, 'w') as f_json:
            json.dump(metrics, f_json, indent=4)
        step = f"Save Metrics at '{metrics_filename}'"
        status = f"{Fore.GREEN}✔ Successful{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
    except Exception as e:
        step = f"Save Metrics at '{metrics_filename}'"
        status = f"{Fore.RED}✖ Failed{Style.RESET_ALL}"
        print(f"{step:<50} {status}")
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

    print(f"{Fore.CYAN}--- Training Script Completed ---{Style.RESET_ALL}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model for Match Result prediction.')
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

    random_states = [22]  # If you want multiple experiments, add more random states here
    overall_metrics = []

    for rs in random_states:
        print(f"\n{Fore.BLUE}Running experiment with random_state = {rs}{Style.RESET_ALL}\n")
        metrics = main(random_state=rs, config=config)
        overall_metrics.append(metrics)

    # Optionally aggregate or analyze overall_metrics here
    if overall_metrics:
        df_agg = pd.DataFrame(overall_metrics)
        print(f"\n{Fore.CYAN}--- Aggregated Results ---{Style.RESET_ALL}")
        print(df_agg.describe())
