import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import nltk
from nltk.corpus import wordnet as wn
import yaml
import argparse

def download_nltk_dependencies():
    nltk_packages = ['wordnet', 'omw-1.4']
    for package in nltk_packages:
        try:
            nltk.download(package, quiet=True)
            print(f"Download {package:<50} ✔ Successful")
        except Exception as e:
            print(f"Download {package:<50} ✖ Failed")
            print(f"Error: {e}")

def get_words(pos, max_words=1000, max_length=5):
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
        print(f"Retrieve {len(words)} '{pos}' words                 ✔ Successful")
    except Exception as e:
        print(f"Retrieve '{pos}' words                              ✖ Failed")
        print(f"Error: {e}")
    return list(words)

def generate_model_name(prefix='v6', adjectives=[], nouns=[], random_state=None):
    try:
        adjective = random.choice(adjectives)
        noun = random.choice(nouns)
        if random_state is not None:
            model_name = f"{prefix}-{adjective}-{noun}-rs{random_state}"
        else:
            model_name = f"{prefix}-{adjective}-{noun}"
        print(f"Generate name: {model_name:<50} ✔ Successful")
        return model_name
    except IndexError:
        print(f"Generate name                                        ✖ Failed")
        print("Error: Adjective or noun list is empty.")
        return f"{prefix}-model"

def create_plots_dir(base_dir):
    plots_dir = os.path.join(base_dir, 'plots')
    try:
        os.makedirs(plots_dir, exist_ok=True)
        print(f"Create plots dir: {plots_dir:<50} ✔ Successful")
    except Exception as e:
        print(f"Create plots dir: {plots_dir:<50} ✖ Failed")
        print(f"Error: {e}")
    return plots_dir

def save_plot(fig, filepath):
    try:
        fig.savefig(filepath)
        plt.close(fig)
        print(f"Save plot: {os.path.basename(filepath):<50} ✔ Successful")
    except Exception as e:
        print(f"Save plot: {filepath:<50} ✖ Failed")
        print(f"Error: {e}")

def main(random_state, config):
    league_name = config['league']
    data_path = config['training_dataset_pathway']['BTTS_FTS']
    metrics_dir_base = config['training_model_pathways']['BTTS_FTS']

    print("--- Start Training Script ---")

    download_nltk_dependencies()

    adjectives = get_words('a', max_words=1000, max_length=7)
    nouns = get_words('n', max_words=1000, max_length=5)

    if not adjectives:
        print("Adjective retrieval                                ✖ Failed")
        print("Error: No adjectives found. Exiting.")
        return
    if not nouns:
        print("Noun retrieval                                     ✖ Failed")
        print("Error: No nouns found. Exiting.")
        return

    model_base_name = generate_model_name(prefix='v6', adjectives=adjectives, nouns=nouns, random_state=random_state)

    metrics_dir = os.path.join(metrics_dir_base, model_base_name)
    model_filename = os.path.join(metrics_dir, f"{model_base_name}.joblib")
    metrics_filename = os.path.join(metrics_dir, "metrics.json")
    plots_dir = create_plots_dir(metrics_dir)

    try:
        os.makedirs(metrics_dir, exist_ok=True)
        print(f"Create metrics dir: {metrics_dir:<50} ✔ Successful")
    except Exception as e:
        print(f"Create metrics dir: {metrics_dir:<50} ✖ Failed")
        print(f"Error: {e}")
        return

    model_notes = f"""
    Model Notes:
    ------------
    - This model predicts BTTS (Both Teams to Score).
    - This model uses two rows per match, which allows for the presence of a 'is_home' variable, aiming to increase the home_win accuracy percentage.
    - This model uses the dataset specified in the config file.
    - This model does not use betting odds.
    - Random state used: {random_state}
    - League: {league_name}
    """

    try:
        notes_filename = os.path.join(metrics_dir, "notes.txt")
        with open(notes_filename, 'w') as notes_file:
            notes_file.write(model_notes)
        print(f"Save Notes at '{notes_filename}'                  ✔ Successful")
    except Exception as e:
        print(f"Save Notes at '{notes_filename}'                  ✖ Failed")
        print(f"Error: {e}")
        return

    print("--- Load & Prepare Data ---")

    try:
        df = pd.read_csv(data_path)
        print(f"Load dataset from '{data_path}'                   ✔ Successful")
    except Exception as e:
        print(f"Load dataset from '{data_path}'                   ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        df["BTTS"] = df["BTTS"].fillna(0)  # Assuming 'BTTS' column exists
        print("Fill missing 'BTTS' with 0                           ✔ Successful")
    except Exception as e:
        print("Fill missing 'BTTS' with 0                           ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        btts_0 = df[df["BTTS"] == 0]
        btts_1 = df[df["BTTS"] == 1]
        print("Separate classes                                   ✔ Successful")
    except Exception as e:
        print("Separate classes                                   ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        min_samples = min(len(btts_0), len(btts_1))
        print(f"Min samples for balancing: {min_samples:<35} ✔ Successful")
    except Exception as e:
        print("Determine min samples for balancing                ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        btts_0_downsampled = btts_0.sample(n=min_samples, random_state=random_state)
        btts_1_downsampled = btts_1.sample(n=min_samples, random_state=random_state)
        print("Resample to balance classes                        ✔ Successful")
    except Exception as e:
        print("Resample to balance classes                        ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        df_balanced = pd.concat([btts_0_downsampled, btts_1_downsampled])
        print("Combine balanced classes                           ✔ Successful")
    except Exception as e:
        print("Combine balanced classes                           ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
        print("Shuffle balanced dataset                           ✔ Successful")
    except Exception as e:
        print("Shuffle balanced dataset                           ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        features = [
            'team_id', 'opponent_id',
            'team_ELO_before', 'opponent_ELO_before',
            'h2h_btts',
            #'team_btts_l5',
            #'team_btts_l10',
            #'team_btts_l20',
            #'opponent_btts_l5',
            #'opponent_btts_l10',
            #'opponent_btts_l20',
            #'team_mc_btts_l5',
            #'team_mc_btts_l10',
            #'team_mc_btts_l20',
            #'opponent_mc_btts_l5',
            #'opponent_mc_btts_l10',
            #'opponent_mc_btts_l20',
            'team_conceded_l5', 
            'team_conceded_l10',
            'team_conceded_l20',
            'opponent_conceded_l5',   
            'opponent_conceded_l10',  
            'opponent_conceded_l20',  
            'team_scoring_streak',    
            'opponent_scoring_streak',
            'team_conceded_streak',
            'opponent_conceded_streak',
            'betting_h2h_betting_odds',
            'betting_h2h_betting_percentage'
        ]
        
        X = df_balanced[features]
        y = df_balanced["BTTS"]  # Changed target variable
        print("Define features and target                         ✔ Successful")
    except Exception as e:
        print("Define features and target                         ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        match_details = df_balanced[["team_id", "opponent_id", "match_id"]]
        print("Store match details                                ✔ Successful")
    except Exception as e:
        print("Store match details                                ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        model = RandomForestClassifier(n_estimators=165, random_state=random_state, min_samples_leaf=1)
        print("Init Random Forest Classifier                      ✔ Successful")
    except Exception as e:
        print("Init Random Forest Classifier                      ✖ Failed")
        print(f"Error: {e}")
        return

    print("--- Stratified K-Fold Cross-Validation ---")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    cv_accuracies, cv_precisions, cv_recalls, cv_f1s = [], [], [], []
    feature_importances = np.zeros(X.shape[1])
    predictions, confidences, actuals = [], [], []
    home_ids, away_ids, match_ids = [], [], []
    class_labels = [0, 1]  # Updated for binary classification
    class_names = ['No BTTS', 'BTTS']  # Updated class names

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        try:
            print(f"Process Fold {fold:<44} In Progress")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            probas = model.predict_proba(X_test)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds, average='macro')
            recall = recall_score(y_test, preds, average='macro')
            f1 = f1_score(y_test, preds, average='macro')
            cv_accuracies.append(acc)
            cv_precisions.append(precision)
            cv_recalls.append(recall)
            cv_f1s.append(f1)
            predictions.extend(preds)
            actuals.extend(y_test)
            confidences.extend(np.max(probas, axis=1))
            home_ids.extend(match_details.iloc[test_index]["team_id"])
            away_ids.extend(match_details.iloc[test_index]["opponent_id"])
            match_ids.extend(match_details.iloc[test_index]["match_id"])
            feature_importances += model.feature_importances_
            cm = confusion_matrix(y_test, preds, labels=class_labels)
            report = classification_report(y_test, preds, labels=class_labels, target_names=class_names, zero_division=0)
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            print(f"Classification Report for Fold {fold}:\n{report}")
            print(f"Confusion Matrix for Fold {fold}:\n{cm_df}\n{'-'*60}\n")
            print(f"Process Fold {fold:<44} ✔ Successful (Acc: {acc:.2f}, Prec: {precision:.2f}, Rec: {recall:.2f}, F1: {f1:.2f})")
        except Exception as e:
            print(f"Process Fold {fold:<44} ✖ Failed")
            print(f"Error: {e}")

    feature_importances /= skf.get_n_splits()
    print("Calculate feature importances                      ✔ Successful")

    print("--- Compile Results ---")

    try:
        results_df = pd.DataFrame({
            'Actual': actuals,
            'Predicted': predictions,
            'Confidence': confidences,
            'HomeID': home_ids,
            'AwayID': away_ids,
            'MatchID': match_ids
        })
        print("Compile DataFrame                                   ✔ Successful")
    except Exception as e:
        print("Compile DataFrame                                   ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        overall_accuracy = (results_df['Actual'] == results_df['Predicted']).mean()
        print(f"Overall Accuracy                                    {overall_accuracy:.2f}")
    except Exception as e:
        print("Calculate Overall Accuracy                          ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        report = classification_report(results_df['Actual'], results_df['Predicted'], output_dict=True, zero_division=0)
        overall_precision = report['macro avg']['precision']
        overall_recall = report['macro avg']['recall']
        overall_f1 = report['macro avg']['f1-score']
        print(f"Overall Precision, Recall, F1-Score                 Prec: {overall_precision:.2f}, Rec: {overall_recall:.2f}, F1: {overall_f1:.2f}")
    except Exception as e:
        print("Calculate Overall Precision, Recall, F1-Score       ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        cm = confusion_matrix(results_df['Actual'], results_df['Predicted'], labels=class_labels)
        print("Calculate Confusion Matrix                          ✔ Successful")
    except Exception as e:
        print("Calculate Confusion Matrix                          ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        high_confidence_threshold = 0.51
        high_confidence_df = results_df[results_df['Confidence'] >= high_confidence_threshold]
        print(f"Filter High-Confidence (>= {int(high_confidence_threshold*100)}%)        ✔ Successful")
    except Exception as e:
        print("Filter High-Confidence Predictions                  ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        if len(high_confidence_df) > 0:
            high_confidence_accuracy = (high_confidence_df['Actual'] == high_confidence_df['Predicted']).mean()
            high_conf_report = classification_report(high_confidence_df['Actual'], high_confidence_df['Predicted'], output_dict=True, zero_division=0)
            high_conf_precision = high_conf_report['macro avg']['precision']
            high_conf_recall = high_conf_report['macro avg']['recall']
            high_conf_f1 = high_conf_report['macro avg']['f1-score']
            print(f"High-Confidence Metrics                             Acc: {high_confidence_accuracy:.2f}, Prec: {high_conf_precision:.2f}, Rec: {high_conf_recall:.2f}, F1: {high_conf_f1:.2f}")
        else:
            high_confidence_accuracy = high_conf_precision = high_conf_recall = high_conf_f1 = 0
            print("High-Confidence Metrics                             ✖ No High-Confidence Predictions")
    except Exception as e:
        print("Calculate High-Confidence Metrics                   ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        mean_cv_accuracy = np.mean(cv_accuracies)
        std_cv_accuracy = np.std(cv_accuracies)
        mean_cv_precision = np.mean(cv_precisions)
        std_cv_precision = np.std(cv_precisions)
        mean_cv_recall = np.mean(cv_recalls)
        std_cv_recall = np.std(cv_recalls)
        mean_cv_f1 = np.mean(cv_f1s)
        std_cv_f1 = np.std(cv_f1s)

        print(f"Mean CV Accuracy                                    {mean_cv_accuracy:.2f}")
        print(f"CV Accuracy Std Dev                                 {std_cv_accuracy:.2f}")
        print(f"Mean CV Precision                                   {mean_cv_precision:.2f}")
        print(f"CV Precision Std Dev                                {std_cv_precision:.2f}")
        print(f"Mean CV Recall                                      {mean_cv_recall:.2f}")
        print(f"CV Recall Std Dev                                   {std_cv_recall:.2f}")
        print(f"Mean CV F1-Score                                    {mean_cv_f1:.2f}")
        print(f"CV F1-Score Std Dev                                 {std_cv_f1:.2f}")
    except Exception as e:
        print("Calculate CV Statistics                             ✖ Failed")
        print(f"Error: {e}")

    print("--- Retrain Model on Entire Dataset ---")

    try:
        final_model = RandomForestClassifier(n_estimators=165, random_state=random_state, min_samples_leaf=1)
        final_model.fit(X, y)
        print("Retrain model on entire dataset                     ✔ Successful")
    except Exception as e:
        print("Retrain model on entire dataset                     ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        joblib.dump(final_model, model_filename)
        print(f"Save trained model to '{model_filename}'            ✔ Successful")
    except Exception as e:
        print(f"Save trained model to '{model_filename}'            ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        feature_importances_dict = dict(zip(features, feature_importances))
        with open(os.path.join(metrics_dir, "feature_importances.json"), 'w') as f:
            json.dump(feature_importances_dict, f, indent=4)
        print("Save feature importances                            ✔ Successful")
    except Exception as e:
        print("Save feature importances                            ✖ Failed")
        print(f"Error: {e}")

    print("--- Generate and Save Plots ---")

    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        cm_plot_path = os.path.join(plots_dir, 'confusion_matrix.png')
        save_plot(plt.gcf(), cm_plot_path)
    except Exception as e:
        print("Plot Confusion Matrix                               ✖ Failed")
        print(f"Error: {e}")

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
        print("Plot Feature Importances                            ✖ Failed")
        print(f"Error: {e}")

    try:
        overall_cm = confusion_matrix(results_df['Actual'], results_df['Predicted'], labels=class_labels)
        overall_cm_list = overall_cm.tolist()
        print("Calculate Overall Confusion Matrix                  ✔ Successful")
    except Exception as e:
        print("Calculate Overall Confusion Matrix                  ✖ Failed")
        print(f"Error: {e}")
        return

    try:
        metrics = {
            "random_state": random_state,
            "overall_accuracy": overall_accuracy,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "cv_accuracies": cv_accuracies,
            "mean_cv_accuracy": mean_cv_accuracy,
            "std_cv_accuracy": std_cv_accuracy,
            "cv_precisions": cv_precisions,
            "mean_cv_precision": mean_cv_precision,
            "std_cv_precision": std_cv_precision,
            "cv_recalls": cv_recalls,
            "mean_cv_recall": mean_cv_recall,
            "std_cv_recall": std_cv_recall,
            "cv_f1s": cv_f1s,
            "mean_cv_f1": mean_cv_f1,
            "std_cv_f1": std_cv_f1,
            "feature_importances": feature_importances_dict,
            "high_confidence_threshold": high_confidence_threshold,
            "high_confidence_accuracy": high_confidence_accuracy,
            "high_confidence_precision": high_conf_precision,
            "high_confidence_recall": high_conf_recall,
            "high_confidence_f1": high_conf_f1,
            "number_high_confidence_predictions": len(high_confidence_df),
            "overall_confusion_matrix": overall_cm_list
        }

        with open(metrics_filename, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Save Metrics at '{metrics_filename}'                ✔ Successful")
    except Exception as e:
        print(f"Save Metrics at '{metrics_filename}'                ✖ Failed")
        print(f"Error: {e}")

    print("--- Training Script Completed ---")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model for BTTS prediction.')  # Updated description
    parser.add_argument('config_name', type=str, help='Name of the YAML config file (without .yaml extension).')
    args = parser.parse_args()

    config_name = args.config_name

    config_base_path = '/root/barnard/ML/Configs/'
    config_path = os.path.join(config_base_path, f'{config_name}.yaml')

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Load config file from '{config_path}'               ✔ Successful")
    except Exception as e:
        print(f"Load config file from '{config_path}'               ✖ Failed")
        print(f"Error: {e}")
        exit(1)

    random_states = [42]
    overall_metrics = []

    for rs in random_states:
        print(f"\nRunning experiment with random_state = {rs}\n")
        metrics = main(random_state=rs, config=config)
        overall_metrics.append(metrics)

    if overall_metrics:
        metrics_df = pd.DataFrame(overall_metrics)
        print("\n--- Aggregated Results ---")
        print(metrics_df.describe())
