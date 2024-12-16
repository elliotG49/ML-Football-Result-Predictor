import subprocess
import argparse

# List of scripts to run
scripts = [
    "/root/barnard/ML/Scripts/Feature-Engineering/home-away-adv_ALL.py",
    #"/root/barnard/ML/Scripts/Feature-Engineering/btts_percentages.py",
    #"/root/barnard/ML/Scripts/Feature-Engineering/conceded-last-x.py",
    #"/root/barnard/ML/Scripts/Feature-Engineering/conceded-streak.py",
    "/root/barnard/ML/Scripts/Feature-Engineering/match-rest_ALL.py",
    #"/root/barnard/ML/Scripts/Feature-Engineering/scored-streak.py"
]

def run_scripts(scripts, league_name):
    for script in scripts:
        try:
            print(f"Running {script} with league '{league_name}'...")
            command = ["python3", script, league_name]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Output of {script}:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e.stderr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple scripts with command-line arguments.')
    parser.add_argument('league', type=str, help='Name of the league (e.g., premier_league)')
    args = parser.parse_args()

    run_scripts(scripts, args.league)
