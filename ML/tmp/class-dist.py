import pymongo

def calculate_match_distribution():
    # 1. Connect to MongoDB (modify the connection string if needed)
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    
    # 2. Specify the database and collection
    db = client["footballDB"]        # Replace <YOUR_DB_NAME> with the actual DB name
    matches_collection = db["matches"]
    
    # 3. Query the matches that are complete, retrieving only winningTeam, homeID, awayID
    cursor = matches_collection.find(
        {"status": "complete"},
        {"winningTeam": 1, "homeID": 1, "awayID": 1}
    )
    
    # 4. Initialize counters
    results_count = {
        "homeWin": 0,
        "awayWin": 0,
        "draw": 0
    }
    total_matches = 0
    
    # 5. Iterate over each match and increment result counters
    for match in cursor:
        total_matches += 1
        
        winning_team = match.get("winningTeam")
        home_id = match.get("homeID")
        away_id = match.get("awayID")
        
        # Check the winningTeam field to classify result
        if winning_team == -1:
            results_count["draw"] += 1
        elif winning_team == home_id:
            results_count["homeWin"] += 1
        elif winning_team == away_id:
            results_count["awayWin"] += 1
        # else: you could handle unexpected cases if needed
    
    # 6. Calculate distributions in percentages
    if total_matches > 0:
        home_win_pct = (results_count["homeWin"] / total_matches) * 100
        away_win_pct = (results_count["awayWin"] / total_matches) * 100
        draw_pct = (results_count["draw"] / total_matches) * 100
    else:
        # If no matches are found
        home_win_pct = 0
        away_win_pct = 0
        draw_pct = 0
    
    # 7. Print or return the results
    print(f"Total Matches Analyzed: {total_matches}")
    print(f"Home Win: {home_win_pct:.2f}%")
    print(f"Away Win: {away_win_pct:.2f}%")
    print(f"Draw: {draw_pct:.2f}%")

if __name__ == "__main__":
    calculate_match_distribution()
