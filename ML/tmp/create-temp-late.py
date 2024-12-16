variable_name = 'away_conceded_last_5'

team_extract_feild =  f'team_{variable_name} = btts.gte("{variable_name}")'
opponent_extract_feild =  f'opponent_{variable_name} = btts.gte("{variable_name}")'
