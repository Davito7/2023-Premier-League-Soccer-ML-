import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
HEADERS = {'x-rapidapi-host': "v3.football.api-sports.io", 
           'x-rapidapi-key': API_KEY}

def get_historical_matches(league_id, season):
    url = "https://v3.football.api-sports.io/fixtures"
    params = {"league": league_id, "season": season}
    response = requests.get(url, headers=HEADERS, params=params).json()
    
    matches = []
    for game in response['response']:
        if game['fixture']['status']['short'] == 'FT':
            h_goals = game['goals']['home']
            a_goals = game['goals']['away']
            
            # 2: Home Win, 1: Draw, 0: Away Win
            if h_goals > a_goals:
                res = 2
            elif h_goals == a_goals:
                res = 1
            else:
                res = 0
                
            matches.append({
                'home_id': game['teams']['home']['id'],
                'away_id': game['teams']['away']['id'],
                'home_name': game['teams']['home']['name'],
                'away_name': game['teams']['away']['name'],
                'result': res
            })
    return pd.DataFrame(matches)

# Fetch Premier League (39) for 2023 season
df = get_historical_matches(39, 2023)
df.to_csv("training_data.csv", index=False)
print("training_data.csv created successfully!")