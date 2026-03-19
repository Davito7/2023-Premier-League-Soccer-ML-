from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # Allows the frontend to call this API from any origin

# --- Load model and data once at startup ---
model = joblib.load("soccer_model.pkl")
data = pd.read_csv("training_data.csv")

# Build name → ID mapping
names = pd.concat([data['home_name'], data['away_name']]).unique()
team_mapping = {}
for name in names:
    row = data[data['home_name'] == name]
    if not row.empty:
        team_mapping[name] = int(row.iloc[0]['home_id'])
    else:
        team_mapping[name] = int(data[data['away_name'] == name].iloc[0]['away_id'])

TEAM_NAMES = sorted(team_mapping.keys())


# ── GET /teams ────────────────────────────────────────────────────────────────
# Returns the list of all team names for populating the dropdowns.
@app.route("/teams")
def get_teams():
    return jsonify(TEAM_NAMES)


# ── GET /predict?home=Arsenal&away=Chelsea ────────────────────────────────────
# Runs the Random Forest model and returns win/draw/loss probabilities.
@app.route("/predict")
def predict():
    home = request.args.get("home")
    away = request.args.get("away")

    if not home or not away:
        return jsonify({"error": "Both 'home' and 'away' query params are required."}), 400
    if home not in team_mapping:
        return jsonify({"error": f"Unknown team: {home}"}), 400
    if away not in team_mapping:
        return jsonify({"error": f"Unknown team: {away}"}), 400
    if home == away:
        return jsonify({"error": "Home and away teams must be different."}), 400

    features = pd.DataFrame(
        [[team_mapping[home], team_mapping[away]]],
        columns=["home_id", "away_id"]
    )

    # predict_proba returns [away_win, draw, home_win] (labels 0, 1, 2)
    probs = model.predict_proba(features)[0]

    return jsonify({
        "home":  round(float(probs[2]), 4),   # label 2 = home win
        "draw":  round(float(probs[1]), 4),   # label 1 = draw
        "away":  round(float(probs[0]), 4),   # label 0 = away win
    })


# ── GET /h2h?teamA=Arsenal&teamB=Chelsea ─────────────────────────────────────
# Returns all historical head-to-head matches between two teams.
@app.route("/h2h")
def head_to_head():
    team_a = request.args.get("teamA")
    team_b = request.args.get("teamB")

    if not team_a or not team_b:
        return jsonify({"error": "Both 'teamA' and 'teamB' query params are required."}), 400

    # Find all matches where either team was home or away
    mask = (
        ((data['home_name'] == team_a) & (data['away_name'] == team_b)) |
        ((data['home_name'] == team_b) & (data['away_name'] == team_a))
    )
    h2h = data[mask].copy()

    if h2h.empty:
        return jsonify({"matches": [], "summary": {"teamA_wins": 0, "draws": 0, "teamB_wins": 0}})

    matches = []
    teamA_wins = draws = teamB_wins = 0

    for _, row in h2h.iterrows():
        result_code = int(row['result'])  # 2=home win, 1=draw, 0=away win

        if row['home_name'] == team_a:
            # team_a is home
            if result_code == 2:
                outcome = "A"   # team_a won
                teamA_wins += 1
            elif result_code == 1:
                outcome = "D"
                draws += 1
            else:
                outcome = "B"   # team_b won
                teamB_wins += 1
        else:
            # team_b is home
            if result_code == 2:
                outcome = "B"   # team_b (home) won
                teamB_wins += 1
            elif result_code == 1:
                outcome = "D"
                draws += 1
            else:
                outcome = "A"   # team_a (away) won
                teamA_wins += 1

        matches.append({
            "home":    row['home_name'],
            "away":    row['away_name'],
            "outcome": outcome,   # "A" = teamA won, "B" = teamB won, "D" = draw
        })

    return jsonify({
        "matches": matches,
        "summary": {
            "teamA_wins":  teamA_wins,
            "draws":       draws,
            "teamB_wins":  teamB_wins,
        }
    })


if __name__ == "__main__":
    # Runs on http://localhost:5000
    app.run(debug=True, port=5000)