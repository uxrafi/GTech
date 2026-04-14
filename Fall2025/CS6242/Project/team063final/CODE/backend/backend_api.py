from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Paths (edit here if needed)
MODEL_PATH = "best_nba_model_full.pkl"
FEATURE_LIST_PATH = "full_feature_list.pkl"
ROLL_PATH = "../data/team_rolled_top1000.csv"
TEAMS_PATH = "../data/teams.csv"

# Load model + features
model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURE_LIST_PATH) if os.path.exists(FEATURE_LIST_PATH) else None

# Load data
roll = pd.read_csv(ROLL_PATH)
teams = pd.read_csv(TEAMS_PATH)

teams["display"] = (teams["CITY"].fillna("") + " " + teams["NICKNAME"].fillna("")).str.strip()
display_to_id = dict(zip(teams["display"], teams["TEAM_ID"]))

# parse dates for sorting
if "game_date" in roll.columns:
    roll["game_date"] = pd.to_datetime(roll["game_date"], errors="coerce")

# Helper: latest rolled row
def latest_team_row(team_id: str):
    sub = roll[roll["team_id"].astype(str) == str(team_id)]
    if sub.empty:
        return None
    sub = sub.sort_values("game_date")
    return sub.iloc[-1]

# Auto-category grouping keys
ENDURANCE_KEYS = ["rest_days", "is_back_to_back", "games_in_last_5_days"]
MOMENTUM_KEYS = ["win_rate", "win_prev", "recent_form", "point_diff", "current_win_pct"]
HUSTLE_KEYS = ["reb", "turnovers"]
OFFENSE_KEYS = ["pts", "fg_pct", "fg3_pct", "efg_pct", "fgm", "fg3m", "ftm", "ast"]

ROLL_NUMERIC_COLS = roll.select_dtypes(include=["int64", "float64"]).columns.tolist()

# normalize each numeric col to 0–1 for category averaging
col_min = roll[ROLL_NUMERIC_COLS].min()
col_max = roll[ROLL_NUMERIC_COLS].max()

def normalize(col, val):
    mn, mx = col_min[col], col_max[col]
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return 0.5
    return (val - mn) / (mx - mn)

def build_category_columns():
    cats = {
        "Offense": [],
        "Defense": [],
        "Momentum": [],
        "Endurance": [],
        "Hustle": []
    }

    for c in ROLL_NUMERIC_COLS:
        name = c.lower()

        # defense = opponent columns
        if name.startswith("opp_"):
            cats["Defense"].append(c)
            continue

        if any(k in name for k in ENDURANCE_KEYS):
            cats["Endurance"].append(c)
            continue

        if any(k in name for k in MOMENTUM_KEYS):
            cats["Momentum"].append(c)
            continue

        if any(k in name for k in HUSTLE_KEYS):
            cats["Hustle"].append(c)
            continue

        if any(k in name for k in OFFENSE_KEYS):
            cats["Offense"].append(c)
            continue

    return cats


CATEGORY_COLS = build_category_columns()

CATEGORY_DESCRIPTIONS = {
    "Offense": "How well a team scores and generates shots (shooting %, points, assists, shot-making).",
    "Defense": "How well a team limits opponents (opponent scoring, opponent shooting %, forced mistakes).",
    "Momentum": "How a team has been trending recently (recent win rate, point differential, form).",
    "Endurance": "Fatigue/rest advantage (rest days, back-to-backs, workload).",
    "Hustle": "Effort and control plays (rebounds, turnovers, extra possessions).",
}


def category_scores(team_row):
    """Return 0-100 scores per category from latest rolled row."""
    scores = {}
    for cat, cols in CATEGORY_COLS.items():
        if not cols:
            scores[cat] = 50.0
            continue

        vals = []
        for c in cols:
            if c in team_row.index and pd.notna(team_row[c]):
                vals.append(normalize(c, float(team_row[c])))

        scores[cat] = float(np.mean(vals) * 100) if vals else 50.0

    return scores


def representative_raw_stats(team_row):
    """
    Return small raw stats subset per category for transparency.
    We show first 5 columns per category as "example drivers".
    """
    raw = {}
    for cat, cols in CATEGORY_COLS.items():
        keep = []
        for c in cols:
            # skip obvious IDs if any slipped in
            if "id" in c.lower() or "season" in c.lower():
                continue
            keep.append(c)
            if len(keep) == 5:
                break

        raw[cat] = {k: float(team_row.get(k, 0)) for k in keep}

    return raw


def build_matchup_features(home_row, away_row):
    """
    Map training feature names -> rolled stats columns.
    """
    x = {}

    for f in feature_cols:
        # home features
        if f.startswith("home_team_"):
            key = f.replace("home_team_", "team_")
            x[f] = float(home_row.get(key, 0))

        elif f.startswith("home_opp_"):
            key = f.replace("home_opp_", "opp_")
            x[f] = float(home_row.get(key, 0))

        elif f.startswith("home_"):
            key = f.replace("home_", "")
            x[f] = float(home_row.get(key, 0))

        # away features
        elif f.startswith("away_team_"):
            key = f.replace("away_team_", "team_")
            x[f] = float(away_row.get(key, 0))

        elif f.startswith("away_opp_"):
            key = f.replace("away_opp_", "opp_")
            x[f] = float(away_row.get(key, 0))

        elif f.startswith("away_"):
            key = f.replace("away_", "")
            x[f] = float(away_row.get(key, 0))

        else:
            if f in home_row.index:
                x[f] = float(home_row.get(f, 0))
            elif f in away_row.index:
                x[f] = float(away_row.get(f, 0))
            else:
                x[f] = 0.0

    X = pd.DataFrame([x], columns=feature_cols)
    return X


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # allow GET or POST
    if request.method == "POST":
        body = request.get_json(force=True)
        home_name = body.get("home_team")
        away_name = body.get("away_team")
    else:
        home_name = request.args.get("home")
        away_name = request.args.get("away")

    if home_name not in display_to_id or away_name not in display_to_id:
        return jsonify({"error": "team not found"}), 404

    home_id = display_to_id[home_name]
    away_id = display_to_id[away_name]

    h = latest_team_row(home_id)
    a = latest_team_row(away_id)

    if h is None or a is None:
        return jsonify({"error": "missing rolled stats for team"}), 404

    # model input
    X = build_matchup_features(h, a)

    prob_home = float(model.predict_proba(X)[0, 1])
    pred_home = int(prob_home >= 0.5)

    # confidence = scaled distance from 0.5 (0=coinflip, 1=very confident)
    confidence = float(abs(prob_home - 0.5) * 2)

    cats_home = category_scores(h)
    cats_away = category_scores(a)

    # contributions per category (positive = home advantage)
    contrib = {k: float(cats_home[k] - cats_away[k]) for k in cats_home.keys()}

    raw_home = representative_raw_stats(h)
    raw_away = representative_raw_stats(a)

    return jsonify({
        "home": home_name,
        "away": away_name,
        "home_win_prob": prob_home,
        "predicted_home_win": pred_home,
        "confidence": confidence,
        "categories_home": cats_home,
        "categories_away": cats_away,
        "contributions": contrib,
        "raw_home": raw_home,
        "raw_away": raw_away,
        "category_descriptions": CATEGORY_DESCRIPTIONS
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)
