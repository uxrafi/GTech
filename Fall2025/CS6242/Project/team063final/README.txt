# 🏀 NBA Game Predictor — Local Setup Guide

This README explains how to run the full **NBA Predictor** locally so anyone cloning the repo can reproduce the exact interactive webpage and backend predictions.

The project consists of:
* **`frontend/`** → D3.js interactive NBA comparison + predictor page
* **`backend/`** → Flask API that loads the trained model + returns win probability
* **`data/`** → CSV data files used for charts + backend
* **`model/`** → Trained machine learning model (RandomForest)

Everything runs locally with **no cloud dependencies**.

---
## DESCRIPTION 

A machine learning-powered web application that predicts NBA game outcomes using historical team performance data. Built with a Random Forest classifier trained on rolling 5-10 game averages, the app features an interactive D3.js visualization comparing team statistics through radar charts, momentum indicators, and key matchup insights. Users select two teams and receive real-time win probability predictions via a Flask API backend.

---

### 📁 Project Structure

Your repo should look like this:

```text
nba_game_predictor/
│
├── team63final.ipynb
│ 
├── backend/
│   ├── backend_api.py
│   ├── best_nba_model_full.pkl           ← trained model
│   ├── full_feature_list.pkl
│   └── requirements.txt
│
├── data/
│   ├── team_rolled_top1000.csv
│   └── teams.csv
│
├── outputs/
│   └── nba_comparison.html              ← D3 frontend
│
└── README.md
```

## INSTALLATION
### 🔧 1. Install Requirements

You need **Python 3.10+**, **Flask**, **scikit-learn 1.7.x**, **pandas**, and **joblib**.

Inside the `backend/` directory, run:

```bash
cd backend
pip install -r requirements.txt
```
---
## EXECUTION
### 🚀 2. Start the Backend API

Navigate into the backend folder:

```bash
cd backend
python backend_api.py
```

If successful, you should see:

```
 * Running on http://127.0.0.1:5000
```

The API now supports:

### ⭐ POST `/predict`

**Request body:**

```json
{
  "home_team": "Los Angeles Lakers",
  "away_team": "Boston Celtics"
}
```

**Response:**

```json
{
  "home": "Los Angeles Lakers",
  "away": "Boston Celtics",
  "home_win_prob": 0.62,
  "predicted_home_win": 1
}
```

## 🌐 3. Launch the Frontend (D3 HTML Page)

A web server is required to serve the frontend. **Make sure to run this from the repo root directory**, then start a local web server:

```bash
python -m http.server
```

Then open your browser and navigate to:

```
http://localhost:8000/outputs/nba_comparison.html
```

The frontend automatically:

- Loads team averages from `/data/ModelDFtop1000.csv`
- Loads team metadata from `/data/teams.csv`
- Sends POST requests to the backend at `http://127.0.0.1:5000/predict`
- Draws radar charts, matchup insights, trends, and model prediction
---
## 🔥 4. Using the Predictor

Once both are running:

1. Select two teams from dropdowns
2. Click Predict Game Outcome
3. Page shows:
   - Predicted winner
   - Win probability
   - Momentum bar
   - Key matchup edges (feature differences)
   - Trend arrows
   - Radar chart(s)

---
## 🙌 Credits

Team 63 – Georgia Tech CSE 6242 NBA ML Predictor + Feature Visualization

Built with:
- Python, Flask
- scikit-learn
- D3.js
- Real NBA game logs
