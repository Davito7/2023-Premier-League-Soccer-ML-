# 2023 Premier League Soccer ML

A machine learning project that predicts the outcome of Premier League matches (home win, away win, or draw) using historical match data and a Random Forest classifier.

> **Note:** The model's current accuracy is limited as it does not yet account for several factors that influence match outcomes, such as player injuries, form, weather, and head-to-head history. This is an area for future improvement.

## How It Works
Match data is fetched from the [API-Football](https://www.api-football.com/) API and processed using pandas. A Random Forest model is trained on historical Premier League data to predict match outcomes. Predictions are served through a Flask API and displayed on a web interface.

## Libraries Used
- `pandas` - data processing
- `scikit-learn` - machine learning model
- `Flask` - API backend
- `requests` - fetching data from API-Football
- `joblib` - saving and loading the trained model

## File Breakdown
| File | Description |
|------|-------------|
| `train_model.py` | Trains the Random Forest model on historical match data |
| `data_fetcher.py` | Fetches match data from the API-Football API |
| `api.py` | Flask API that serves match predictions |
| `index.html` | Web interface for viewing predictions |
| `soccer_model.pkl` | The saved trained model |
| `training_data.csv` | Dataset used to train the model |

## Installation
Install the required libraries:
```bash
pip install pandas scikit-learn flask requests joblib
```

## How to Run
1. Fetch the latest match data:
```bash
python data_fetcher.py
```
2. Train the model:
```bash
python train_model.py
```
3. Start the Flask API:
```bash
python api.py
```
4. Open `index.html` in your browser to view predictions

## Future Improvements
- Account for player injuries and suspensions
- Include team form (last 5 matches)
- Factor in head-to-head history
- Include home/away performance stats
- Expand to other leagues