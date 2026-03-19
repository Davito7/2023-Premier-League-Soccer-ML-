import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("training_data.csv")

# X = Features (IDs), y = Target (Result)
X = df[['home_id', 'away_id']] 
y = df['result']

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, "soccer_model.pkl")
print("soccer_model.pkl trained and saved!")