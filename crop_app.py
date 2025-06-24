# train_model.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load the dataset
df = pd.read_csv("Crop_recommendation.csv")  # Make sure this CSV is in the same folder

# Step 2: Split into features and target
X = df.drop("label", axis=1)
y = df["label"]

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Save the model
with open("crop_model.pkl", "wb") as file:
    pickle.dump(model, file, protocol=4)

print("✅ Model trained and saved as crop_model.pkl")
