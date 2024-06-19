import json
import os
import sys
import pickle
import pandas as pd
# import sklearn
# import numpy as np

# Ensure the script receives the correct file path as an argument
if len(sys.argv) != 2:
    print("Usage: python3 main.py path/to/output/file.json")
    sys.exit(1)

file_path = sys.argv[1]

# Load the models
all_nba_model_path = 'pipeline/Model_all_nba.pkl'
all_rookie_model_path = 'pipeline/Model_all_rookie.pkl'


class ReplaceLowGTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.loc[X['G'] < 65, ['eFG%', 'FG', 'FGA', 'FG%', '3P',
                            '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB',
                            'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']] = 0
        return X
with open(all_nba_model_path, 'rb') as file:
    model_nba = pickle.load(file)

with open(all_rookie_model_path, 'rb') as file:
    model_rookie = pickle.load(file)

# Load and prepare the data
all_nba_data = pd.read_csv("./csv/all_nba_cleaned.csv")
all_rookie_data = pd.read_csv("./csv/all_rookies_cleaned.csv")
all_nba_data = all_nba_data.drop(["Unnamed: 0"], axis=1)
all_rookie_data = all_rookie_data.drop(["Unnamed: 0"], axis=1)
all_nba_data = all_nba_data.dropna(subset=['Pos'])
all_nba_data = all_nba_data.fillna(0)
all_rookie_data = all_rookie_data.fillna(0)

# Define the features to be used for predictions
p_all_nba = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'season']
p_all_rookie = ['Age', 'G', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'ORB', 'TOV', 'PF', 'FG%', '3P%', 'FT%', 'MP.1', 'PTS.1', 'TRB.1', 'AST.1', 'STL.1', 'BLK.1', 'season']

# Filter data for the 2022 season
test_all_nba = all_nba_data[all_nba_data["season"] == 2024]
test_all_rookie = all_rookie_data[all_rookie_data["season"] == 2024]

# Prepare the feature sets for prediction
X_all_nba = test_all_nba[p_all_nba]
X_all_rookie = test_all_rookie[p_all_rookie]

# Make predictions
predykcja_all_nba = model_nba.predict(X_all_nba)
predykcja_all_rookie = model_rookie.predict(X_all_rookie)

# Convert predictions to DataFrames and sort by prediction scores
predykcja_all_nba = pd.DataFrame(predykcja_all_nba, columns=["Wynik all nba"], index=test_all_nba.index)
predykcja_all_rookie = pd.DataFrame(predykcja_all_rookie, columns=["Wynik all rookie"], index=test_all_rookie.index)

wyniki_all_nba = pd.concat([test_all_nba[["Player"]], predykcja_all_nba], axis=1).sort_values("Wynik all nba", ascending=False)
wyniki_all_rookie = pd.concat([test_all_rookie[["Player"]], predykcja_all_rookie], axis=1).sort_values("Wynik all rookie", ascending=False)

# Create the output data structure
data_out = {
    "first all-nba team": wyniki_all_nba.head(5)["Player"].tolist(),
    "second all-nba team": wyniki_all_nba.iloc[5:10]["Player"].tolist(),
    "third all-nba team": wyniki_all_nba.iloc[10:15]["Player"].tolist(),
    "first rookie all-nba team": wyniki_all_rookie.head(5)["Player"].tolist(),
    "second rookie all-nba team": wyniki_all_rookie.iloc[5:10]["Player"].tolist()
}

# Ensure the output directory exists
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Write the output data to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(data_out, json_file, ensure_ascii=False, indent=2)
