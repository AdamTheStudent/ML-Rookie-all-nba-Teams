Projekt WZUM Predykcji wynikow NBA dla all-nba-teams i all-rookie-teams

Projekt wykorzystuje stworzone modele do predykcji piatek najlepszych zawodnikow ligi NBA, oraz najlepszych piatek debiutantow.

W projekcie znajduje sie plik requirements.txt z wymaganymi bibliotekami, ktore mozna uruchomic poleceniem:

$ pip install -r ./requirements.txt

Glowna aplikacje 'Adam_Nawrocki.py' mozna uruchomic poleceniem z przekazeniem argumetnu scieszki do nieistniejacego pliku '.json':

$ python3 Adam_Nawrocki.py ./path/output.json

Program pobiera pliki .csv z folderu 'csv' z baza danych zawodnikow i debiutantow. Wykorzstuje ona rowniez modele zapisane w folderze 'pipeline'.
W podamym jako argument nieistniejacym pliku '.json' zostanie stworzony plik zawierajacy nazwiska predykcji w formacie:
{
  "first all-nba team": [
    "Luka Dončić",
    "Shai Gilgeous-Alexander",
    "LeBron James",
    "Nikola Jokić",
    "Giannis Antetokounmpo"
  ],
  "second all-nba team": [
    "Anthony Edwards",
    "Kevin Durant",
    "Anthony Davis",
    "Devin Booker",
    "Jayson Tatum"
  ],
  "third all-nba team": [
    "Stephen Curry",
    "Jalen Brunson",
    "Kawhi Leonard",
    "Tyrese Haliburton",
    "De'Aaron Fox"
  ],
  "first rookie all-nba team": [
    "Victor Wembanyama",
    "Chet Holmgren",
    "Brandon Miller",
    "Amen Thompson",
    "Scoot Henderson"
  ],
  "second rookie all-nba team": [
    "Brandin Podziemski",
    "GG Jackson II",
    "Jaime Jaquez Jr.",
    "Keyonte George",
    "Dereck Lively II"
  ]
}

Pliki NBA-zdobycie-danych.ipynb oraz NBA-czysczenie-danych.ipynb sa plikami notatnika jupyter, ktore posluzuly do zdobycia danych 
zawodnikow z strony www.basketball-reference.com oraz glosowan oraz przetworzenie plikow .csv z nie potrzebnych danych.

Notatnik o nazwie testowanie_modeli.ipynb wykorzystano do przetestowania regresorow, modeli i pipelinow oraz zapisania wybranych.
Glowna metryka, wykorzystana do wybrania najlepszego rozwiazania wybrano Compute Normalized Discounted Cumulative Gain.

'from sklearn.metrics import ndcg_score

ndcg = ndcg_score([true_scores], [predicted_scores], k=15)'

dla all_nba_team oraz 

'ndcg = ndcg_score([true_scores], [predicted_scores], k=10)'

dla all_rookie

Wybrany pipeline dla all_nba_team:

from sklearn.ensemble import GradientBoostingRegressor
pipeline = Pipeline(steps=[
    ('replace_low_g', ReplaceLowGTransformer()),  
    ('scaler', StandardScaler()),  
    ('regressor', GradientBoostingRegressor())  
])

class ReplaceLowGTransformer:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X.loc[X['G'] < 65, ['eFG%','FG', 'FGA', 'FG%', '3P',
       '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB',
       'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']] = 0
        return X
        
Wybrany pipeline dla all_rookie_team:

pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),  
    ('regressor', GradientBoostingRegressor())  
])


Plik output.json zawiera to wynik predykcji po uruchomieniu programu za pomoca polecenia:

python3 Adam_Nawrocki.py ./output.json



