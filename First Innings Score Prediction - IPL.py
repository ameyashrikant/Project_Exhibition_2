import pandas as pd
import pickle

#loadind dataset
matches = pd.read_csv('ipl 1.csv')

# --- Data Cleaning ---
# Removing unwanted columns
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
matches.drop(labels=columns_to_remove, axis=1, inplace=True)
matches.replace(to_replace ="Delhi Daredevils", value = "Delhi Capitals", inplace = True)
# Keeping only current teams
current_teams = ['Chennai Super Kings', 'Delhi Capitals' , 'Kings XI Punjab', 
                 'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
                 'Royal Challengers Bangalore', 'Sunrisers Hyderabad','Gujarat Titans', 'Lucknow Super Giants']

matches = matches[(matches['bat_team'].isin(current_teams)) & (matches['bowl_team'].isin(current_teams))]

# Removing the first 6 overs data in every match
matches = matches[matches['overs']>=6.0]

# Converting the column 'date' from string into datetime object
from datetime import datetime
matches['date'] = matches['date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))

# --- Data Preprocessing ---
# Converting categorical features using OneHotEncoding method
match_df = pd.get_dummies(data=matches, columns=['bat_team', 'bowl_team'])

match_df = match_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Capitals', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad','bat_team_Gujarat Titans','bat_team_Lucknow Super Giants',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Capitals', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad','bowl_team_Gujarat Titans','bowl_team_Lucknow Super Giants',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]

# Splitting the data into train and test set
X = match_df.drop(labels='total',axis=1)
X_train = match_df.drop(labels='total',axis=1)[match_df['date'].dt.year <= 2022]

y = match_df[match_df['date'].dt.year <= 2022]['total'].values
y_train = match_df[match_df['date'].dt.year <= 2022]['total'].values

# Removing the 'date' column
X.drop(labels='date',axis=True, inplace=True)
X_train.drop(labels='date',axis=True, inplace=True)

# --- Model Building ---

# Linear Regression Model
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train,y_train)

# Creating a pickle file for the classifier
filename = 'first-innings-score-lr-model-linear.pkl'
pickle.dump(linear_regressor, open(filename, 'wb'))

# Ridge Regression Model
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge = Ridge()
parameters={'alpha':[1.25,3.56,7.26,10.85,14.95,18.99]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=10)
ridge_regressor.fit(X_train,y_train)


# Creating a pickle file for the classifier
file_name = 'first-innings-score-lr-model-ridge.pkl'
pickle.dump(ridge_regressor, open(file_name, 'wb'))


# Random Forest Regression Model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Creating a pickle file for the classifier
file_name = 'first-innings-score-lr-model-rf.pkl'
pickle.dump(regressor, open(file_name,'wb'))

# Articial Neural Network
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(50,60,70,80,90,100), activation="logistic", alpha=0.0056, max_iter=1000, warm_start=True, max_fun=1007)
mlp.fit(X_train,y_train)

# Creating a pickle file for the classifier
file_name = 'first-innings-score-lr-model-ann.pkl'
pickle.dump(regressor, open(file_name,'wb'))
