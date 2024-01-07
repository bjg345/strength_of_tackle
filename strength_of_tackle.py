from tabulate import tabulate
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Read in data
data_dir = '/kaggle/input/nfl-big-data-bowl-2024'
os.chdir(data_dir)
games = pd.read_csv('games.csv')
plays = pd.read_csv('plays.csv')
players = pd.read_csv('players.csv')
tackles = pd.read_csv('tackles.csv')
tracking_weeks = [pd.read_csv(f'tracking_week_{w}.csv') for w in range(1, 10)]
tracking = pd.concat(tracking_weeks, ignore_index=True)

# Define a function to find the closest players (to the ball carrier)
def find_closest_players(df, tracking, carrier_id, game_id, play_id, frame_id, num_players):
    # Filter the DataFrame for the specific game, play, and frame
    play_frame_data = tracking.loc[(game_id, play_id, frame_id), :]

    # Get the tackler ID
    tackler_id = df.loc[(game_id, play_id), ['tacklerId']].values[0]
    
    # Get the team of the player of interest
    player_team = play_frame_data.loc[play_frame_data['nflId'] == carrier_id, 'club'].values[0]

    # Split the data into offense and defense relative to the player of interest
    offense = play_frame_data[play_frame_data['club'] == player_team]
    defense = play_frame_data[(play_frame_data['club'] != player_team) & (play_frame_data['club'] != 'football')]
    data_dict = {'offense':offense, 'defense':defense}
    
    return_dict = {}
    for team in ['offense', 'defense']:
        distances = np.sqrt((data_dict[team]['x'] - offense.loc[offense['nflId'] == carrier_id, 'x'].values[0]) ** 2 +
                                (data_dict[team]['y'] - offense.loc[offense['nflId'] == carrier_id, 'y'].values[0]) ** 2)
        # Exclude the ball carrier
        distances[data_dict[team]['nflId'] == carrier_id] = np.inf
        # Exclude the tackler
        distances[data_dict[team]['nflId'] == tackler_id] = np.inf
        # Get the indices of the closest num_players players
        closest_indices = distances.nsmallest(num_players).index
        return_dict[team] = data_dict[team].loc[closest_indices, ['x', 'y']]
    
    return return_dict


# Get model dataframe from temp rows
def extract(row, frame_delay, tracking = tracking, players = players):
    play_id = row['playId']
    frame_id = row['frameId']
    game_id = row['gameId']
    tackler_id = row['tacklerId']
    carrier_id = row['ballCarrierId']

    start_frame = max(frame_id - frame_delay, 1)

    start_x = tracking.loc[(game_id, play_id, start_frame, carrier_id), 'x']
    end_x = tracking.loc[(game_id, play_id, frame_id, carrier_id), 'x']

    direction = row['playDirection']
    flip_x = 1 if direction == 'right' else -1
    extra_yards = (end_x - start_x)*flip_x     

    carrier_track = tracking.loc[(game_id, play_id, start_frame, carrier_id)]
    tackler_track = tracking.loc[(game_id, play_id, start_frame, tackler_id)]

    carrier_weight = players.loc[carrier_id, 'weight']
    tackler_weight = players.loc[tackler_id, 'weight']
    
    # engineer movement angles to be symmetric regardless of play direction
    carrier_angle = 90 - carrier_track['dir'] if direction == 'right' else 270 - carrier_track['dir']
    tackler_angle = 90 - tackler_track['dir'] if direction == 'right' else 270 - tackler_track['dir']
    
    # closest players
    off_closest_df = row['off_closest']
    def_closest_df = row['def_closest']
    
    # note that locations are calculated from the carrier's body as origin
    return pd.Series({
        'extra_yards': extra_yards,
        'carrier_weight': carrier_weight,
        'carrier_speed': carrier_track['s'],
        'carrier_angle': carrier_angle,
        'tackler_weight': tackler_weight,
        'tackler_speed': tackler_track['s'],
        'tackler_angle': tackler_angle,
        'tackler_x': (tackler_track['x'] - carrier_track['x'])*flip_x,
        'tackler_y': tackler_track['y'] - carrier_track['y'],
        'carrier_y': carrier_track['y'],
        'off_x_1': (off_closest_df.iloc[0]['x'] - carrier_track['x'])*flip_x,
        'off_y_1': off_closest_df.iloc[0]['y'] - carrier_track['y'],
        'off_x_2': (off_closest_df.iloc[1]['x'] - carrier_track['x'])*flip_x,
        'off_y_2': off_closest_df.iloc[1]['y'] - carrier_track['y'],
        'def_x_1': (def_closest_df.iloc[0]['x'] - carrier_track['x'])*flip_x,
        'def_y_1': def_closest_df.iloc[0]['y'] - carrier_track['y'],
        'def_x_2': (def_closest_df.iloc[1]['x'] - carrier_track['x'])*flip_x,
        'def_y_2': def_closest_df.iloc[1]['y'] - carrier_track['y'],
        'carrier_id': carrier_id,
        'tackler_id': tackler_id
    })


# Function to fit model and get results from analytical data set
def analyze_tackle_and_carry(df, players, features, param_grid, seed=213):
    # Reset index of the dataframe
    df.reset_index(drop=True, inplace=True)

    # Function to calculate tackle score
    def calculate_tackle_score(actual_yards, predictions, row_index):
        for percentile in sorted(predictions.keys()):
            if actual_yards <= predictions[percentile][row_index]:
                return 10 - int(percentile * 10)
        return 0

    # Split the data into features and labels
    X = df[features]
    y = df['extra_yards']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

    # Grid Search
    gbr = GradientBoostingRegressor(loss='quantile', alpha=0.5, random_state=seed)
    grid_search = GridSearchCV(gbr, param_grid, cv=KFold(5, shuffle=True, random_state=seed), 
                               n_jobs=-1, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    # Train models and predict
    models = {}
    full_predictions = {}
    for percentile in np.arange(0.1, 1.0, 0.1):
        model = GradientBoostingRegressor(loss='quantile', alpha=percentile, **best_params, random_state=seed)
        model.fit(X_train, y_train)
        models[percentile] = model
        full_predictions[percentile] = model.predict(X)

    # Calculate Tackle Score
    df['tackle_score'] = [calculate_tackle_score(row['extra_yards'], full_predictions, index) for index, row in df.iterrows()]
    df['carry_score'] = 9 - df['tackle_score']

    # Display results
    display_player_scores(df, players, 'tackler_id', 'tackle_score', "Strength of Tackle")
    display_player_scores(df, players, 'carrier_id', 'carry_score', "Strength of Carry")

    # Generate predictions for test set
    test_predictions = {}
    for percentile in [0.1, 0.5, 0.9]:
        model = models[percentile]
        test_predictions[percentile] = model.predict(X_test)

    # Calculate and print R^2 score
    r2 = r2_score(y_test, test_predictions[0.5])
    print(f"R^2 Score for 50th percentile predictions: {r2}")

    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, test_predictions[0.1], alpha=0.5, label='10th Percentile')
    plt.scatter(y_test, test_predictions[0.5], alpha=0.5, label='50th Percentile')
    plt.scatter(y_test, test_predictions[0.9], alpha=0.5, label='90th Percentile')
    plt.title("Scatter Plot of True vs. Predicted Yards While Tackled in Test Set")
    plt.xlabel("True Yards While Tackled")
    plt.ylabel("Predicted Yards While Tackled")
    plt.legend()
    plt.show()

def display_player_scores(df, players, player_id_column, score_column, score_description):
    # Filter out players with few events
    event_counts = df[player_id_column].value_counts()
    eligible_players = event_counts[event_counts >= 15].index

    # Calculate the average score for each eligible player
    avg_scores = df[df[player_id_column].isin(eligible_players)].groupby(player_id_column)[score_column].mean()

    # Create a DataFrame to store player details and average score
    player_details = players.set_index('nflId').loc[eligible_players, ['displayName', 'position', 'weight']]
    player_details[score_column] = avg_scores
    player_details = player_details.sort_values(by=score_column, ascending=False)

    # Drop 'nflId' from the DataFrame
    player_details = player_details.reset_index(drop=True)

    # Select the top 10 for highest and lowest average scores
    top_players = player_details.head(10)
    bottom_players = player_details.tail(10)

    # Format and print the tables
    print(f"Top 10 Players with the Highest Average {score_description}:")
    print(tabulate(top_players, headers='keys', tablefmt='github', showindex=False))
    print(f"\nTop 10 Players with the Lowest Average {score_description}:")
    print(tabulate(bottom_players, headers='keys', tablefmt='github', showindex=False))

# Filter the tracking DataFrame for the frames where a tackle occurs
tackle_frames = tracking.query('event == "tackle"')

# In case there are multiple tackles, just take first one
tackle_frames = tackle_frames.drop_duplicates(subset=['gameId', 'playId'])

# Get rid of penalized plays
temp = plays[(plays['penaltyYards'].isna()) & (plays['playNullifiedByPenalty'] == "N")]

# Merge games and plays
temp = temp.merge(games, on='gameId')

# Merge games, plays, and tackles
temp = temp.merge(tackles, on=['gameId', 'playId'], how='inner')

# Look at only tackles and not forced fumbles
temp = temp.query('tackle == 1 and forcedFumble == 0')

# Make column name clearer
temp.rename(columns={'nflId': 'tacklerId'}, inplace=True)

# Merge the tackle_frames DataFrame with the temp DataFrame to get the frameId of the tackle
temp = temp.merge(tackle_frames, on=['gameId', 'playId'], how='inner')

# Set indices for faster queries
tracking.set_index(['gameId', 'playId', 'frameId', 'nflId'], inplace=True, drop=False)
tracking.sort_index(inplace=True)
players.set_index('nflId', inplace=True, drop=False)
players.sort_index(inplace=True)
temp.set_index(['gameId', 'playId'], inplace=True, drop=False)
temp.sort_index(inplace=True)


num_players = 2  # get these many closest players on offense and defense (each)
frame_delay = 10 # look back this many frames from moment of tackle

# get closest players
temp[['off_closest', 'def_closest']] = temp.apply(
    lambda row: pd.Series(find_closest_players(temp, tracking, row['ballCarrierId'], row['gameId'], row['playId'], max(row['frameId'] - frame_delay, 1), num_players)),
    axis=1
)

# Create the analysis DataFrame
df = temp.apply(
    extract,
    axis=1,
    frame_delay = frame_delay
)

# Run the defensive analysis
features_to_use = ['carrier_weight', 'carrier_speed', 'tackler_speed',
                    'tackler_x', 'tackler_y', 'carrier_y',
                'tackler_angle', 'carrier_angle', 'off_x_1', 'off_y_1', 'off_x_2', 'off_y_2', 
                'def_x_1', 'def_y_1', 'def_x_2', 'def_y_2']
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
}
analyze_tackle_and_carry(df, players, features_to_use,  param_grid)


# Run the offensive analysis
features_to_use = ['tackler_weight', 'carrier_speed', 'tackler_speed',
                    'tackler_x', 'tackler_y', 'carrier_y',
                'tackler_angle', 'carrier_angle', 'off_x_1', 'off_y_1', 'off_x_2', 'off_y_2', 
                'def_x_1', 'def_y_1', 'def_x_2', 'def_y_2']
analyze_tackle_and_carry(df, players, features_to_use,  param_grid)