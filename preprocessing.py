import pandas as pd
from graph import PlayersGraph


from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, lit


def flatten_players_movement_data(player):
    # Replace both '[' and ']' characters with an empty string
    expression = F.regexp_replace(col("splitted")[player], "[\[\]]", "")

    # Split the resulting string using ','
    expression = F.split(expression, ",")
    return expression


def get_player_movement_data(moments_data):
    players_names = []
    players_x_coord = []
    players_y_coord = []
    players_team = []

    for player in range(0,11):
        flattened_player = flatten_players_movement_data(player=player)
        if player == 0: player_name = 'ball' 
        else: player_name = player

        players_names.append(flattened_player[1].alias(f"player{player_name}_name"))
        players_x_coord.append(flattened_player[2].cast("float").alias(f"player{player_name}_x_coord"))
        players_y_coord.append(flattened_player[3].cast("float").alias(f"player{player_name}_y_coord"))
        players_team.append(flattened_player[0].alias(f"player{player_name}_team"))


    player_movement_data = (
        moments_data
        .withColumn("playerposition",col("moments")[5])
        .withColumn("splitted",F.split("playerposition", r"\],\["))
        .select(
            col("moments")[0].cast("integer").alias("quarter"),
            col("moments")[2].cast("float").alias("game_clock"),
            col("moments")[3].cast("float").alias("shot_clock"),
            *players_names, 
            *players_x_coord, 
            *players_y_coord,
            *players_team
            )
    )

    return player_movement_data


def cleansing_score_data(score_data):

    ## Drop unnesary rows (clock stopped)
    score_data  = score_data.drop_duplicates(['quarter_score', 'secleft'], keep="last")
    score_data['Secleft_end'] = score_data['secleft'].shift(-1)

    ## Keep only necesary columns
    cols_keep = ["quarter_score", "secleft", "Secleft_end", "awayscore", "homescore", "AwayPlay", "HomePlay", "Shooter", 'ShotOutcome', "FoulType", "ReboundType", "FreeThrowNum", 'FreeThrowOutcome', "TurnoverType", "FreeThrowNum", "EnterGame", "TimeoutTeam"]
    score_data = score_data[cols_keep]

    ## Combine Shot anf Free Throw shots
    score_data['AnyShotOutcome'] = score_data['ShotOutcome'].combine_first(score_data['FreeThrowOutcome'])

    return score_data


def create_posessions(score_data):
    cond_away_shot_made = score_data['AwayPlay'].notnull() & (score_data['AnyShotOutcome'] == 'make')
    cond_away_shot_miss_def = score_data['AwayPlay'].notnull() & (score_data['ReboundType'] == 'defensive')
    cond_away_shot_miss_of = score_data['AwayPlay'].notnull() & (score_data['ReboundType'] == 'offensive')
    cond_away_turnover = score_data['AwayPlay'].notnull() & (score_data['TurnoverType'].notnull())
    cond_away_pers_foul = score_data['AwayPlay'].notnull() & (score_data['FoulType'] == 'personal')


    cond_home_shot_made = score_data['HomePlay'].notnull() & (score_data['AnyShotOutcome'] == 'make')
    cond_home_shot_miss_def = score_data['HomePlay'].notnull() & (score_data['ReboundType'] == 'defensive')
    cond_home_shot_miss_of = score_data['HomePlay'].notnull() & (score_data['ReboundType'] == 'offensive')
    cond_home_turnover = score_data['HomePlay'].notnull() & (score_data['TurnoverType'].notnull())
    cond_home_pers_foul = score_data['HomePlay'].notnull() & (score_data['FoulType'] == 'personal')

    score_data.loc[cond_away_shot_made, 'posession'] = 'home'
    score_data.loc[cond_away_shot_miss_def, 'posession'] = 'away'
    score_data.loc[cond_away_shot_miss_of, 'posession'] = 'away'
    score_data.loc[cond_away_turnover, 'posession'] = 'home'
    score_data.loc[cond_away_pers_foul, 'posession'] = 'away'

    score_data.loc[cond_home_shot_made, 'posession'] = 'away'
    score_data.loc[cond_home_shot_miss_def, 'posession'] = 'home'
    score_data.loc[cond_home_shot_miss_of, 'posession'] = 'home'
    score_data.loc[cond_home_turnover, 'posession'] = 'away'
    score_data.loc[cond_home_pers_foul, 'posession'] = 'home'

    cond_enter_game = score_data['EnterGame'].notnull() | score_data['TimeoutTeam'].notnull()
    score_data['previous_posession'] = score_data['posession'].shift(1)

    score_data.loc[cond_enter_game, 'posession'] = score_data['previous_posession']

    return score_data

def create_value_x_posession(score_data):
    
    cols_keep = ['quarter_score', 'secleft', 'Secleft_end', 'awayscore', 'homescore','posession']
    score_data = score_data[cols_keep]

    score_data['y_away'] = score_data['awayscore'].shift(-1) - score_data['awayscore']
    score_data['y_home'] = score_data['homescore'].shift(-1) - score_data['homescore']

    return score_data





def create_posession_id(data: pd.DataFrame):
    # Identify changes in possession
    data['new_possession'] = data['posession_team'] != data['posession_team'].shift(1)

    # Assign a possession ID
    data['possession_id'] = data['new_possession'].cumsum()

    # Calculate the duration of each possession period
    data['next_game_clock'] = data['game_clock'].shift(-1)
    data['possession_duration'] = data['game_clock'] - data['next_game_clock']

    # Handle the last possession for each team
    data.loc[data['new_possession'].shift(-1, fill_value=True), 'possession_duration'] = None

    # Drop intermediate columns
    data = data.drop(columns=['new_possession', 'next_game_clock'])

    return data


def clean_short_posessions(data: pd.DataFrame, limit: float=.4):
    length_posessions = data.groupby(['possession_id']).agg({'possession_duration':'sum'}).reset_index()
    filter_very_short_posession = length_posessions[length_posessions['possession_duration'] < limit].drop("possession_duration",axis=1)

    filter_very_short_posession['change'] = 1

    data_cleaned_posessions = data.merge(filter_very_short_posession, how="left", on="possession_id")

    data_cleaned_posessions.loc[data_cleaned_posessions['change'] == 1, 'posession_team'] = None

    data_cleaned_posessions['posession_team'] = data_cleaned_posessions['posession_team'].ffill()

    data_cleaned_posessions = data_cleaned_posessions.drop('change',axis=1)

    return data_cleaned_posessions

def get_posession_team(data, teams_list):
    posession_team = []
    for idx, row in data.iterrows():
        # Create an empty graph
        PG = PlayersGraph(input_data=row, team_list=teams_list)

        PG.create_nodes()

        PG.add_ball_node()

        PG.add_closest_player_to_ball_edge()

        PG.set_posession_team()

        posession_team.append(PG.get_posession_team())

    return posession_team


def drop_null_in_players(data):
    columns_to_check = ['player1_name', 'player2_name', 'player3_name', 'player4_name', 'player5_name', 'player6_name', 'player7_name', 'player8_name', 'player9_name', 'player10_name']

    # Check if all columns are not null for each row
    mask = data[columns_to_check].notnull().all(axis=1)

    # Filter the DataFrame based on the mask
    data_wo_empty_posessions = data[mask]

    return data_wo_empty_posessions
