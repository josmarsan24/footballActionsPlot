import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mplsoccer import VerticalPitch, Pitch
from math import sqrt

CONFIG_PATH = 'db_config.txt'

c1 = '#181818'
c2 = '#E8E8E8'
c3 =  '#00FF00'
c4 = '#FF0000'

def dict_games():
    path = 'data'
    d = {}
    for file in Path(path).iterdir():
         if file.is_file():
              try:
                d[file.name.split('En Directo ')[1].split(' - ')[0]] = file.name
              except:
                d[file.name] = file.name
    return d

def read_df(path):
    df = pd.read_excel('data/' + path)
    if 'type_value_OwnGoal' in df.columns:
        df.loc[df['type_value_OwnGoal'] == 28, 'teamName'] = df['oppositionTeamName']
        df.loc[df['type_value_OwnGoal'] == 28, 'x'] = 105 - df['x']
        df.loc[df['type_value_OwnGoal'] == 28, 'y'] = 68 - df['y']
    return df

def percentage_try(a,b):
  try:
    p = 100*a/(a + b)
  except:
    p = 0
  return p

def plot_player_actions(df, team, player, action, period):
    if period:
        df = df.loc[df.period == period]
    if action == 'Passes':
        plot_passes(df, team, player)
    elif action == 'Shots':
        plot_shots(df, team, player)
    elif action == 'Dribbles':
        plot_dribbles(df, team, player)
    elif action == 'Carries':
        plot_carries(df, team, player)
    elif action == 'Def. Actions':
        plot_def_actions(df, team, player)
    elif action == 'Starting XI':
        plot_XI(df, team)
    elif action == 'Pass map':
        plot_pass_map(df, team)
    elif action == 'xT by minute':
        plot_XT(df)
    elif action == 'Touches by zone':
        plot_touch_zones(df, team)

def plot_passes(df, team, player):
    pitch = Pitch(pitch_type='custom', pitch_width=68, pitch_length=105, pitch_color=c1, line_color=c2, line_zorder=1, linewidth=1, spot_scale = 0.00)
    fig, ax = pitch.draw(figsize=(16, 9))

    df = df.loc[(df['type'] == 'Pass') & (df['teamName'] == team), ['x', 'y', 'endX', 'endY','outcomeType','teamName','oppositionTeamName','name','type_value_KeyPass']]
    opp = df.iloc[0]['oppositionTeamName']
    name = team
    if player:
        df = df.loc[df['name'] == player]
        name = player
    
    title = name + ' passes vs ' + opp
    plt.title(title, color='black', ha='center', y=1.05, fontsize=30)

    succ = df.loc[df.outcomeType=='Successful']
    unsucc = df.loc[df.outcomeType!='Successful']
    xLastThird = (2 * 105)/3
    LastThird = len(succ.loc[(succ.outcomeType=='Successful') & (succ.endX >= xLastThird)])

    penalty_area_start_x = 105 - 16.5
    penalty_area_end_x = 105
    penalty_area_start_y = (68 - 40.3) / 2
    penalty_area_end_y = penalty_area_start_y + 40.3

    penalty = len(succ.loc[
        (df['endX'] >= penalty_area_start_x) &
        (df['endX'] <= penalty_area_end_x) &
        (df['endY'] >= penalty_area_start_y) &
        (df['endY'] <= penalty_area_end_y)])

    try:
        kp = len(df.loc[df.type_value_KeyPass==11113])
    except:
        kp = 0

    supltitle = str(len(succ)) + '/'+ str(len(df)) + ' (' + str(round(100*len(succ)/len(df), 2)) + '%) passes ' + str(kp) + ' key passes ' + str(LastThird) + ' into final third ' + str(penalty) + ' into the penalty area'
    plt.suptitle(supltitle, color='#3A3B3C', ha='center', y=0.9, fontsize=10)

    pitch.arrows(succ['x'],succ['y'],succ['endX'],succ['endY'], width=2, headwidth=10, headlength=10, color=c3, ax=ax)
    pitch.arrows(unsucc['x'],unsucc['y'],unsucc['endX'],unsucc['endY'], width=2, headwidth=10, headlength=10, color=c4, ax=ax)
    st.pyplot(fig)

def plot_shots(df, team, player):

    pitch = VerticalPitch(half=True, pitch_type='custom', pitch_width=68, pitch_length=105, pitch_color=c1, line_color=c2, line_zorder=1, linewidth=1, spot_scale = 0.00)
    fig, ax = pitch.draw(figsize=(16, 9))

    df = df.loc[(df['teamName']==team) & ((df['type']=='MissedShots') | (df['type']=='SavedShot')  | (df['type']=='Goal') | (df['type']=='ShotOnPost'))][['type','x','y','teamName','playerId','oppositionTeamName','name','cumulative_mins']]

    opp = df.iloc[0]['oppositionTeamName']
    name = df.iloc[0]['teamName']
    if player:
        df = df.loc[df['name'] == player]
        name = player
    
    title = name + ' shots vs ' + opp
    plt.title(title, color='black', ha='center', y=1.05, fontsize=30)

    pitch.draw(ax=ax)
    succ = df.loc[df['type']=='Goal']
    unsucc = df.loc[df['type']!='Goal']
    
    pitch.scatter(unsucc.x, unsucc.y, edgecolors='#606060', c=c4, marker='o', ax=ax)
    pitch.scatter(succ.x, succ.y, edgecolors='#606060', c=c3, marker='o', ax=ax)

    st.pyplot(fig)

def plot_def_actions(df, team, player):
    pitch = Pitch(pitch_type='custom', pitch_width=68, pitch_length=105, pitch_color=c1, line_color=c2, line_zorder=1, linewidth=1, spot_scale = 0.00)
    fig, ax = pitch.draw(figsize=(16, 9))

    df = df.loc[(df.teamName == team) & ((df['type']=='Aerial') | (df['type']=='Tackle') | (df['type']=='Interception') | (df['type']=='BlockedPass') | (df['type']=='Clearance'))][['type', 'name', 'teamName','playerId','outcomeType','x', 'y', 'oppositionTeamName']]
    opp = df.iloc[0]['oppositionTeamName']
    name = team
    if player:
        df = df.loc[df['name'] == player]
        name = player
    
    areal_s = len(df.loc[(df['type']=='Aerial') & (df['outcomeType']=='Successful')])
    areal_u = len(df.loc[(df['type']=='Aerial') & (df['outcomeType']!='Successful')])
    suptitle = str(areal_s) + '/' + str(areal_u+areal_s) + ' (' + str(round(percentage_try(areal_s, areal_u),2)) + '%) aereal duels '
    t_s = len(df.loc[(df['type']=='Tackle') & (df['outcomeType']=='Successful')])
    t_u = len(df.loc[(df['type']=='Tackle') & (df['outcomeType']!='Successful')])
    suptitle = suptitle + str(t_s) + '/' + str(t_u+t_s) + ' (' + str(round(percentage_try(t_s, t_u),2)) + '%) tackles '
    inter = len(df.loc[df['type']=='Interception'])
    suptitle = suptitle + str(inter) + ' interceptions '
    bp = len(df.loc[df['type']=='BlockedPass'])
    suptitle = suptitle + str(bp) + ' blocked passes '
    clear = len(df.loc[df['type']=='Clearance'])
    suptitle = suptitle + str(clear) + ' clearances '

    title = name + ' defensive actions vs ' + opp + ' (' + str(len(df)) + ')'
    plt.title(title, color='black', ha='center', y=1.05, fontsize=30)
    plt.title(title, color='black', ha='center', y=1.05, fontsize=30)
    plt.suptitle(suptitle, color='#3A3B3C', ha='center', y=0.9, fontsize=10)

    succ = df.loc[df['outcomeType']=='Successful']
    unsucc = df.loc[df['outcomeType']!='Successful']

    pitch.scatter(unsucc.x, unsucc.y, c='#FF0000', marker='o', ax=ax)
    pitch.scatter(succ.x, succ.y, c=c3, marker='o', ax=ax)
    st.pyplot(fig)
    
def plot_carries(df, team, player):
    pitch = Pitch(pitch_type='custom', pitch_width=68, pitch_length=105, pitch_color=c1, line_color=c2, line_zorder=1, linewidth=1, spot_scale = 0.00)
    fig, ax = pitch.draw(figsize=(16, 9))

    df = df.loc[(df['type'] == 'Carry') & (df['teamName'] == team), ['x', 'y', 'endX', 'endY','teamName','oppositionTeamName','name']]
    opp = df.iloc[0]['oppositionTeamName']
    name = team
    if player:
        df = df.loc[df['name'] == player]
        name = player
    
    title = name + ' carries vs ' + opp
    plt.title(title, color='black', ha='center', y=1.05, fontsize=30)

    supltitle = str(len(df)) + ' total carries'
    plt.suptitle(supltitle, color='#3A3B3C', ha='center', y=0.9, fontsize=10)

    pitch.arrows(df['x'],df['y'],df['endX'],df['endY'], width=2, headwidth=10, headlength=10, color=c3, ax=ax)
    st.pyplot(fig)

def plot_dribbles(df, team, player):
    pitch = Pitch(pitch_type='custom', pitch_width=68, pitch_length=105, pitch_color=c1, line_color=c2, line_zorder=1, linewidth=1, spot_scale = 0.00)
    fig, ax = pitch.draw(figsize=(16, 9))

    df = df.loc[(df['type'] == 'TakeOn') & ((df['teamName'] == team))][['x', 'y', 'outcomeType', 'name','oppositionTeamName', 'teamName']]
    opp = df.iloc[0]['oppositionTeamName']
    name = team
    if player:
        df = df.loc[df['name'] == player]
        name = player
    
    succ = df.loc[df.outcomeType=='Successful']
    unsucc = df.loc[df.outcomeType!='Successful']

    title = name + ' dribbles vs ' + opp
    plt.title(title, color='black', ha='center', y=1.05, fontsize=30)
    supltitle = str(len(succ)) + '/'+ str(len(df)) + ' (' + str(round(percentage_try(len(succ),len(df)), 2)) + '%) dribbles'
    plt.suptitle(supltitle, color='#3A3B3C', ha='center', y=0.9, fontsize=10)

    pitch.scatter(unsucc.x, unsucc.y, c='#FF0000', marker='o', ax=ax)
    pitch.scatter(succ.x, succ.y, c='#00FF00', marker='o', ax=ax)
    st.pyplot(fig)

def avgPositions(df):
    df_starters = df.loc[df['isFirstEleven'] == True]
    return df_starters.groupby('playerId').agg({
        'x': 'mean',
        'y': 'mean',
        'name': 'first',
        'shirtNo': 'first',
        'teamName': 'first'
        }).reset_index()

def plot_XI(df, team):
    df = avgPositions(df.loc[df.teamName == team])

    pitch = VerticalPitch(pitch_type='custom', pitch_width=68, pitch_length=105, pitch_color=c1, line_color=c2, line_zorder=1, linewidth=1, spot_scale = 0.00)
    fig, ax = pitch.draw(figsize=(16, 9))

    ax.scatter(df['y'], df['x'], c='white', s=300, edgecolors='white', zorder=2)
    for i, row in df.iterrows():
        ax.text(row['y'], row['x'], str(row['shirtNo']).split('.')[0], c='black', ha='center', va='center', fontsize=12, zorder=3)
        ax.text(row['y'], row['x'] + 2, row['name'], color='white', ha='center', va='center', fontsize=10, zorder=3)
    
    st.pyplot(fig)

def passReciever(df):
    passes = df.loc[(df['type']=='Pass') & (df['outcomeType']=='Successful')]
    df['passRecipientId'] = pd.Series(dtype='int')
    for idx, row in passes.iterrows():
        if idx < len(df):
            try:
                next_touch = df.iloc[idx+1]
                if next_touch['teamId'] == row['teamId']:
                    df.at[idx, 'passRecipientId'] = next_touch['playerId']
            except:
                pass
    return df

def get_player_shirtno_mapping(df, team):
    df_aux = df.loc[df['teamName'] == team]
    player_shirtno_dict = {}

    for player_id, group in df_aux.groupby('playerId'):
        shirt_no = group['shirtNo'].dropna().iloc[0] if not group['shirtNo'].dropna().empty else ""

        player_shirtno_dict[player_id] = shirt_no

    return player_shirtno_dict

def plot_pass_map(df, team):
    pitch = Pitch(pitch_type='custom', pitch_width=68, pitch_length=105, pitch_color=c1, line_color=c2, line_zorder=1, linewidth=1, spot_scale = 0.00)
    fig, ax = pitch.draw(figsize=(16, 9))

    df = df.loc[(df['type']=='Pass') & (df['outcomeType']=='Successful') & (df['teamName']==team)]
    passes = passReciever(df)
    
    passes_group = passes.groupby(['playerId','passRecipientId']).size()

    numbers_dict = get_player_shirtno_mapping(df, team)

    pitch.draw(ax=ax)
    opp = df.iloc[0]['oppositionTeamName']
    title = team + ' pass map vs ' + opp
    plt.title(title, color='black', ha='center', fontsize=30, y=1.05)

    player_means = passes.groupby('playerId').agg({'x': 'mean', 'y': 'mean'})
    player_counts = passes['playerId'].value_counts()
    passes_group = passes.groupby(['playerId','passRecipientId']).size().reset_index(name='pass_count')

    for player, row in player_means.iterrows():
        size = 200 + 15 * player_counts[player]
        ax.scatter(row['x'], row['y'], color='white', zorder=10, s=size)
        ax.annotate(str(int(numbers_dict[player])), (row['x'], row['y']), color='black', zorder=15, fontsize=12, ha='center', va='center')

    for _, row in passes_group.iterrows():
        player_from = row['playerId']
        player_to = row['passRecipientId']

        start_x, start_y = player_means.loc[player_from]
        try:
            end_x, end_y = player_means.loc[player_to]
        except:
            pass

        line_width = row['pass_count'] * 0.5
        alpha = min(1, 0.1 + (row['pass_count'] / passes_group['pass_count'].max()) * 0.9)
        pitch.lines(start_x, start_y, end_x, end_y, lw=line_width, alpha=alpha, color='white', zorder=5, ax=ax)

    st.pyplot(fig)

def plot_XT(df):
    df = df.loc[df['xT'].notna()][['teamName','xT','cumulative_mins','expandedMinute']]
    teams = df.teamName.unique()
    team1 = teams[0]
    team2 = teams[1]
    
    fig = plt.figure(figsize=(16, 9))

    team1_data = df[df['teamName'] == team1].sort_values('cumulative_mins')
    team2_data = df[df['teamName'] == team2].sort_values('cumulative_mins')

    xTsum1 = str(round(team1_data['xT'].sum(),2))
    xTsum2 = str(round(team2_data['xT'].sum(),2))

    plt.plot(team1_data['cumulative_mins'], team1_data['xT'], label=team1, color='red')
    plt.plot(team2_data['cumulative_mins'], -team2_data['xT'], label=team2, color='blue')

    plt.xlabel('Cumulative Minutes')
    plt.title(f'xT comparision: {team1} ({xTsum1}) vs {team2} ({xTsum2})')
    plt.legend()
    plt.grid(True)

    st.pyplot(fig)

def groupTouches(df):

  df.loc[:, 'endX'] = pd.to_numeric(df['endX'], errors='coerce')
  df.loc[:, 'endY'] = pd.to_numeric(df['endY'], errors='coerce')

  df = df.dropna(subset=['endX'])
  df = df.dropna(subset=['endY'])

  x_bins = pd.cut(df['endX'], bins=6, labels=False)
  y_bins = pd.cut(df['endY'], bins=4, labels=False)

  df['x_group'] = x_bins
  df['y_group'] = y_bins

  grouped = df.groupby(['teamName', 'x_group', 'y_group']).size().reset_index(name='count')
  total_by_team = df.groupby('teamName').size().reset_index(name='total')
  merged_df = pd.merge(grouped, total_by_team, on='teamName')

  return merged_df

def fieldTilt(df, team):
  lastThird = 2*105/3
  df = df.loc[df['x'] >= lastThird]
  df_team = df.loc[df['teamName'] == team]

  return len(df_team)/len(df)

def plot_touch_zones(df, team):
    df_touches = df.loc[(df['isTouch'] == True)]
    df = groupTouches(df_touches)

    field_tilt = fieldTilt(df_touches, team) * 100
    team_data = df[df['teamName'] == team]

    total_touches = len(df_touches)

    pitch = Pitch(pitch_type='custom', pitch_width=68, pitch_length=105, pitch_color=c2, line_color=c1, line_zorder=1, linewidth=1, spot_scale = 0.00)
    fig, ax = pitch.draw(figsize=(16, 9))

    num_x_bins = 6
    num_y_bins = 4

    for _, row in team_data.iterrows():
        x_center = (row['x_group'] + 0.5) * (105 / num_x_bins)
        y_center = (row['y_group'] + 0.5) * (68 / num_y_bins)

        percentage = (row['count'] / total_touches) * 100

        color_intensity = min(1, row['count'] / total_touches)
        color = (1, (1 - color_intensity*10), (1 - color_intensity*10))

        rect = patches.Rectangle((row['x_group'] * (105 / num_x_bins), row['y_group'] * (68 / num_y_bins)),
                                 (105 / num_x_bins), (68 / num_y_bins),
                                 linewidth=1, edgecolor='black', facecolor=color, alpha=0.6)
        ax.add_patch(rect)

        ax.text(x_center, y_center, f'{percentage:.1f}%', ha='center', va='center', fontsize=12, color='black')

    title = team + ' touches by zone (' + str(round(field_tilt, 2)) + '% field tilt)'
    plt.title(title, fontsize=30, y=1.05)
    plt.show()

    st.pyplot(fig)












