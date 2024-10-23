import streamlit as st
import functions as f

st.title("Game actions plotter")

d = f.dict_games()
game = st.selectbox("Game: ", d.keys(), index=None)

if game is not None:
    df = f.read_df(d[game])
    team = st.selectbox("Team: ", df['teamName'].sort_values().unique(), index=None)
    player = st.selectbox("Player: ", df.loc[df.teamName == team]['name'].sort_values().unique(), index=None)
    plots = ['Starting XI', 'Pass map', 'Touches by zone', 'xT by minute', 'Passes', 'Shots', 'Dribbles', 'Carries', 'Def. Actions']
    period = st.selectbox("Period: ", ['FirstHalf','SecondHalf'], index=None)
    action = st.selectbox('Action: ', plots, index=None)
    if team:
        f.plot_player_actions(df, team, player, action, period)
