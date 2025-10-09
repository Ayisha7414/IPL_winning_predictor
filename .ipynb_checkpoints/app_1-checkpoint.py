import streamlit as st
import numpy as np
import joblib
import pandas as pd
model = joblib.load('ipl_model.pkl')
st.set_page_config(page_title = "IPL Match Winning Predictor",page_icon = "🏏",layout="wide")
st.header("🏏IPL MATCH WINNING PREDICTOR🏏")
batting_team = st.selectbox("Batting Team:",[
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad', 
    'Delhi Capitals', 
    'Chennai Super Kings',
    'Gujarat Titans', 
    'Lucknow Super Giants', 
    'Kolkata Knight Riders',
    'Punjab Kings', 
    'Mumbai Indians'
])
bowling_team = st.selectbox("Bowling Team:",[
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad', 
    'Delhi Capitals', 
    'Chennai Super Kings',
    'Gujarat Titans', 
    'Lucknow Super Giants', 
    'Kolkata Knight Riders',
    'Punjab Kings', 
    'Mumbai Indians'
])
city = st.selectbox("City :",["Chandigarh","Bangalore","Delhi","Mumbai","Chennai","Kolkata"])
runs_left = st.number_input("Runs Left:",min_value = 1,max_value = 300, step = 1)
balls_left = st.number_input("Ball Left:",min_value = 1,max_value = 120, step = 1)
wicket_left = st.number_input("Wicket Left:",min_value = 1,max_value = 10, step = 1)
current_run_rate = st.number_input("Current run rate:",min_value = 1.0,max_value = 200.0, step = 1.0,format = "%.2f")
target = st.number_input(" Taget :",min_value = 1,max_value = 300, step = 1)
required_run_rate = (runs_left/balls_left) * 6
st.write("Required run rate:")
st.write(required_run_rate)
import pandas as pd

# Exact column names from training
feature_names = [
    "BattingTeam",
    "BowlingTeam",
    "City",
    "runs_left",
    "balls_left",
    "wickets_left",
    "current_run_rate",
    "target",
    "required_run_rate"
]

# Wrap input into DataFrame
features = pd.DataFrame([[
    batting_team,
    bowling_team,
    city,
    runs_left,
    balls_left,
    wicket_left,   # variable name (singular), maps to "wickets_left"
    current_run_rate,
    target,
    required_run_rate
]], columns=feature_names)

if st.button("Predict"):
    # Get probability of winning
    win_prob = model.predict_proba(features)[0][1]  # [0][1] → probability of class "1" (win)

    st.write(f"🏏 **Win Probability for {batting_team}: {win_prob*100:.2f}%**")
    st.write(f"🏏 **Lose Probability for {bowling_team}: {(1-win_prob)*100:.2f}%**")


