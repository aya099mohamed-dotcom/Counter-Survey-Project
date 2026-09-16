import streamlit as st 
import pandas as pd 
import joblib


model = joblib.load('model.pkl')
features = joblib.load('features.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Counter Survey Prediction')
columns = st.columns(len(features))

listed_in = columns[0].selectbox('Listed in',['simple; free', 'cute', 'cute; free', 'simple'])
lock_screen_per_description= columns[1].selectbox('Lock Screen Per Description',['yes', 'no'])
in_app_purchases= columns[2].selectbox('In App Purchases',['yes', 'none'])
kind = columns[3].selectbox('Kind',[                 'step counter',                  'walking game',
         'counter + virtual dog',         'fitness streaks + pet',
           'habit tracker + pet', 'counter + collectable animals',
             'counter + hamster'])

df = pd.DataFrame([[listed_in ,lock_screen_per_description,in_app_purchases,kind]],columns=features)

if st.button('predict'):
    for encode in features:
        df[encode]= encoder[encode].transform(df[encode])
    df[features]= scaler.transform(df[features])
    predict= model.predict(df[features])
    st.write('Prediction is : ',predict)   
     