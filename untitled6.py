# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 13:18:25 2021

@author: Cu Chi
"""


#Import the libraries
import pandas as pd
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st 
#import matplotlib.pyplot as plt
import numpy as np
from datetime import time

#Create a title and sub-title 

st.title('Microgrid proactive scheduling')
st.write("""
         Microgrid proactive scheduling subjecting to extreme events using safe reinforcement learning method
         """)
         
#Open and display an image 
image = Image.open('C:/Users/Cu Chi/Dropbox/My PC (CuChi)/Desktop/Web-based GUI/Picture1.png')
st.image(image, caption='Major components in a simulated microgrid',use_column_width=True)

df = pd.read_csv('C:/Users/Cu Chi/Dropbox/My PC (CuChi)/Desktop/Web-based GUI/Data.csv')


col1, col2 = st.columns([4, 1])
col1.header('Basic data')
col2.header('Results')
col2.caption('Rewards')
col3, col4 = st.columns(2)
col3.header('ESSs scheduling')
col3.caption('Discharging power quantity [MW] from ESSs at each slot')
col4.header('CHPs scheduling')
col4.caption('Discharging power quantity [MW] from CHPs at each slot')


st.sidebar.header('Basic data')
app_model = st.sidebar.selectbox('Choose data',
                               ['Load demand', 'Capacity', 'Solar irradiance'])
if app_model == 'Load demand':
    col1.caption(f"{app_model}")
    @st.cache
    def load_data(nrows):
        data = pd.read_csv('C:/Users/Cu Chi/Dropbox/My PC (CuChi)/Desktop/Web-based GUI/Base load.csv', nrows=nrows)
        return data
    load_demand = load_data(365)
    col1.line_chart(load_demand)
    
    #st.write(weekly_data)
elif app_model == 'Solar irradiance':
    col1.write(f"{app_model}")
    @st.cache
    def solar_irradiance(nrows):
        data = pd.read_csv('C:/Users/Cu Chi/Dropbox/My PC (CuChi)/Desktop/Web-based GUI/Solar irradiance.csv', nrows=nrows)
        return data
    solar_data = solar_irradiance(8760)
    col1.line_chart(solar_data)
elif app_model == 'Capacity':
    col1.write(f"{app_model}")
    @st.cache
    def output_capacity(nrows):
        data = pd.read_csv('C:/Users/Cu Chi/Dropbox/My PC (CuChi)/Desktop/Web-based GUI/Capacity.csv', nrows=nrows)
        return data
    capacity_data = output_capacity(8760)
    #st.line_chart(capacity_data)
    col1.area_chart(capacity_data)


def get_user_input():
    st.sidebar.header('Customized input data')
    Interruption_time = st.sidebar.time_input('Interruption_time')
    Desired_temp_HVAC = st.sidebar.selectbox('Desired_temp_HVAC [°C]',['select', 18, 20, 22, 24, 26, 28, 30, 32])
    Desired_temp_EWH = st.sidebar.selectbox('Desired_temp_EWH [°C]', ['select', 30, 35, 40, 45, 50, 55, 60, 65, 70])
    Confidence_level = st.sidebar.slider('Confidence_level [%]', 0.00, 1.00, 0.1)
    user_data = {'Interruption_time': Interruption_time,
                 'Desired_temp_HVAC': Desired_temp_HVAC,
                 'Desired_temp_EWH': Desired_temp_EWH,
                 'Confidence_level': Confidence_level
                 }
    features = pd.DataFrame(user_data, index = [0])
    return features 
user_input = get_user_input()

@st.cache
def load_data(nrows):
    data = pd.read_csv('C:/Users/Cu Chi/Dropbox/My PC (CuChi)/Desktop/Web-based GUI/Discharging ESS.csv', nrows=nrows)
    return data
@st.cache
def load_center_data(nrows):
    data = pd.read_csv('C:/Users/Cu Chi/Dropbox/My PC (CuChi)/Desktop/Web-based GUI/Discharging CHP.csv',nrows=nrows)
    return data
weekly_data = load_data(96)
center_info_data = load_center_data(96)


if st.sidebar.button('Click me'):
    #X = df.iloc[:, 0:8].values 
    #Y = df.iloc[:, -1].values 
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.25)
    #RandomForestClassifier = RandomForestClassifier()
    #RandomForestClassifier.fit(X_train, Y_train)
    col2.metric('Comfort level', '99.98 %')
    col2.metric('Operating cost', '148.35 $')
    df_1 = pd.DataFrame(weekly_data[:96],columns = ['ESS1','ESS2'])
    col3.line_chart(df_1)
    col3.write(weekly_data)
    df_2 = pd.DataFrame(center_info_data[:96], columns = ['CHP1','CHP2'])
    col4.line_chart(df_2)
    col4.write(center_info_data)      
else:
    st.write('Loading data....')
