# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:58:19 2021

@author: Cu Chi
"""

#Import the libraries
import pandas as pd
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import requests
from io import BytesIO
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
url = 'https://raw.githubusercontent.com/TeckVo/GUI-design/main/Figure_set/Picture1.png'
response = requests.get(url)
image = Image.open(BytesIO(response.content))
#imge.show()



#image = Image.open('https://raw.githubusercontent.com/TeckVo/GUI-design/main/Figure_set/Picture1.png')
st.image(image, caption='Major components in a simulated microgrid',use_column_width=True)

df = pd.read_csv('https://raw.githubusercontent.com/TeckVo/GUI-design/main/Data_set/Data.csv')


col1, col2 = st.columns([2, 2])
col1.header('Basic data')
col2.header('Input data')



col3, col4 = st.columns([2, 2])
col3.header('Scheduling')
col4.header('Rewards')

#st.sidebar.header('Basic data')
app_model = col1.selectbox('Choose data',
                               ['Load demand', 'Capacity', 'Solar irradiance'])
if app_model == 'Load demand':
    col1.caption(f"{app_model}")
    @st.cache
    def load_data(nrows):
        data = pd.read_csv('https://raw.githubusercontent.com/TeckVo/GUI-design/main/Data_set/Base%20load.csv', nrows=nrows)
        return data
    load_demand = load_data(365)
    col1.line_chart(load_demand)
    
    #st.write(weekly_data)
elif app_model == 'Solar irradiance':
    col1.caption(f"{app_model}")
    @st.cache
    def solar_irradiance(nrows):
        data = pd.read_csv('https://raw.githubusercontent.com/TeckVo/GUI-design/main/Data_set/Solar%20irradiance.csv', nrows=nrows)
        return data
    solar_data = solar_irradiance(8760)
    col1.line_chart(solar_data)
elif app_model == 'Capacity':
    col1.caption(f"{app_model}")
    @st.cache
    def output_capacity(nrows):
        data = pd.read_csv('https://raw.githubusercontent.com/TeckVo/GUI-design/main/Data_set/Capacity.csv', nrows=nrows)
        return data
    capacity_data = output_capacity(8760)
    #st.line_chart(capacity_data)
    col1.area_chart(capacity_data)


def get_user_input():
    #st.sidebar.header('Customized input data')
    minTime = time(00,00)
    maxTime = time(23,00)
    defaultMin = time(10,00)
    defaultMax = time(20,00)
    Interruption_time = col2.slider('Interruption_time [H]', min_value=minTime, max_value=maxTime,value=(defaultMin, defaultMax), format="LT")
    col2.caption('*"Start and end time of an extreme event that as the extreme event that makes the microgrid unable to buy power from the main grid."*')
    Confidence_level = col2.slider('Confidence_level [%]', 0.00, 1.00, 0.1) 
    col2.caption('*"Confidence level in the range [0; 1] to denote the decision maker attitude in dealing with uncertainties."*')
    Desired_temp_HVAC = col2.number_input('Desired_temp_HVAC [°C]',18.00, 36.00, 26.00, 2.00)
    col2.caption('*"Desired temperature (°C) of HVAC system in the range [18°C; 36°C] during during the microgrid islanding period."*')
    Desired_temp_EWH = col2.number_input('Desired_temp_EWH [°C]', 30.00, 70.00, 40.00, 5.00) 
    col2.caption('*"Desired temperature (°C) of EWH system in the range [30°C; 70°C] during during the microgrid islanding period."*')
    #Interruption_time = col2.time_input('Interruption_time')
    #Desired_temp_HVAC = col2.selectbox('Desired_temp_HVAC [°C]',['select', 18, 20, 22, 24, 26, 28, 30, 32])
    #Desired_temp_EWH = col2.selectbox('Desired_temp_EWH [°C]', ['select', 30, 35, 40, 45, 50, 55, 60, 65, 70])
    
    user_data = {'Interruption_time': Interruption_time,
                 'Desired_temp_HVAC': Desired_temp_HVAC,
                 'Desired_temp_EWH': Desired_temp_EWH,
                 'Confidence_level': Confidence_level
                 }
    #features = pd.DataFrame(user_data, index = [0])
    #return features 
user_input = get_user_input()


        


@st.cache
def load_data(nrows):
     data = pd.read_csv('https://raw.githubusercontent.com/TeckVo/GUI-design/main/Data_set/Discharging%20ESS.csv', nrows=nrows)
     return data
weekly_data = load_data(96)
df_1 = pd.DataFrame(weekly_data[:96],columns = ['ESS1','ESS2'])
@st.cache
def load_center_data(nrows):
    data = pd.read_csv('https://raw.githubusercontent.com/TeckVo/GUI-design/main/Data_set/Discharging%20CHP.csv',nrows=nrows)
    return data
center_info_data = load_center_data(96)  
df_2 = pd.DataFrame(center_info_data[:96], columns = ['CHP1','CHP2'])   

app_model = col2.selectbox('Choose system',
                               ['ESS', 'CHP'])
              
        
if  col2.button('Click me'):
    #X = df.iloc[:, 0:8].values 
    #Y = df.iloc[:, -1].values 
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.25)
    #RandomForestClassifier = RandomForestClassifier()
    #RandomForestClassifier.fit(X_train, Y_train)
    col4.metric('Comfort level', '99.98 %', '4.98 % basiline 95 %')
    col4.metric('Operating cost', '110.54 $', '-44.04 % basiline 197.54 $')
    if app_model == 'ESS':
         col3.caption('*"Desired temperature (°C) of EWH system in the range [30°C; 70°C] during during the microgrid islanding period."*')
         col3.caption(f"{app_model} system")
         col3.write("""
         Discharging power [MW] from ESSs at each time slot.
         """)
         col3.line_chart(df_1)
    elif app_model == 'CHP':
        col3.caption(f"{app_model} system: ")
        col3.write("""
         Discharging power [MW] from CHPs at each time slot.
         """)
        col3.line_chart(df_2)
else:
    st.write('Loading data....')

 

#if  col2.button('Click me'):
    
   
    
        #col3.caption(f"{app_model}")
        #
    #elif app_model == 'CHP':
       #col3.caption(f"{app_model}")
        #col3.line_chart(df_2)


        
        #col3.write(weekly_data)

        #col3.write(center_info_data)        
    
        
   
   


        
    
    
