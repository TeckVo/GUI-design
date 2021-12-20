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
import altair as alt
import csv

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



def base_data(nrows):
     data = pd.read_csv('https://raw.githubusercontent.com/TeckVo/GUI-design/main/Data_set/Base%20load.csv', nrows=nrows)
     return data
load_demand = base_data(365)
df_1 = pd.DataFrame(load_demand[:365],columns = ['Load'],
                    index=pd.RangeIndex(365, name='x'))    
df_1 = df_1.reset_index().melt('x', var_name='Load', value_name='y')
line_chart = alt.Chart(df_1).mark_line().encode(
    alt.X('x', title='Time slot [day]'),
    alt.Y('y', title='Base load [p.u]'),
    color=alt.Color('Load:N', legend=alt.Legend(orient='bottom'))).properties(title='Load demand', width=300, height=300)


def capacity_data(nrows):
         data = pd.read_csv('https://raw.githubusercontent.com/TeckVo/GUI-design/main/Data_set/Power%20output.csv',nrows=nrows)
         return data
power_out = capacity_data(8760)
df_2 = pd.DataFrame(power_out[:8760], columns = ['Power'],
                    index=pd.RangeIndex(8760, name='x'))
df_2 = df_2.reset_index().melt('x', var_name='Output', value_name='y')
line_chart_3 = alt.Chart(df_2).mark_line().encode(
         alt.X('x', title='Time slot [hour]'),
         alt.Y('y', title='Output power [Kw/h]'),
         color=alt.Color('Output:N', legend=alt.Legend(orient='bottom'))).properties(title='Output power from solar panel system', width=300, height=300)

def solar_data(nrows):
         data = pd.read_csv('https://raw.githubusercontent.com/TeckVo/GUI-design/main/Data_set/Solar%20irradiance%20data.csv',nrows=nrows)
         return data
solar_out = solar_data(8760)
df_4 = pd.DataFrame(solar_out[:8760], columns = ['Solar'],
                    index=pd.RangeIndex(8760, name='x'))
df_4 = df_4.reset_index().melt('x', var_name='Solar irradiance', value_name='y')
line_chart_4 = alt.Chart(df_4).mark_line().encode(
         alt.X('x', title='Time slot [hour]'),
         alt.Y('y', title='Solar irradiance [W/m2]'),
         color=alt.Color('Solar irradiance:N', legend=alt.Legend(orient='bottom'))).properties(title='Solar irradiance data during one year', width=300, height=300)
        
    



      
         
#st.sidebar.header('Basic data')
app_model = col1.selectbox('Choose data',
                               ['Load demand', 'Capacity', 'Solar irradiance'])
if app_model == 'Load demand':
    #col1.caption(f"{app_model}")
    col1.altair_chart(line_chart)
    with col1.expander("See explanation"):
                  st.caption("""*Base load for one year during 365 days [p.u/h].*""")





    #col1.line_chart( base_load)
         
 
    #st.write(weekly_data)
elif app_model == 'Solar irradiance':
    #col1.caption(f"{app_model}")
    col1.altair_chart(line_chart_4)
    with col1.expander("See explanation"):
                  st.caption("""*Solar irradiance data [W/m2] for one year during 8,760 time slots [hour].*""")
    
    #def solar_irradiance(nrows):
        #data = pd.read_csv('https://raw.githubusercontent.com/TeckVo/GUI-design/main/Data_set/Solar%20irradiance.csv', nrows=nrows)
        #return data
    #solar_data = solar_irradiance(8760)
    #col1.line_chart(solar_data)
elif app_model == 'Capacity':
    #col1.caption(f"{app_model}")
    col1.altair_chart(line_chart_3)
    with col1.expander("See explanation"):
                  st.caption("""*Outout power [KW/h] of a roof-top solar panel with installed capacity 6MW for one year during 8,760 time slots [hour].*""")
    
    #def output_capacity(nrows):
        #data = pd.read_csv('https://raw.githubusercontent.com/TeckVo/GUI-design/main/Data_set/Capacity.csv', nrows=nrows)
        #return data
    #capacity_data = output_capacity(8760)
    #st.line_chart(capacity_data)
    #col1.area_chart(capacity_data)
         
         
uploaded_files = col1.file_uploader("Upload a new CSV file data", accept_multiple_files=True)
for uploaded_file in uploaded_files:
     bytes_data = uploaded_file.read()
     st.write("filename:", uploaded_file.name)
     st.write(bytes_data)

def get_user_input():
    #st.sidebar.header('Customized input data')
    minTime = time(00,00)
    maxTime = time(23,00)
    defaultMin = time(10,00)
    defaultMax = time(11,00)
    Interruption_time = col2.slider('1. Interruption_time [H]', min_value=minTime, max_value=maxTime,value=(defaultMin, defaultMax), format="LT")
    with col2.expander("See explanation"):
         st.caption("""*Start and end time of an extreme event that as the extreme event that makes the microgrid unable to buy power from the main grid.*""")
         
    #col2.caption('*"Start and end time of an extreme event that as the extreme event that makes the microgrid unable to buy power from the main grid."*')
    Confidence_level = col2.slider('2. Confidence_level [%]', 0.00, 1.00, 0.95) 
    with col2.expander("See explanation"):
         st.caption("""*Confidence level in [0; 1] to denote the decision maker attitude in dealing with uncertainties.*""")
    #col2.caption('*"Confidence level in [0; 1] to denote the decision maker attitude in dealing with uncertainties."*')
    Desired_temp_HVAC = col2.number_input('3. Desired_temp_HVAC [°C]',18.00, 36.00, 26.00, 1.00)
    with col2.expander("See explanation"):
         st.caption("""*Desired temperature (°C) of HVAC system in [18°C; 36°C] during during the microgrid islanding period.*""")
    #col2.caption('*"Desired temperature (°C) of HVAC system in [18°C; 36°C] during during the microgrid islanding period."*')
    Desired_temp_EWH = col2.number_input('4. Desired_temp_EWH [°C]', 30.00, 70.00, 50.00, 1.00)
    with col2.expander("See explanation"):
         st.caption("""*Desired temperature (°C) of EWH system in [30°C; 70°C] during during the microgrid islanding period.*""")
    #col2.caption('*"Desired temperature (°C) of EWH system in [30°C; 70°C] during during the microgrid islanding period."*')
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

def load_data(nrows):
     data = pd.read_csv('https://raw.githubusercontent.com/TeckVo/GUI-design/main/Data_set/Discharging%20ESS.csv', nrows=nrows)
     return data
weekly_data = load_data(96)
df_1 = pd.DataFrame(weekly_data[:96],columns = ['ESS1','ESS2'],
                    index=pd.RangeIndex(100, name='x'))    
df_1 = df_1.reset_index().melt('x', var_name='ESS', value_name='y')
line_chart_1 = alt.Chart(df_1).mark_line().encode(
    alt.X('x', title='Time slot [min]'),
    alt.Y('y', title='Discharging power [MW]'),
    color=alt.Color('ESS:N', legend=alt.Legend(orient='bottom'))).properties(title='ESS scheduling', width=300, height=300)





def load_center_data(nrows):
     data = pd.read_csv('https://raw.githubusercontent.com/TeckVo/GUI-design/main/Data_set/Discharging%20CHP.csv', nrows=nrows)
     return data
center_info_data = load_center_data(96)
df_2 = pd.DataFrame(center_info_data[:96],columns = ['CHP1','CHP2'],
                    index=pd.RangeIndex(100, name='x'))    
df_2 = df_2.reset_index().melt('x', var_name='CHP', value_name='y')
line_chart_2 = alt.Chart(df_2).mark_line().encode(
    alt.X('x', title='Time slot [min]'),
    alt.Y('y', title='Discharging power [MW]'),
    color=alt.Color('CHP:N', legend=alt.Legend(orient='bottom'))).properties(title='CHP scheduling', width=300, height=300)
                   

                    
                    
                    


                   


#col2.caption('*"Selecting system needs to schedule for reacting to the extreme events."*')
              
        
if  col2.button('Click me'):
    #X = df.iloc[:, 0:8].values 
    #Y = df.iloc[:, -1].values 
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.25)
    #RandomForestClassifier = RandomForestClassifier()
    #RandomForestClassifier.fit(X_train, Y_train)
    col4.metric('Comfort level', '99.98 %', '5.09 % compared with baseline 94.89 %')
    with col4.expander("See explanation"):
                  st.caption("""*1. Comfort level indicates the ability to continously supply power to critical loads, 
                  such as HVAC and EWH systems during the islanding microgrid period caused by extreme events.*""")
                  st.caption ("""*2. Baseline comfort leve is defined based on scenario-based stochastic programmin method.*""")
                 
    col4.metric('Operating cost', '110.54 $', '-44.04 % compared withbaseline 197.54 $')
    with col4.expander("See explanation"):
                  st.caption("""*1. Operating cost of microgrid consists the following cost components: (1) power purchase cost from the main gird; 
                  (2) degradation cost of energy storage systems (ESSs); (3) operating cost of gas-combined heat and power systems (CHPs); 
                  (4) penalty cost for power mismatches caused by extreme events.*""")
                  st.caption ("""*2. Baseline comfort leve is defined based on scenario-based stochastic programmin method.*""")
    d = {'electricity cost': [49.12 $], 'degradation cost': [27.21 $], 'operating cost': [25.08 $], 'penalty cost': [09.13 $] }
    df = pd.DataFrame(data=d)
    col4.table(df)
    col3.caption('Choose system needs to schedule')
    
    with col3.expander("See explanation"):
                  st.caption("""*Selecting system needs to schedule for reacting to the extreme events 
         in which:*""")
                  st.caption("""*ESS denotes the energy storage system and CHP is the gas-combined heat and power system.*""")  
else:
    col2.write('Loading data....')

app_model = col3.selectbox('Choose system',
                           ['Please select system', 'ESS', 'CHP'])
if app_model == 'Please select system':
         col3.caption(f"{app_model}")
elif app_model == 'ESS':
         col3.caption(f"{app_model} system")
         col3.altair_chart(line_chart_1)
         with col3.expander("See explanation"):
                  st.caption("""*Discharging power amount [MW] of each ESS to enhance the microgrid resilience during the islanding period.*""")  
         
elif app_model == 'CHP':
         col3.caption(f"{app_model}")
         col3.altair_chart(line_chart_2)
         with col3.expander("See explanation"):
                  st.caption("""*Discharging power amount [MW] of each CHP to enhance the microgrid resilience during the islanding period.*""")   
                    
                      

                      
                      

         
         
         
         
         
     
    
        
   
   


    
