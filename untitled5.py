4# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 20:34:06 2021

@author: Cu Chi
"""

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import csv







def solar_data(nrows):
     data_4 = pd.read_csv('C:/Users/Cu Chi/Dropbox/My PC (CuChi)/Desktop/Web-based GUI/Data_set/Solar irradiance data.csv', nrows=nrows)
     return data_4
weekly_data = solar_data(8760)
df_4 = pd.DataFrame(weekly_data[:8760],columns = ['Solar'],
                    index=pd.RangeIndex(8760, name='x'))    
df_4 = df_4.reset_index().melt('x', var_name='Solar irradiance', value_name='y')
line_chart = alt.Chart(df_4).mark_line().encode(
    alt.X('x', title='Time slot [hour]'),
    alt.Y('y', title='Solar irradiance [W/m2]'),
    color=alt.Color('Solar irradiance:N', legend=alt.Legend(orient='bottom'))).properties(title='Solar irradiance data during one year', width=300, height=300)
                                               
st.altair_chart(line_chart)

def capacity_data(nrows):
    data_2 = pd.read_csv('C:/Users/Cu Chi/Dropbox/My PC (CuChi)/Desktop/Web-based GUI/Data_set/Power output.csv', nrows=nrows)
    return data_2
power_data = capacity_data(365)
df_2 = pd.DataFrame(power_data[:365], columns =['Output1','Output2'],
                    index=pd.RangeIndex(365, name='x'))
df_2 = df_2.reset_index().melt('x', var_name='Output', value_name='y')
line_chart_2 = alt.Chart(df_2).mark_line().encode(
    alt.X('x', title='Time slot [hour]'),
    alt.Y('y', title='Output power [Kw/h]'),
    color=alt.Color('Output:N', legend=alt.Legend(orient='bottom'))).properties(title='Output power of solar panel', width=300, height=300)
st.altair_chart(line_chart_2)
    

















def CHP_data(nrows):
     data = pd.read_csv('C:/Users/Cu Chi/Dropbox/My PC (CuChi)/Desktop/Web-based GUI/Data_set/Discharging CHP.csv', nrows=nrows)
     return data
SCL_data = CHP_data(96)
df_4 = pd.DataFrame(SCL_data[:96],columns = ['CHP1', 'CHP2'],
                    index=pd.RangeIndex(96, name='x'))    
df_4 = df_4.reset_index().melt('x', var_name='CHP', value_name='y')
line_chart_4 = alt.Chart(df_4).mark_line().encode(
    alt.X('x', title='Time slot [min]'),
    alt.Y('y', title='Discharging power [MW]'),
    color='CHP:N').properties(title='CHP scheduling')

st.altair_chart(line_chart_4)