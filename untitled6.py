# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 12:47:11 2021

@author: Cu Chi
"""

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import csv

with open("C:/Users/Cu Chi/Dropbox/My PC (CuChi)/Desktop/Web-based GUI/Data_set/Capacity.csv","r")as i:
    rawdata = list(csv.reader(i,delimiter=","))
exampledata=np.array(rawdata[1:],dtype=np.float64)   
xdata=exampledata[:,0]
ydata_1=exampledata[:,1]
ydata_2=exampledata[:,2]

ax=plt.subplots()
plt.plot(xdata, ydata_1)
         
plt.xlabel("Time slot [min]")
plt.ylabel("Power mismatch [kW]")
ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
ax.xaxis.set_minor_locator(tck.AutoMinorLocator(10))
plt.title("(RDRL)", y=1.0, pad=-14, loc='right')
plt.show()
