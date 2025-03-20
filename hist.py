# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 16:31:42 2025

@author: Elsa_Ehrhart
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

demand_data = pd.read_csv('./data/demand2050_ADEME.csv', header=None)
demand_data.columns = ["time","demand"]

solar_data = pd.read_csv('./data/solar.csv')['facteur_charge'].values
wind_data = pd.read_csv('./data/wind_onshore.csv')['facteur_charge'].values

wind_capacity = 170.1  # given ? or to be calculated ? or to be set with different values to test ?
solar_capacity = 308.4 # max of energy we can gather using wind / sun in one step

prod_res=wind_capacity*wind_data+solar_capacity*solar_data-demand_data['demand'].values


plt.figure("figure1")
plt.plot(np.ones(len(demand_data['time'].values))*(wind_capacity+solar_capacity))
plt.plot(demand_data['time'].values,prod_res)
plt.show()