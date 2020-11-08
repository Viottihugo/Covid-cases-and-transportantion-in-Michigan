# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 12:06:41 2020

@author: viotti hugo
"""

import pandas as pd

#trips info taken from https://data.bts.gov/Research-and-Statistics/Trips-by-Distance/w96p-f2qv
trips_df=pd.read_csv('data\Trips_by_Distance.csv')

trips_MI=trips_df[trips_df['State Postal Code']=='MI'].iloc[:,:-10]

trips_MI=trips_MI[pd.to_datetime(trips_MI['Date'])>= '2020-03-01']

trips_MI.set_index('Date', inplace=True)

#saving the data to a csv file for later use
(trips_MI.groupby(trips_MI.index).sum()).to_csv('data\Trips_MI.csv')