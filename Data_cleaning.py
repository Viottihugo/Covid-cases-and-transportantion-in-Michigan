# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 12:06:41 2020

@author: viotti hugo
"""

import pandas as pd


# covid data taken from https://covidtracking.com/data/state/michigan/cases
covid_df=pd.read_excel('data\MI_COVID.xlsx')


covid_df['MONTH']=pd.to_datetime(covid_df.Date).dt.month

covid_df.head()
covid_by_month=covid_df[['New cases', 'MONTH']].groupby('MONTH').sum()
covid_by_month=covid_by_month[:5]
print(covid_df[['New cases', 'MONTH']].groupby('MONTH').sum())
print(covid_by_month)


#trips info taken from https://data.bts.gov/Research-and-Statistics/Trips-by-Distance/w96p-f2qv
trips_df=pd.read_csv('data\Trips_by_Distance.csv')

trips_national=trips_df[trips_df['Level']=='National'].iloc[:,:-10]

trips_national=trips_national[pd.to_datetime(trips_national['Date'])>= '2020-03-01']


	
