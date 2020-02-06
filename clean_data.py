import numpy as np
import pandas as pd
import os

data_folder= "data/"
output_folder= "output/"
if not os.path.exists(output_folder): os.mkdir(output_folder)

#read file
df= pd.read_csv(data_folder+ "2019_nCoV_data.csv")

#clean datatime column
df["Last Update"]= pd.to_datetime(df["Last Update"])
df['Day']= df['Last Update'].apply(lambda x: x.day)
df.to_csv(output_folder+ "cleaned_data.csv", index= False)


Confirmed = df.groupby('Day').sum()['Confirmed']
Deaths = df.groupby('Day').sum()['Deaths']
Recovered = df.groupby('Day').sum()['Recovered']
df2= pd.DataFrame(data= [Confirmed, Deaths, Recovered])
df2= df2.T
df2.to_csv(output_folder+ "grouped_by_day.csv")