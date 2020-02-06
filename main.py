from model import load_data, seq2seq, base_model, normalize_data
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
folder_name= "processed_data/"
import numpy as np

#read and prepare data 
scaler= MinMaxScaler(feature_range=(0, 1))
df= pd.read_csv(folder_name+ "cleaned_data.csv")
data= normalize_data(df, scaler)
train, train_labels, validate, validate_labels= load_data(data, time_step= 3, after_day=4, train_percent= 0.5)

print("train_data:", train.shape, train_labels.shape)
print("validate_data:", validate.shape, validate_labels.shape)
