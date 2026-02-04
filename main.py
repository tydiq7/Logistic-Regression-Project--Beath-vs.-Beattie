#import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#read dataset
raw_df = pd.read_csv(r"C:\Users\aravk\OneDrive\Documents\GitHub\Logistic-Regression-Project--Beath-vs.-Beattie\data\Logistic Regression Project Raw Data.csv")
raw_df

#save games with non-bibary (W/L) outcomes to a separate dataframe
tied_half = raw_df[(raw_df.get('Outcome') != 'W') & (raw_df.get('Outcome') != 'L')].set_index('GameID')

#clean data
raw_df = raw_df.set_index('GameID')
cleaned_df = raw_df.drop(tied_half.index)

#convert outcome to binary
outcomes = ['L', 'W']
enc = OrdinalEncoder(categories=[outcomes])
cleaned_df['Outcome'] = enc.fit_transform(cleaned_df[['Outcome']])

#find P/R/A averages for each player
leo_points_avg = cleaned_df['L_Points'].mean()
leo_rebounds_avg = cleaned_df['L_Rebounds'].mean()
leo_assists_avg = cleaned_df['L_Assists'].mean()
tom_points_avg = cleaned_df['T_Points'].mean()
tom_rebounds_avg = cleaned_df['T_Rebounds'].mean()
tom_assists_avg = cleaned_df['T_Assists'].mean()

#replace P/R/A with relative P/R/A by finding change in performance relative to averages
finished_df = cleaned_df.assign(L_PminusAvg = (cleaned_df.get('L_Points') - leo_points_avg).round(3))
finished_df = finished_df.assign(L_RminusAvg = (cleaned_df.get('L_Rebounds') - leo_rebounds_avg).round(3))
finished_df = finished_df.assign(L_AminusAvg = (cleaned_df.get('L_Assists') - leo_assists_avg).round(3))
finished_df = finished_df.assign(T_PminusAvg = (cleaned_df.get('T_Points') - tom_points_avg).round(3))
finished_df = finished_df.assign(T_RminusAvg = (cleaned_df.get('T_Rebounds') - tom_rebounds_avg).round(3))
finished_df = finished_df.assign(T_AminusAvg = (cleaned_df.get('T_Assists') - tom_assists_avg).round(3))
finished_df.drop(columns=['L_Points', 'L_Rebounds', 'L_Assists', 'T_Points', 'T_Rebounds', 'T_Assists'], inplace=True)

#apply z-score normalization on features
