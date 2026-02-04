#import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
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

#calculate standard deviations of relative P/R/A
leo_points_std = finished_df['L_PminusAvg'].std(ddof=0)
leo_rebounds_std = finished_df['L_RminusAvg'].std(ddof=0)
leo_assists_std = finished_df['L_AminusAvg'].std(ddof=0)
tom_points_std = finished_df['T_PminusAvg'].std(ddof=0)
tom_rebounds_std = finished_df['T_RminusAvg'].std(ddof=0)
tom_assists_std = finished_df['T_AminusAvg'].std(ddof=0)

#apply z-score normalization on relative P/R/A
finished_df = finished_df.assign(L_P_zs = ((finished_df.get('L_PminusAvg')) / leo_points_std).round(3))
finished_df = finished_df.assign(L_R_zs = ((finished_df.get('L_RminusAvg')) / leo_rebounds_std).round(3))
finished_df = finished_df.assign(L_A_zs = ((finished_df.get('L_AminusAvg')) / leo_assists_std).round(3))
finished_df = finished_df.assign(T_P_zs = ((finished_df.get('T_PminusAvg')) / tom_points_std).round(3))
finished_df = finished_df.assign(T_R_zs = ((finished_df.get('T_RminusAvg')) / tom_rebounds_std).round(3))
finished_df = finished_df.assign(T_A_zs = ((finished_df.get('T_AminusAvg')) / tom_assists_std).round(3))
finished_df.drop(columns=['L_PminusAvg', 'L_RminusAvg', 'L_AminusAvg', 'T_PminusAvg', 'T_RminusAvg', 'T_AminusAvg'], inplace=True)
finished_df

#split data into training and testing sets
X = finished_df.drop(columns=['Half', 'Outcome'])
y = finished_df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train logistic regression model
model = LogisticRegression(l1_ratio=0, C=1.0)
model.fit(X_train, y_train)

#make predictions
y_pred = model.predict(X_test)
model.score(X_test, y_test)

#evaluate model
log_loss(y, model.predict_proba(X)[:,1])

#get beta values for features
betas = model.coef_[0]
betas