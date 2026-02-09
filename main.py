#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

#Read dataset
raw_df = pd.read_csv(r"C:\Users\aravk\OneDrive\Documents\GitHub\Logistic-Regression-Project--Beath-vs.-Beattie\data\Logistic Regression Project Raw Data.csv")
raw_df

#Save games with non-bibary (W/L) outcomes to a separate dataframe
tied_half = raw_df[(raw_df.get('Outcome') != 'W') & (raw_df.get('Outcome') != 'L')].set_index('GameID')

#Clean data
raw_df = raw_df.set_index('GameID')
cleaned_df = raw_df.drop(tied_half.index)

#Convert outcome to binary
outcomes = ['L', 'W']
enc = OrdinalEncoder(categories=[outcomes])
cleaned_df['Outcome'] = enc.fit_transform(cleaned_df[['Outcome']])

#Find P/R/A averages for each player
leo_points_avg = cleaned_df['L_Points'].mean()
leo_rebounds_avg = cleaned_df['L_Rebounds'].mean()
leo_assists_avg = cleaned_df['L_Assists'].mean()
tom_points_avg = cleaned_df['T_Points'].mean()
tom_rebounds_avg = cleaned_df['T_Rebounds'].mean()
tom_assists_avg = cleaned_df['T_Assists'].mean()

#Replace P/R/A with relative P/R/A by finding change in performance relative to averages
finished_df = cleaned_df.assign(L_PminusAvg = (cleaned_df.get('L_Points') - leo_points_avg).round(3))
finished_df = finished_df.assign(L_RminusAvg = (cleaned_df.get('L_Rebounds') - leo_rebounds_avg).round(3))
finished_df = finished_df.assign(L_AminusAvg = (cleaned_df.get('L_Assists') - leo_assists_avg).round(3))
finished_df = finished_df.assign(T_PminusAvg = (cleaned_df.get('T_Points') - tom_points_avg).round(3))
finished_df = finished_df.assign(T_RminusAvg = (cleaned_df.get('T_Rebounds') - tom_rebounds_avg).round(3))
finished_df = finished_df.assign(T_AminusAvg = (cleaned_df.get('T_Assists') - tom_assists_avg).round(3))
finished_df.drop(columns=['L_Points', 'L_Rebounds', 'L_Assists', 'T_Points', 'T_Rebounds', 'T_Assists'], inplace=True)

#Calculate standard deviations of relative P/R/A
leo_points_std = finished_df['L_PminusAvg'].std(ddof=0)
leo_rebounds_std = finished_df['L_RminusAvg'].std(ddof=0)
leo_assists_std = finished_df['L_AminusAvg'].std(ddof=0)
tom_points_std = finished_df['T_PminusAvg'].std(ddof=0)
tom_rebounds_std = finished_df['T_RminusAvg'].std(ddof=0)
tom_assists_std = finished_df['T_AminusAvg'].std(ddof=0)

#Apply z-score normalization on relative P/R/A
finished_df = finished_df.assign(L_P_zs = ((finished_df.get('L_PminusAvg')) / leo_points_std).round(3))
finished_df = finished_df.assign(L_R_zs = ((finished_df.get('L_RminusAvg')) / leo_rebounds_std).round(3))
finished_df = finished_df.assign(L_A_zs = ((finished_df.get('L_AminusAvg')) / leo_assists_std).round(3))
finished_df = finished_df.assign(T_P_zs = ((finished_df.get('T_PminusAvg')) / tom_points_std).round(3))
finished_df = finished_df.assign(T_R_zs = ((finished_df.get('T_RminusAvg')) / tom_rebounds_std).round(3))
finished_df = finished_df.assign(T_A_zs = ((finished_df.get('T_AminusAvg')) / tom_assists_std).round(3))
finished_df.drop(columns=['L_PminusAvg', 'L_RminusAvg', 'L_AminusAvg', 'T_PminusAvg', 'T_RminusAvg', 'T_AminusAvg'], inplace=True)
finished_df

#Split data into training and testing sets
X = finished_df.drop(columns=['Half', 'Outcome'])
y = finished_df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train logistic regression model
model = LogisticRegression(l1_ratio=0, C=1.0)
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

#Get beta values for features
betas = model.coef_[0]
betas
betas_df = pd.DataFrame({
    'feature': ['L_P_zs', 'L_R_zs', 'L_A_zs', 'T_P_zs', 'T_R_zs', 'T_A_zs'],
    'beta': betas
}).sort_values('beta', ascending=False).set_index('feature')
betas_plot = betas_df.plot(kind='barh', title='Beta Values for Each Feature').invert_yaxis()

#Evaluate model
score = model.score(X_test, y_test)
log_loss = log_loss(y, model.predict_proba(X)[:,1])
confusion_matrix = confusion_matrix(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

#Plot evaluation metrics

#model score
plt.figure(figsize=(4,4))
plt.bar(['Model Score'], [score], color='green')
plt.ylim(0, 1)
plt.title('Model Score (Accuracy)')
plt.show()

#log loss
plt.figure(figsize=(4,4))
plt.bar(['Log Loss'], [log_loss], color='green')
plt.axhline(y=0.5, color='red', linestyle='--', label='"Good" Threshold')
plt.legend(loc='upper right')
plt.ylim(0, log_loss + 1)
plt.title('Log Loss')
plt.show()

#confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#classification report
print('Classification Report:\n', classification_report)

#Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for UCSD Game Outcome Classification')
plt.legend(loc='lower right')
plt.show()