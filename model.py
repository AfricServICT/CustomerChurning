
import pandas as pd
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.offline as po
import plotly.graph_objects as go
import pickle




sns.set(style = 'white')
path="datasets_13996_18858_WA_Fn-UseC_-Telco-Customer-Churn.csv"
dataChurn=pd.read_csv(path)

# Converting Total Charges to a numerical data type.
dC1=dataChurn.TotalCharges = pd.to_numeric(dataChurn.TotalCharges, errors='coerce')
#Notice that they are 11 missing values for Total Charges
#dC2=dataChurn.isnull().sum()
#print(dC2)
#ENCODING
#Removing the missing values 
dataChurn.dropna(inplace = True)
#Removing customer IDs from the data set
df2 = dataChurn.iloc[:,1:]
#Converting the predictor variable in a binary numeric variable
df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df2['Churn'].replace(to_replace='No',  value=0, inplace=True)

#Converting all the categorical variables into dummy variables
df_dummies = pd.get_dummies(df2)
#df_dummies.head()
#print(df_dummies)

#QUESTION 1
#Getting the Correlation of "Churn" with other variables:
plt.figure(figsize=(15,15))
plt.title('CORRELATION GRAPH', fontsize=20)
questionAns=df_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar',color='blue')
print(questionAns)

#QUESTION 2
#Using relevant mapping features show features which have the strongest correlation with churning.    

colors = ['#4D3425','#E4512B']
ax = (dataChurn['Churn'].value_counts()*100.0 /len(dataChurn)).plot(kind='bar',stacked = True, rot = 0, color = colors, figsize = (8,6))
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('Percentage of Customers',size = 15)
ax.set_xlabel('Churn',size = 14)
ax.set_title('Churn Rate', size = 14)
# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_width())

# set individual bar lables using above list
total = sum(totals)

for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_x()+.15, i.get_height()-4.0, \
            str(round((i.get_height()/total), 1))+'%',
            fontsize=12,
            color='white',
           weight = 'bold',
           size = 14)
plt.title('Churn Rate', fontsize=14)    
plt.show()
    
#Churn vs Tenure:The customers who do not churn,tend to stay for a longer tenure with the telecom company.
churnTenure=sns.boxplot(x = dataChurn.Churn, y = dataChurn.tenure)
print(churnTenure)
plt.title('Churn vs Tenure', fontsize=14)
plt.show()

# Churn by Contract Type,The customers who have a month to month contract have a very high churn rate.

colors = ['#4D3425','#E4512B']
contract_churn =dataChurn.groupby(['Contract','Churn']).size().unstack()

ax = (contract_churn.T*100.0 / contract_churn.T.sum()).T.plot(kind='bar', width = 0.3,stacked = True,rot = 0, figsize = (10,6),color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='best',prop={'size':14},title = 'Churn')
ax.set_ylabel(' Percentage of Customers',size = 14)
ax.set_title('Churn by Contract Type',size = 14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height), color = 'white',weight = 'bold',size = 14)
plt.show()

#Churn by Seniority: Senior Citizens have almost double the churn rate than younger population.

colors = ['#4D3425','#E4512B']
seniority_churn =dataChurn.groupby(['SeniorCitizen','Churn']).size().unstack()

ax = (seniority_churn.T*100.0 / seniority_churn.T.sum()).T.plot(kind='bar', width = 0.2, stacked = True, rot = 0, figsize = (8,6),color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='center',prop={'size':14},title = 'Churn')
ax.set_ylabel('Percentage of Customers')
ax.set_title('Churn by Seniority Level',size = 14)
plt.show()

#Churn by Monthly Charges: Higher percentage of customers churn when the monthly charges are high.
ax = sns.kdeplot(dataChurn.MonthlyCharges[(dataChurn["Churn"] == 'No') ],color="Red", shade = True)
ax = sns.kdeplot(dataChurn.MonthlyCharges[(dataChurn["Churn"] == 'Yes') ],ax =ax, color="Blue", shade= True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Monthly Charges')
ax.set_title('Distribution of monthly charges by churn')
plt.show()

# Churn by Total Charges: It seems that there is higer churn when the total charges are lower.

ax = sns.kdeplot(dataChurn.TotalCharges[(dataChurn["Churn"] == 'No') ],color="Red", shade = True)
ax = sns.kdeplot(dataChurn.TotalCharges[(dataChurn["Churn"] == 'Yes') ],ax =ax, color="Blue", shade= True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Total Charges')
ax.set_title('Distribution of total charges by churn')
plt.show()

#QUESTION 3

# We will use the data frame where we had created dummy variables
y = df_dummies['Churn'].values
X = df_dummies.drop(columns = ['Churn'])

# Scaling all the variables to a range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
features = X.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features

# Create Train & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Running logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, y_train)
from sklearn import metrics
prediction_test = model.predict(X_test)
# Print the prediction accuracy
print (metrics.accuracy_score(y_test, prediction_test))

# To get the weights of all the variables
weights = pd.Series(model.coef_[0],index=X.columns.values)
#print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))

from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
model_rf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
#print (metrics.accuracy_score(y_test, prediction_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

from sklearn.svm import SVC

model.svm = SVC(kernel='linear') 
model.svm.fit(X_train,y_train)
preds = model.svm.predict(X_test)
#metrics.accuracy_score(y_test, preds)

# Create the Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix  
#print(confusion_matrix(y_test,preds))  

#QUESTION 4
#Extreme Gradient Boosting “XGBOOST” model

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)
metrics.accuracy_score(y_test, preds)

#QUESTION 4
#Evaluate the model’s accuracy and calculate the AUC value

from sklearn import metrics
prediction_test=model.predict(X_test)
print('Precision : '+str(round(metrics.precision_score(y_test,prediction_test.round()),5)))
print('Accuracy : '+str(round(metrics.accuracy_score(y_test,prediction_test.round()),5)))
print('Recall : '+str(round(metrics.recall_score(y_test,prediction_test.round(),average='binary'),5)))
print('F1 Score : '+str(round(metrics.f1_score(y_test,prediction_test.round(),average='binary'),5)))
print('ROC_AUC : '+str(round(metrics.roc_auc_score(y_test,prediction_test.round()),5)))

#QUESTION 5 AND QUESTION 6

#Create a web based platform to host the model using heroku or Streamlite.
#Allow users to use the application to enter new data and your model 

pickle.dump(model,open('CustomerChurn.pkl','wb'))




  







