#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Create DF (import CSV dataset)
lead_df = pd.read_csv("Leads.csv")
lead_df.head()


# In[4]:


lead_df.shape


# In[5]:


lead_df.info()


# In[8]:


lead_df.describe()


# In[9]:


# Checking for duplicates
sum(lead_df.duplicated(subset= 'Prospect ID')) == 0


# In[10]:


# Checking for duplicates
sum(lead_df.duplicated(subset= 'Lead Number')) == 0


# ### Exploratory Data Analysis

# #### Data Cleaning & Treatment

# In[11]:


lead_df.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)


# In[12]:


lead_df = lead_df.replace('Select', np.nan)


# In[13]:


lead_df.isnull().sum()


# In[14]:


#checking percentage of null values in each column

round(100*(lead_df.isnull().sum()/len(lead_df.index)), 2)


# In[15]:


#dropping cols with more than 45% missing values

cols=lead_df.columns

for i in cols:
    if((100*(lead_df[i].isnull().sum()/len(lead_df.index))) >= 45):
        lead_df.drop(i, 1, inplace = True)


# In[16]:


#checking null values percentage

round(100*(lead_df.isnull().sum()/len(lead_df.index)), 2)


# In[17]:


#checking value counts of Country column

lead_df['Country'].value_counts(dropna=False)


# In[18]:


#plotting spread of Country columnn 
plt.figure(figsize=(15,5))
s1=sns.countplot(lead_df.Country)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[19]:


# Since India is the most common occurence among the non-missing values we can impute all missing values with India

lead_df['Country'] = lead_df['Country'].replace(np.nan,'India')


# In[20]:


#plotting spread of Country columnn after replacing NaN values

plt.figure(figsize=(15,5))
s1=sns.countplot(lead_df.Country, hue=lead_df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[21]:


#creating a list of columns to be droppped

cols_to_drop=['Country']


# In[22]:


#checking value counts of "City" column

lead_df['City'].value_counts(dropna=False)


# In[23]:


lead_df['City'] = lead_df['City'].replace(np.nan,'Mumbai')


# In[24]:


#plotting spread of City columnn after replacing NaN values

plt.figure(figsize=(10,5))
s1=sns.countplot(lead_df.City, hue=lead_df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[25]:


#checking value counts of Specialization column

lead_df['Specialization'].value_counts(dropna=False)


# In[26]:


lead_df['Specialization'] = lead_df['Specialization'].replace(np.nan, 'Not Specified')


# In[27]:


#plotting spread of Specialization columnn 

plt.figure(figsize=(15,5))
s1=sns.countplot(lead_df.Specialization, hue=lead_df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[28]:


#combining Management Specializations because they show similar trends

lead_df['Specialization'] = lead_df['Specialization'].replace(['Finance Management','Human Resource Management','Marketing Management','Operations Management',
'IT Projects Management','Supply Chain Management','Healthcare Management','Hospitality Management','Retail Management'] ,'Management_Specializations')


# In[29]:


plt.figure(figsize=(15,5))
s1=sns.countplot(lead_df.Specialization, hue=lead_df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[30]:


lead_df['What is your current occupation'].value_counts(dropna=False)


# In[31]:


lead_df['What is your current occupation'] = lead_df['What is your current occupation'].replace(np.nan, 'Unemployed')


# In[32]:


lead_df['What is your current occupation'].value_counts(dropna=False)


# In[33]:


#visualizing count of Variable based on Converted value

s1=sns.countplot(lead_df['What is your current occupation'], hue=lead_df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[34]:


lead_df['What matters most to you in choosing a course'].value_counts(dropna=False)


# In[35]:


lead_df['What matters most to you in choosing a course'] = lead_df['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')


# In[36]:


#visualizing count of Variable based on Converted value

s1=sns.countplot(lead_df['What matters most to you in choosing a course'], hue=lead_df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[37]:


lead_df['What matters most to you in choosing a course'].value_counts(dropna=False)


# In[38]:


cols_to_drop.append('What matters most to you in choosing a course')
cols_to_drop


# In[39]:


lead_df['Tags'].value_counts(dropna=False)


# In[40]:


lead_df['Tags'] = lead_df['Tags'].replace(np.nan,'Not Specified')


# In[41]:


plt.figure(figsize=(15,5))
s1=sns.countplot(lead_df['Tags'], hue=lead_df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[42]:


lead_df['Tags'] = lead_df['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',
                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',
                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',
                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',
                                    'University not recognized'], 'Other_Tags')

lead_df['Tags'] = lead_df['Tags'].replace(['switched off',
                                      'Already a student',
                                       'Not doing further education',
                                       'invalid number',
                                       'wrong number given',
                                       'Interested  in full time MBA'] , 'Other_Tags')


# In[43]:


#checking percentage of missing values
round(100*(lead_df.isnull().sum()/len(lead_df.index)), 2)


# In[44]:


#checking value counts of Lead Source column

lead_df['Lead Source'].value_counts(dropna=False)


# In[45]:


#replacing Nan Values and combining low frequency values
lead_df['Lead Source'] = lead_df['Lead Source'].replace(np.nan,'Others')
lead_df['Lead Source'] = lead_df['Lead Source'].replace('google','Google')
lead_df['Lead Source'] = lead_df['Lead Source'].replace('Facebook','Social Media')
lead_df['Lead Source'] = lead_df['Lead Source'].replace(['bing','Click2call','Press_Release',
                                                     'youtubechannel','welearnblog_Home',
                                                     'WeLearn','blog','Pay per Click Ads',
                                                    'testone','NC_EDM'] ,'Others')  


# In[46]:


#visualizing count of Variable based on Converted value
plt.figure(figsize=(15,5))
s1=sns.countplot(lead_df['Lead Source'], hue=lead_df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[47]:


# Last Activity:

lead_df['Last Activity'].value_counts(dropna=False)


# In[48]:


#replacing Nan Values and combining low frequency values

lead_df['Last Activity'] = lead_df['Last Activity'].replace(np.nan,'Others')
lead_df['Last Activity'] = lead_df['Last Activity'].replace(['Unreachable','Unsubscribed',
                                                        'Had a Phone Conversation', 
                                                        'Approached upfront',
                                                        'View in browser link Clicked',       
                                                        'Email Marked Spam',                  
                                                        'Email Received','Resubscribed to emails',
                                                         'Visited Booth in Tradeshow'],'Others')


# In[49]:


lead_df['Last Activity'].value_counts(dropna=False)


# In[50]:


round(100*(lead_df.isnull().sum()/len(lead_df.index)), 2)


# In[51]:


#Drop all rows which have Nan Values. Since the number of Dropped rows is less than 2%, it will not affect the model
lead_df = lead_df.dropna()


# In[52]:


round(100*(lead_df.isnull().sum()/len(lead_df.index)), 2)


# In[53]:


lead_df['Lead Origin'].value_counts(dropna=False)


# In[54]:


#visualizing count of Variable based on Converted value

plt.figure(figsize=(8,5))
s1=sns.countplot(lead_df['Lead Origin'], hue=lead_df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[55]:


plt.figure(figsize=(15,5))

ax1=plt.subplot(1, 2, 1)
ax1=sns.countplot(lead_df['Do Not Call'], hue=lead_df.Converted)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)

ax2=plt.subplot(1, 2, 2)
ax2=sns.countplot(lead_df['Do Not Email'], hue=lead_df.Converted)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
plt.show()


# In[56]:


lead_df['Do Not Call'].value_counts(dropna=False)


# In[57]:


#checking value counts for Do Not Email
lead_df['Do Not Email'].value_counts(dropna=False)


# In[58]:


cols_to_drop.append('Do Not Call')
cols_to_drop


# In[59]:


lead_df.Search.value_counts(dropna=False)


# In[60]:


lead_df.Magazine.value_counts(dropna=False)


# In[61]:


lead_df['Newspaper Article'].value_counts(dropna=False)


# In[62]:


lead_df['X Education Forums'].value_counts(dropna=False)


# In[63]:


lead_df['Newspaper'].value_counts(dropna=False)


# In[64]:


lead_df['Digital Advertisement'].value_counts(dropna=False)


# In[65]:


lead_df['Through Recommendations'].value_counts(dropna=False)


# In[66]:


lead_df['Receive More Updates About Our Courses'].value_counts(dropna=False)


# In[67]:


lead_df['Update me on Supply Chain Content'].value_counts(dropna=False)


# In[68]:


lead_df['Get updates on DM Content'].value_counts(dropna=False)


# In[69]:


lead_df['I agree to pay the amount through cheque'].value_counts(dropna=False)


# In[70]:


lead_df['A free copy of Mastering The Interview'].value_counts(dropna=False)


# In[71]:


#adding imbalanced columns to the list of columns to be dropped

cols_to_drop.extend(['Search','Magazine','Newspaper Article','X Education Forums','Newspaper',
                 'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',
                 'Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque'])


# In[72]:


#checking value counts of last Notable Activity
lead_df['Last Notable Activity'].value_counts()


# In[73]:


#clubbing lower frequency values

lead_df['Last Notable Activity'] = lead_df['Last Notable Activity'].replace(['Had a Phone Conversation',
                                                                       'Email Marked Spam',
                                                                         'Unreachable',
                                                                         'Unsubscribed',
                                                                         'Email Bounced',                                                                    
                                                                       'Resubscribed to emails',
                                                                       'View in browser link Clicked',
                                                                       'Approached upfront', 
                                                                       'Form Submitted on Website', 
                                                                       'Email Received'],'Other_Notable_activity')


# In[74]:


#visualizing count of Variable based on Converted value

plt.figure(figsize = (14,5))
ax1=sns.countplot(x = "Last Notable Activity", hue = "Converted", data = lead_df)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
plt.show()


# In[75]:


lead_df['Last Notable Activity'].value_counts()


# In[76]:


cols_to_drop


# In[77]:


#dropping columns
lead_df = lead_df.drop(cols_to_drop,1)
lead_df.info()


# ## Numerical Attributes Analysis:

# In[78]:


#Check the % of Data that has Converted Values = 1:

Converted = (sum(lead_df['Converted'])/len(lead_df['Converted'].index))*100
Converted


# In[79]:


#Checking correlations of numeric values
# figure size
plt.figure(figsize=(10,8))

# heatmap
sns.heatmap(lead_df.corr(), cmap="YlGnBu", annot=True)
plt.show()


# In[80]:


#Total Visits
#visualizing spread of variable

plt.figure(figsize=(6,4))
sns.boxplot(y=lead_df['TotalVisits'])
plt.show()


# In[81]:


#checking percentile values for "Total Visits"

lead_df['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[89]:


#Outlier Treatment: Remove top & bottom 1% of the Column Outlier values

Q3 = lead_df.TotalVisits.quantile(0.99)
lead_df = lead_df[(lead_df.TotalVisits <= Q3)]
Q1 = lead_df.TotalVisits.quantile(0.01)
lead_df = lead_df[(lead_df.TotalVisits >= Q1)]
sns.boxplot(y=lead_df['TotalVisits'])
plt.show()


# In[90]:


lead_df.shape


# In[91]:


lead_df['Total Time Spent on Website'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[92]:


plt.figure(figsize=(6,4))
sns.boxplot(y=lead_df['Total Time Spent on Website'])
plt.show()


# In[93]:


#checking spread of "Page Views Per Visit"

lead_df['Page Views Per Visit'].describe()


# In[94]:


plt.figure(figsize=(6,4))
sns.boxplot(y=lead_df['Page Views Per Visit'])
plt.show()


# In[95]:


Q3 = lead_df['Page Views Per Visit'].quantile(0.99)
lead_df = lead_df[lead_df['Page Views Per Visit'] <= Q3]
Q1 = lead_df['Page Views Per Visit'].quantile(0.01)
lead_df = lead_df[lead_df['Page Views Per Visit'] >= Q1]
sns.boxplot(y=lead_df['Page Views Per Visit'])
plt.show()


# In[96]:


sns.boxplot(y = 'TotalVisits', x = 'Converted', data = lead_df)
plt.show()


# In[97]:


#checking Spread of "Total Time Spent on Website" vs Converted variable

sns.boxplot(x=lead_df.Converted, y=lead_df['Total Time Spent on Website'])
plt.show()


# In[98]:


#checking Spread of "Page Views Per Visit" vs Converted variable

sns.boxplot(x=lead_df.Converted,y=lead_df['Page Views Per Visit'])
plt.show()


# In[99]:


#checking missing values in leftover columns/

round(100*(lead_df.isnull().sum()/len(lead_df.index)),2)


# ### Dummy Variable Creation

# In[101]:


#getting a list of categorical columns

cat_cols= lead_df.select_dtypes(include=['object']).columns
cat_cols


# In[102]:


varlist =  ['A free copy of Mastering The Interview','Do Not Email']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
lead_df[varlist] = lead_df[varlist].apply(binary_map)


# In[103]:


#getting dummies and dropping the first column and adding the results to the master dataframe
dummy = pd.get_dummies(lead_df[['Lead Origin','What is your current occupation',
                             'City']], drop_first=True)

lead_df = pd.concat([lead_df,dummy],1)


# In[104]:


dummy = pd.get_dummies(lead_df['Specialization'], prefix  = 'Specialization')
dummy = dummy.drop(['Specialization_Not Specified'], 1)
lead_df = pd.concat([lead_df, dummy], axis = 1)


# In[105]:


dummy = pd.get_dummies(lead_df['Lead Source'], prefix  = 'Lead Source')
dummy = dummy.drop(['Lead Source_Others'], 1)
lead_df = pd.concat([lead_df, dummy], axis = 1)


dummy = pd.get_dummies(lead_df['Last Activity'], prefix  = 'Last Activity')
dummy = dummy.drop(['Last Activity_Others'], 1)
lead_df = pd.concat([lead_df, dummy], axis = 1)


dummy = pd.get_dummies(lead_df['Last Notable Activity'], prefix  = 'Last Notable Activity')
dummy = dummy.drop(['Last Notable Activity_Other_Notable_activity'], 1)
lead_df = pd.concat([lead_df, dummy], axis = 1)


dummy = pd.get_dummies(lead_df['Tags'], prefix  = 'Tags')
dummy = dummy.drop(['Tags_Not Specified'], 1)
lead_df = pd.concat([lead_df, dummy], axis = 1)


# In[106]:


#dropping the original columns after dummy variable creation

lead_df.drop(cat_cols,1,inplace = True)


# In[107]:


lead_df.head()


# In[ ]:





# ## Train-Test Split & Logistic Regression Model Building:

# In[108]:


from sklearn.model_selection import train_test_split

# Putting response variable to y
y = lead_df['Converted']

y.head()

X=lead_df.drop('Converted', axis=1)


# In[109]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[110]:


X_train.info()


# ### Scaling 

# In[113]:


#scaling numeric columns

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols=X_train.select_dtypes(include=['float64', 'int64']).columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

X_train.head()


# In[112]:


import warnings
warnings.filterwarnings('ignore')


# ## Model Building using Stats Model & RFE:

# In[114]:


import statsmodels.api as sm


# In[115]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE
rfe = RFE(logreg, 15)  # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[116]:


rfe.support_


# In[117]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[118]:


#list of RFE supported columns
col = X_train.columns[rfe.support_]
col


# In[119]:


X_train.columns[~rfe.support_]


# In[120]:


#BUILDING MODEL #1

X_train_sm = sm.add_constant(X_train[col])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[ ]:





# In[122]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[123]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[124]:


#dropping variable with high VIF

col = col.drop('Last Notable Activity_SMS Sent',1)


# In[125]:


#BUILDING MODEL #2
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[126]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[127]:


# Getting the Predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[128]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[129]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# In[130]:


y_train_pred_final['Predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# ### Confusion Metrics

# In[131]:


from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# In[132]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[133]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[134]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[135]:


# Let us calculate specificity
TN / float(TN+FP)


# In[136]:


# Calculate False Postive Rate - predicting conversion when customer does not have convert
print(FP/ float(TN+FP))


# In[137]:


# positive predictive value 
print (TP / float(TP+FP))


# In[138]:


# Negative predictive value
print (TN / float(TN+ FN))


# ### PLOTTING ROC CURVE

# In[139]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[140]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )


# In[141]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# ### Finding Optimal Cutoff Point

# In[142]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[143]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[144]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[145]:


y_train_pred_final['final_Predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()


# In[146]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))

y_train_pred_final[['Converted','Converted_prob','Prospect ID','final_Predicted','Lead_Score']].head()


# In[147]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[148]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion2


# In[149]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[150]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[151]:


# Let us calculate specificity
TN / float(TN+FP)


# ## Observation
The ROC curve has a value of 0.97, which is very good.
We have the following values for the Train Data:

Accuracy : 92.29%
Sensitivity : 91.70%
Specificity : 92.66%
# In[153]:


# Calculate False Postive Rate - predicting conversion when customer does not have convert
print(FP/ float(TN+FP))


# In[154]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[155]:


# Negative predictive value
print (TN / float(TN+ FN))


# In[156]:


#Looking at the confusion matrix again

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion


# In[159]:


##### Precision
TP / TP + FP

confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[160]:


##### Recall
TP / TP + FN

confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[161]:


from sklearn.metrics import precision_score, recall_score


# In[162]:


precision_score(y_train_pred_final.Converted , y_train_pred_final.final_Predicted)


# In[163]:


recall_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[164]:


from sklearn.metrics import precision_recall_curve


# In[165]:


y_train_pred_final.Converted, y_train_pred_final.final_Predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)

plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[166]:


#scaling test set

num_cols=X_test.select_dtypes(include=['float64', 'int64']).columns
X_test[num_cols] = scaler.fit_transform(X_test[num_cols])
X_test.head()


# In[167]:


X_test = X_test[col]
X_test.head()


# In[168]:


X_test_sm = sm.add_constant(X_test)


# ## PREDICTIONS ON TEST SET

# In[169]:


y_test_pred = res.predict(X_test_sm)


# In[170]:


y_test_pred[:10]


# In[171]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[172]:


# Let's see the head
y_pred_1.head()


# In[173]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[174]:


# Putting CustID to index
y_test_df['Prospect ID'] = y_test_df.index


# In[175]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[176]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[177]:


y_pred_final.head()


# In[178]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})


# In[179]:


y_pred_final.head()


# In[180]:


# Rearranging the columns
y_pred_final = y_pred_final[['Prospect ID','Converted','Converted_prob']]
y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))


# In[181]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[182]:


y_pred_final['final_Predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.3 else 0)


# In[183]:


y_pred_final.head()


# In[184]:


metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[185]:


confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_Predicted )
confusion2


# In[186]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[187]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[188]:


# Let us calculate specificity
TN / float(TN+FP)


# In[189]:


precision_score(y_pred_final.Converted , y_pred_final.final_Predicted)


# In[190]:


recall_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# ## Observation:
# After running the model on the Test Data these are the figures we obtain:
# 
# Accuracy : 92.78%
# Sensitivity : 91.55%
# Specificity : 92.83%

# In[ ]:




