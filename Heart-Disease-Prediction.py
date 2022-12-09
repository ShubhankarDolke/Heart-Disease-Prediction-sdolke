#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import numpy as np
import pandas as pd


# In[3]:


# loading the dataset
df = pd.read_csv("heart.csv")


# In[4]:


# Returns number of rows and columns of the dataset
df.shape


# In[6]:


# returns an object with all of the column headers
df.columns


# In[8]:


# returns different datatypes for each colums (float, int, strings)
df.dtypes


# In[9]:


# return the first X number of rows when head(x). Without a no
df.head()


# In[10]:


# returns the last x no. of rows when tail(x). without a no. 
df.tail()


# In[11]:


# returns true for a column having null values, else false
df.isnull().any()


# In[12]:


# returns basic information on all columns 
df.info()


# In[13]:


# returns badic statistics on numeric columns
df.describe().T


# In[14]:


# importing libraries for data visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[15]:


# plotting histogram for the entire dataset
fig = plt.figure(figsize = (15,15))
ax = fig.gca()
g = df.hist(ax=ax)


# In[16]:


# visualization to check if the dataset is balanced or not
g = sns.countplot(x= 'target', data=df)
plt.xlabel('Target')
plt.ylabel('Count')


# In[18]:


#fearture selection
# selecting corelated feature using heatmap
#get correlation of all the features of th edataset
corr_matrix = df.corr()
top_corr_feature = corr_matrix.index

# plotting the heatmap
plt.figure(figsize=(20,20))
sns.heatmap(data=df[top_corr_feature].corr(), annot= True, cmap='RdYlGn')


# In[19]:


# data preprocessing
#Handling categorical features
dataset = pd.get_dummies(df, columns= ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])


# In[20]:


# feature scaling
dataset.columns


# In[26]:


from sklearn.preprocessing import StandardScaler
standScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standScaler.fit_transform(dataset[columns_to_scale])


# In[28]:


dataset.head()


# In[29]:


# spliting the dataset into dependent and independent feature
x = dataset.drop('target', axis=1)
y = dataset['target']


# In[30]:


#model building
#1:KNeighbors Classifier
#2:Desion Tree Classifier
#3:Random forest Classifier

#importing libraries 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# In[33]:


#finding the best accuracy for knn algorithum using cross_val_score
knn_scores = []
for i in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors=i)
    cvs_scores = cross_val_score(knn_classifier, x, y, cv=10)
    knn_scores.append(round(cvs_scores.mean(),3))


# In[37]:


#plotting the result of knn_scores
plt.figure(figsize = (20,15))
plt.plot([k for k in range (1,21)], knn_scores, color = "red" )
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1,21)])
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Scores')
plt.title('K neighbors classifier scores for diffreent K values ')
    
    


# In[39]:


# training the knn classifier model with k value as 12 
knn_classifier = KNeighborsClassifier(n_neighbors= 12)
cvs_scores = cross_val_score(knn_classifier, x, y, cv = 10)
print("KNeighbors Classifiers Accuracy with k=12 is : {}%". format(round(cvs_scores.mean(), 4)*100))


# In[41]:


# importing libraries for decision tree
from sklearn.tree import DecisionTreeClassifier


# In[46]:


# finding the best accuracy for decision tree algorithum using cross_val_score
decision_scores = []
for i in range(1, 11):
    decision_classifier = DecisionTreeClassifier(max_depth = i )
    cvs_scores = cross_val_score(decision_classifier, x, y, cv = 10)
    decision_scores.append(round(cvs_scores.mean(), 3))
    
                           


# In[49]:


#plotting the result of decision scores
plt.figure(figsize=(20,15))
plt.plot([i for i in range(1,11)], decision_scores, color = 'red')
for i in range (1,11):
    plt.text(i, decision_scores[i-1], (i, decision_scores[i-1]))
plt.xticks([i for i in range (1,11)])
plt.xlabel("Depth of Decision Tree (N) ")
plt.ylabel("Scores")
plt.title("Decision Tress classifier values for Different depth values")


# In[52]:


#training the decision tree classifier model with max_depth value as 3
decision_calssifier = DecisionTreeClassifier(max_depth=3)
cvs_scores = cross_val_score(decision_classifier, x, y, cv = 10)
print("Decision tree classifier Accuracy with max_depth = 3 is: {}%".format(round(cvs_scores.mean(), 4)*100))


# In[53]:


#Random forest Classifier 
#importing libraries for Random Classifier
from sklearn.ensemble import RandomForestClassifier


# In[56]:


#finding the best accuracy for random forest algorithm using cross_val_score
forest_score= []
for i in range(10,101,10):
    forest_classifier = RandomForestClassifier(n_estimators = i)
    cvs_scores = cross_val_score(forest_classifier, x, y, cv = 5)
    forest_score.append(round(cvs_scores.mean(), 3))
    


# In[60]:


#plotting the result of forest_scores
plt.figure(figsize = (20,15))
plt.plot([n for n in range (10, 101, 10)], forest_score, color = "red")
for i in range(1, 11):
    plt.text(i*10, forest_score[i-1], (i*10, forest_score[i-1]))
plt.xticks([i for i in range(10, 101, 10)])
plt.xlabel("Number of Estimators (N)")
plt.ylabel("Scores")
plt.title("Random Forest Classifier scores for diffrent N values")


# In[61]:


# Training the random forest classifier model with n value as 90 
forest_classifier = RandomForestClassifier(n_estimators= 90)
cvs_scores = cross_val_score(forest_classifier, x, y, cv = 5)
print("Random Forest classifier Accuracy with n_estimators = 90 is: {}%".format(round(cvs_scores.mean(), 4)*100))


# In[ ]:




