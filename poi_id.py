#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle as pkl
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, NMF

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss


from pprint import pprint


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


import os

os.getcwd()


# In[4]:


with open("C:\\Users\\josep\\ud120-projects_modified for 3\\final_project\\final_project_dataset.pkl", "r") as data_file:
    data_dict = pkl.load(data_file)


# In[5]:


# make a pandas dataframe to explore

enron_df = pd.DataFrame.from_dict(data_dict,orient='index')


# In[6]:


# check the data structure

enron_df.info()


# 146 Total entries.  21 total features (columns)

# In[7]:


# Checking the dataset

enron_df.head()


# In[8]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Convert dictionary to DataFrame

enron_df = pd.DataFrame.from_dict(data_dict, orient = 'index', dtype = float)

# reorganize columns

enron_df = enron_df[
['salary',
'bonus',
'long_term_incentive',
'deferred_income',
'deferral_payments',
'loan_advances',
'other',
'expenses',
'director_fees',
'total_payments',
 'exercised_stock_options',
'restricted_stock',
 'restricted_stock_deferred',
 'total_stock_value',
 'email_address',
 'to_messages',
 'shared_receipt_with_poi',
 'from_messages',
 'from_this_person_to_poi',
 'poi',
 'from_poi_to_this_person']]


# In[9]:


# Check the dataset

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

enron_df


# In[10]:


names = sorted(data_dict.keys())  #sort names of Enron employees in dataset by first letter of last name

print('Sorted list of Enron employees by last name:\n')
pprint(names)


# ## Outlier Investigation and Removal
# 
# Study of last two queries reveals two entries of concern:  "The Travel Agency in the Park" and "Total".  
# 
# Will remove as they are obviously not individuals and a majority of the features have 'Nan' as a value.

# In[11]:


data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)


# Next, I want to search each individual datapoint to see if any records exist with similiar concerns as the 2 I removed above.  I will start by searching the feature "Total Payments" to isolate individuals with no entry. 

# In[12]:


# List of people with no total payment data

for entry in data_dict:
    if data_dict[entry]['total_payments'] == 'NaN':
        print(entry)


# Now, I will run another query using an additional feature to drill down and try to isolate individuals with little to no valuable entries.  'Total Stock Value' seems like an obvious place to start since it is potential compensation that may not have been paid out or included in total payment data. 

# In[13]:


# List of people with no total payment data and no stock option data

for entry in data_dict:
    if data_dict[entry]['total_payments'] == 'NaN' and data_dict[entry]['total_stock_value'] == 'NaN':
        print(entry)


# In[14]:


# Investigating CHAN RONNIE

data_dict['CHAN RONNIE']


# #### Most of the entries contain no data.  Entries with data are canceled out by negative values in other columns.  Will remove this datapoint. 

# In[15]:


# Investigating POWERS WILLIAM

data_dict['POWERS WILLIAM']


# #### This individual datapoint does not contain much financial data.  However, there are several relevant values related to communications.  I will retain this datapoint in future investigations.

# In[16]:


# Investigating LOCKHART EUGENE E

data_dict['LOCKHART EUGENE E']


# #### Again, most of the entries contain no data.  Entries with data are canceled out by negative values in other columns.  Will remove this datapoint. 

# In[17]:


# Removing the 2 outlier datapoints as outlined above

data_dict.pop('LOCKHART EUGENE E', 0)
data_dict.pop('CHAN RONNIE', 0)


# In[18]:


names = sorted(data_dict.keys())  #sort names of Enron employees in dataset by first letter of last name

print('Sorted list of Enron employees by last name:\n')
pprint(names)


# #### Verified entries removed.

# In[19]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Convert dictionary to DataFrame
enron_df = pd.DataFrame.from_dict(data_dict, orient = 'index', dtype = float)

# reorganize columns
enron_df = enron_df[
['salary',
'bonus',
'long_term_incentive',
'deferred_income',
'deferral_payments',
'loan_advances',
'other',
'expenses',
'director_fees',
'total_payments',
 'exercised_stock_options',
'restricted_stock',
 'restricted_stock_deferred',
 'total_stock_value',
 'email_address',
 'to_messages',
 'shared_receipt_with_poi',
 'from_messages',
 'from_this_person_to_poi',
 'poi',
 'from_poi_to_this_person']]


# In[20]:


enron_df.info()


# In[21]:


### Total Number of Data Points

print('Total Number of data points: %d' %len(data_dict))


# #### * Verified 21 total Columns -- just like the data structure queries above
# #### * Index query reveals 4 concerning entries now removed.

# In[22]:


# Saving the new dataset without the outliers

my_dataset = data_dict


# ## Feature Exploration

# Now I can begin to isolate Persons of Interest(POIs) based on the boolean feature "POI".

# In[23]:


# Counting the POIs

print("number of poi: {}".format(enron_df[enron_df['poi']==True]['poi'].count()))


# Counting the NON-POIs

print("number of non_poi: {}".format(enron_df[enron_df['poi']==False]['poi'].count()))


# In[24]:


# print the POIs

index = enron_df.index

condition_poi = enron_df["poi"] == True


person_poi = index[condition_poi]

#arrange in alphabetical order

person_poi_list = sorted(person_poi.tolist())

print("The following are individuals labeled as POI (Person of Interest):\n")
print('\n'.join(map(str, person_poi_list)))


# Query of random record to validate features and entries:

# In[25]:


print('Example Value Dictionary of Features:\n')
pprint(data_dict['METTS MARK']) 
pprint(len(data_dict['METTS MARK']))


# In[26]:


# List of original features

features_list = [
 'poi',
 'bonus',
 'deferral_payments',
 'deferred_income',
 'director_fees',
 'email_address',
 'exercised_stock_options',
 'expenses',
 'from_messages',
 'from_poi_to_this_person',
 'from_this_person_to_poi',
 'loan_advances',
 'long_term_incentive',
 'other',
 'restricted_stock',
 'restricted_stock_deferred',
 'salary',
 'shared_receipt_with_poi',
 'to_messages',
 'total_payments',
 'total_stock_value'
]


# In[27]:


# Count number of entries missing values for each feature

features_nan = {}

for name, feature in data_dict.items():
    for feature_key, val in feature.items():
        if val == 'NaN':
            # Assign 0 to value
            feature[feature_key] = 0
            if features_nan.get(feature_key):
                features_nan[feature_key] = features_nan.get(feature_key) + 1
            else:
                features_nan[feature_key] = 1

print('# of Missing Values by Feature:\n')
print("{:<25} {:<5}".format('FEATURE', 'COUNT'))
print('-------------------------------')
for key, value in features_nan.items(): 
    print("{:<25} {:<5}".format(key, value))


# In[28]:


# Replace missing values with a zero

enron_df = enron_df.fillna(0)

# Query to verify

enron_df.head()


# In[29]:


# Re-checking for missing values

features_nan = {}

for name, feature in data_dict.items():
    for feature_key, val in feature.items():
        if val == 'NaN':
            # Assign 0 to value
            feature[feature_key] = 0
            if features_nan.get(feature_key):
                features_nan[feature_key] = features_nan.get(feature_key) + 1
            else:
                features_nan[feature_key] = 1

print('# of Missing Values by Feature:\n')
print("{:<25} {:<5}".format('FEATURE', 'COUNT'))
print('-------------------------------')
for key, value in features_nan.items(): 
    print("{:<25} {:<5}".format(key, value))


# In[30]:


dataPoints = float(len(my_dataset))  ## count of data points
featureCount = float(len(features_list))  ## count of all features

## calculation of poi's and non-poi's

pois = 0.
nonPois = 0.
for i in my_dataset.values():
    if i["poi"] == 1:
        pois += 1.
    else:
        nonPois += 1.

## data exploration

print "Number of Data Points : ", dataPoints
print "Number of Poi's : ", pois
print "Number of non-Poi's : ", nonPois
print "Number of missing Poi values : ", dataPoints - (pois + nonPois)
print "Poi percentage : %", round((pois / (nonPois + pois)), 2) * 100  ## poi's percentage for all records
print "Number of Features : ", featureCount


# #### Verified new dataset contains appropriate number of executive datapoints after outliers removed.  Still retain 21 total features.  POI proves to be a solid feature as no remaining entries are missing data. 
# 
# *Side note:  It's quite concerning that 13% of executives are labeled as POIs!*

# In[31]:


import numpy as np

def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    """ convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
    """


    return_list = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print("error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )

    return np.array(return_list)


def targetFeatureSplit( data ):
    """ 
        given a numpy array like the one returned from
        featureFormat, separate out the first feature
        and put it into its own list (this should be the 
        quantity you want to predict)

        return targets and features as separate lists

        (sklearn can generally handle both lists and numpy arrays as 
        input formats when training/predicting)
    """

    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features


# #### Introduction of included code to format features.  I next want to search for outliers and see if any interesting relationships emerge that I can use as the basis for new features.

# ### Next, I'll do a search for specific outlier entries.

# In[32]:


outlier_POIs = dict()

cases = 0
for column_name in list(enron_df):
    if enron_df[column_name].dtype == 'float' and column_name != 'poi':
        cases +=1
        test_data = enron_df[enron_df[column_name]!=0]
        Q1 = test_data[column_name].quantile(0.25)
        Q3 = test_data[column_name].quantile(0.75)
        IQR = Q3 - Q1

        # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
        query_str = '(@Q1 - 2 * @IQR) >= ' + column_name + ' or  (@Q3 + 2 * @IQR) <=' + column_name
        filtered = test_data.query(query_str)

        print(column_name + ": " + str(len(filtered)))
        filtered = filtered.sort_values(column_name)
        print(filtered[column_name].head(10))
        print("")
        
        for person in filtered.index.tolist():
            try:
               outlier_POIs[person] = 1 + outlier_POIs[person]
            except:
               outlier_POIs[person] = 1 

print("Total Cases:" + str(cases))


# ### Top 10 outlier individuals:

# In[33]:


outlier_POIs_df = pd.DataFrame.from_dict(outlier_POIs, orient = 'index', dtype = int)
outlier_POIs_df.columns = ['Count']
outlier_POIs_df = outlier_POIs_df.sort_values('Count', ascending = False)
outlier_POIs_df.head(10)


# Next, I want to check to see if any powerful correlations exist that can also shed light on potential new features.

# In[34]:


# looking for correlations in the data

enron_df.corr(method ='pearson')


# I'll start by examining the above chart to find any strong correlations between features to use as the basis for any new features.
# 
# An obvious place to start is with the "from_this_person_to_poi" feature, as this will reveal individuals with a lot of communications with POIs.  
# 
# * "loan_advances" emerges as the first obvious choice.  The correlation ratio of 0.935 is extremely strong and is one of the highest on the chart.
# * "from_poi_to_this_person" is also an interesting choice.  The correlation ratio is 0.497, which proves a relationship
# * I will create a new feature named 'percent_poi_emails' to calculate the percentage of total email communications in which POIs were involved
# 
# 

# ### New Feature Creation

# In[35]:


# Creates a new list by adding two lists together

def get_total_list(list1, list2):
    new_list = []
    for i in my_dataset:
        if my_dataset[i][list1] == 'NaN' or my_dataset[i][list2] == 'NaN':
            new_list.append(0.)
        elif my_dataset[i][list1]>=0:
            new_list.append(float(my_dataset[i][list1]) + float(my_dataset[i][list2]))
    return new_list


# In[36]:


# Establishing POI email list

poi_emails_list = get_total_list('from_this_person_to_poi', 'from_poi_to_this_person')


# In[37]:


# Establishing total email list

total_emails_list = get_total_list('to_messages', 'from_messages')


# In[38]:


# Divides one list by another list
def fraction_list(list1, list2):
    new_list = []
    for i in range(0,len(list1)):
        if list2[i] == 0.0:
            new_list.append(0.0)
        else:
            new_list.append(float(list1[i])/float(list2[i]))
    return new_list


# In[39]:


# Getting new list by dividing previously created lists

percent_poi_emails = fraction_list(poi_emails_list, total_emails_list)


# In[40]:


len(percent_poi_emails)


# In[41]:


# Adding this new feature to the dataset

count = 0
for i in my_dataset:
    my_dataset[i]['percent_poi_emails'] = percent_poi_emails[count]
    count += 1


# In[42]:


# Printing all features

print(len((my_dataset['SKILLING JEFFREY K'].keys())))
print(my_dataset['SKILLING JEFFREY K'].keys())


# #### Verified 'loan_advances_feature' now added to list.
# 
# Next, I will add the new feature to features_list and remove the "email_address" feature as it has no value for this exercise.  I will also remove the 'other' feature as it is undefined and might skew the data. 

# In[43]:


# Adding new feature to features_list and removing "email_address" feature 

features_list = [
 'poi',
 'bonus',
 'deferral_payments',
 'deferred_income',
 'director_fees',
 'exercised_stock_options',
 'expenses',
 'from_messages',
 'from_poi_to_this_person',
 'from_this_person_to_poi',
 'loan_advances',
 'long_term_incentive',
 'percent_poi_emails',
 'restricted_stock',
 'restricted_stock_deferred',
 'salary',
 'shared_receipt_with_poi',
 'to_messages',
 'total_payments',
 'total_stock_value'
]


# Now, I want to use Decision Tree to see which set of features best suites my needs:

# In[44]:


# Decision Tree with original set of features

import numpy as np
np.random.seed(42)
from time import time
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# features_list

features_list = ['poi','salary', 'from_poi_to_this_person', 'from_this_person_to_poi', 'to_messages', 
                 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 
                 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 
                 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 
                 'director_fees', 'deferred_income', 'long_term_incentive']

data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# split data inton training and testing

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

# choose decision tree

from sklearn.tree import DecisionTreeClassifier
t0 = time()
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(acc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)


# In[45]:


# Decision Tree with updated set of features

import numpy as np
np.random.seed(42)
from time import time
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# features_list

features_list = ['poi', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees',
                 'exercised_stock_options', 'expenses', 'from_messages', 'from_poi_to_this_person',
                 'from_this_person_to_poi', 'loan_advances', 'long_term_incentive', 'percent_poi_emails',
                 'restricted_stock', 'restricted_stock_deferred', 'salary', 'shared_receipt_with_poi',
                 'to_messages', 'total_payments', 'total_stock_value'
                ]

data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# split data inton training and testing

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

# choose decision tree

from sklearn.tree import DecisionTreeClassifier
t0 = time()
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(acc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)


# In[46]:


# Decision Tree with original set of features minus email address and other

import numpy as np
np.random.seed(42)
from time import time
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# features_list

features_list = ['poi', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees',
                 'exercised_stock_options', 'expenses', 'from_messages', 'from_poi_to_this_person',
                 'from_this_person_to_poi', 'loan_advances', 'long_term_incentive',
                 'restricted_stock', 'restricted_stock_deferred', 'salary', 'shared_receipt_with_poi',
                 'to_messages', 'total_payments', 'total_stock_value'
                ]

data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# split data inton training and testing

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

# choose decision tree

from sklearn.tree import DecisionTreeClassifier
t0 = time()
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(acc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)


# I will now run Select K Best to isolate the most important features from the dataset.

# In[47]:


from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

def selectFeatures(nParam):
    kBest = SelectKBest(k=nParam)
    kBest.fit_transform(features, labels)
    kResult = zip(kBest.get_support(), kBest.scores_, features_list[1:])
    return list(sorted(kResult, key=lambda x: x[1], reverse=True))


resultsAll = selectFeatures("all")

import pprint

pprint.pprint(resultsAll)


# A brief examination of the above results reveals the top five strongest features:
# 
# * 'exercised_stock_options'
# * 'total_stock_value'
# * 'bonus'
# * 'salary'
# * 'deferred_income
# 
# I will now run Decision Tree using the top 4, 5, and 6 features to check which feature set performs the best.

# In[48]:


# features_list top 4
features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary']
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# split data inton training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

# choose decision tree
from sklearn.tree import DecisionTreeClassifier
t0 = time()
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(acc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)


# In[49]:


# Top 5 features

features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 
                 'salary', 'deferred_income']
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# split data inton training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

# choose decision tree
from sklearn.tree import DecisionTreeClassifier
t0 = time()
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(acc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)


# In[50]:


# Top 6 features

features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 
                 'salary', 'deferred_income','long_term_incentive']
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# split data into training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

# choose decision tree
from sklearn.tree import DecisionTreeClassifier
t0 = time()
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(acc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)


# In[51]:


# features_list top 4 with new feature

features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'percent_poi_emails']
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# split data inton training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

# choose decision tree
from sklearn.tree import DecisionTreeClassifier
t0 = time()
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(acc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)


# In[52]:


# features_list top 5 with new feature

features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 
                 'salary', 'deferred_income', 'percent_poi_emails']
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# split data inton training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

# choose decision tree
from sklearn.tree import DecisionTreeClassifier
t0 = time()
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(acc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)


# ### New Feature Test Results
# 
# Using DecisionTreeClassifier in combination with Select K Best, I determined using the top 5 features, in conjunction with my newly created feature 'percent_poi_emails' (as outlined above), performed the best out of the bunch.  The test results were as follows:
# 
# * Accuracy: 0.81
# * Precision:  0.40
# * Recall:  0.50

# In[53]:


# establish list of features used

features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'percent_poi_emails']
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# split data inton training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

# choose decision tree
from sklearn.tree import DecisionTreeClassifier
t0 = time()
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(acc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)


# ## Algorithms

# ### Classifiers

# In[54]:


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# In[55]:


randomState = 42
nFeatures = 5


# In[56]:


## returns stored classifiers and their parameters

def createClassifiersAndParams():
    rec = {}  ## store all in this list

    def addToDict(name, clf, params):  ## helper for adding to the list
        ## name : name of the classifier
        ## clf : classifier object
        ## params : parameters object
        rec[name] = {"clf": clf,
                     "params": params}

    ##naive bayes
    addToDict("NaiveBayes", GaussianNB(), {})

    ##support vector machines
    addToDict("SVM", SVC(), {'kernel': ['poly', 'rbf', 'sigmoid'],
                             'cache_size': [7000],
                             'tol': [0.0001, 0.001, 0.005, 0.05],
                             'decision_function_shape': ['ovo', 'ovr'],
                             'random_state': [randomState],
                             'verbose' : [False],
                             'C': [100, 1000, 10000]
                             })

    ##DecisionTree
    addToDict("DecisionTree", DecisionTreeClassifier(), {'criterion': ['gini', 'entropy'],
                                                         'splitter': ['best', 'random'],
                                                         'min_samples_split': [2, 10, 20],
                                                         'max_depth': [None, 2, 4, 8, 16],
                                                         'min_samples_leaf': [1, 3, 5, 7, 9],
                                                         'max_leaf_nodes': [None, 6, 12, 24],
                                                         'random_state': [randomState]})
   
    ##AdaBoost
    addToDict("AdaBoost", AdaBoostClassifier(), {'n_estimators': [25, 50, 100],
                                                 'algorithm': ['SAMME', 'SAMME.R'],
                                                 'learning_rate': [.2, .5, 1, 1.4, 2.],
                                                 'random_state': [randomState]})
   

    
    ##LogisticRegression
    addToDict("LogisticRegression", LogisticRegression(), {'penalty': ['l1', 'l2'],
                                                           'tol': [0.0001, 0.0005, 0.001, 0.005],
                                                           'C': [1, 10, 100, 1000, 10000, 100000, 1000000],
                                                           'fit_intercept': [True, False],
                                                           'solver': ['liblinear'],
                                                           'class_weight': [None, 'balanced'],
                                                           'verbose': [False],
                                                           'random_state': [randomState]
                                                           })

    return rec


# In[57]:


def train(clf, params, features_train, labels_train):  ## trainer function
    # train
    t0 = time()  ## timer for calculating trainin time
    clft = GridSearchCV(clf, params)  ## grid search with parameters
    clft = clft.fit(features_train, labels_train)  ## training the searcher for best fit

    # print "training time:", round(time() - t0, 3), "s"  ##print the training time
    return clft, (time() - t0)  ## return best parameters


# In[58]:


def predict(clf, features_test):  ## predictor function
    # predict
    t0 = time()  ##timer for calculating the prediction time
    pred = clf.predict(features_test)  ## predict the result for test features
    # print "predicting time:", round(time() - t0, 3), "s"  ## print the prediction time
    return pred, (time() - t0)  ## return all predictions


# In[59]:


def scores(pred, labels_test):  ## scoring function
    ## inspired from tester script

    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    total_predictions = 0
    accuracy = 0.
    precision = 0.
    recall = 0.
    f1 = 0.
    f2 = 0.

    ## get the prediction details
    for prediction, truth in zip(pred, labels_test):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1
        else:
            print "Warning: Found a predicted label not == 0 or 1."
            print "All predictions should take value 0 or 1."
            print "Evaluating performance for processed predictions:"
            break

    try:
        ##calculate each metric
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
        precision = 1.0 * true_positives / (true_positives + false_positives)
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
        f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)

    except:
        print "Got a divide by zero"


    return accuracy, precision, recall, f1, f2  ## return all scores


# In[60]:


## function for doing everything in a single area
## gets classifier, their parameters, train and test features
## returns the best-fit classifier, its parameters and the scores
def doTheMath(clf, params, features_train, labels_train, features_test, labels_test):
    clft, trainTime = train(clf, params, features_train, labels_train)  ## call train function with given params
    preds, predictTime = predict(clft, features_test)  ## make predictions with given classifier

    accuracy, precision, recall, f1, f2 = scores(preds, labels_test)  ## calculate the scores

    ## return best-fit classifier, and its parameters , also the score values
    return clft.best_estimator_, clft.best_params_, accuracy, precision, recall, f1, f2, trainTime, predictTime


# In[61]:


allClassifiers = createClassifiersAndParams()  ## create the list of classifiers


# In[62]:


## format the data with only selected features
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)


# In[63]:


features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)


# In[64]:


scaler = MinMaxScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.fit_transform(features_test)


# In[65]:


for x in allClassifiers:  ##loop through all classifiers

    clft = allClassifiers[x]["clf"]  ##get the classifier
    params = allClassifiers[x]["params"]  ##get the parameters

    ## call the function for all processes
    estimator, params, accuracy, precision, recall, f1, f2, trainTime, predictTime = doTheMath(clft,
                                                                                               params,
                                                                                               features_train,
                                                                                               labels_train,
                                                                                               features_test,
                                                                                               labels_test)

    ## record results in dictionary
    allClassifiers[x]["clf"] = estimator
    allClassifiers[x]["params"] = params
    allClassifiers[x]["accuracy"] = accuracy
    allClassifiers[x]["precision"] = precision
    allClassifiers[x]["recall"] = recall
    allClassifiers[x]["f1"] = f1
    allClassifiers[x]["f2"] = f2
    allClassifiers[x]["trainTime"] = trainTime
    allClassifiers[x]["predictTime"] = predictTime


    score = (allClassifiers[x]["f1"] * allClassifiers[x]["precision"] *              allClassifiers[x]["recall"] * allClassifiers[x]["accuracy"]) /             (allClassifiers[x]["trainTime"] + allClassifiers[x]["predictTime"])

    ## store new score in dictionary
    allClassifiers[x]["my_score"] = score

print "| ", x, " | ", round(score, 4), " | ", round(allClassifiers[x]["accuracy"], 3), " | ", round(
allClassifiers[x]["precision"], 3), \
" | ", round(allClassifiers[x]["recall"], 3), " | ", round(allClassifiers[x]["f1"], 3), " | ", round(
allClassifiers[x]["f2"], 3), " | ", \
        round(allClassifiers[x]["trainTime"], 3), " | ", round(allClassifiers[x]["predictTime"], 3), " |"


# The following was created using GridSearchCV to get the parameters tested for each method.  

# In[66]:


for i in allClassifiers:
    if i in ['NaiveBayes', 'SVM', 'AdaBoost', 'RandomForest', 'KMeans', 'DecisionTree', 'LogisticRegression', 'KNeighbors']:
        print allClassifiers[i]["clf"]


# Next, I will test each classifier to find the best performance

# In[67]:


clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=3, min_samples_split=10,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='random')

clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(acc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)
print 'F1 score:', f1_score(labels_test, pred)


# In[68]:


clf = SVC(C=100, cache_size=7000, class_weight=None, coef0=0.0,
  decision_function_shape='ovo', degree=3, gamma='auto_deprecated',
  kernel='poly', max_iter=-1, probability=False, random_state=42,
  shrinking=True, tol=0.0001, verbose=False)

clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(acc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)
print 'F1 score:', f1_score(labels_test, pred)


# In[69]:


clf = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l1', random_state=42, solver='liblinear',
          tol=0.0001, verbose=False, warm_start=False)

clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(acc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)
print 'F1 score:', f1_score(labels_test, pred)


# In[70]:


clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
          n_estimators=25, random_state=42)

clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(acc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)
print 'F1 score:', f1_score(labels_test, pred)


# In[71]:


clf = GaussianNB()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(acc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)
print 'F1 score:', f1_score(labels_test, pred)


# DecisionTree displays the best performance, so that is the classifier I will use:

# In[72]:


features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'percent_poi_emails']
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# split data inton training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

# choose decision tree
from sklearn.tree import DecisionTreeClassifier
t0 = time()
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)
print 'Accuracy: ' + str(acc)
print 'Precision: ', precision_score(labels_test, pred)
print 'Recall: ', recall_score(labels_test, pred)


# # Validation and Evaluation
# 

# In[73]:


import numpy as np

def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    """ convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
    """


    return_list = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print("error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )

    return np.array(return_list)


def targetFeatureSplit( data ):
    """ 
        given a numpy array like the one returned from
        featureFormat, separate out the first feature
        and put it into its own list (this should be the 
        quantity you want to predict)

        return targets and features as separate lists

        (sklearn can generally handle both lists and numpy arrays as 
        input formats when training/predicting)
    """

    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features


# In[74]:


import pickle
import sys
from sklearn.model_selection import StratifiedShuffleSplit
sys.path.append("C:\\Users\\josep\\ud120-projects_modified for 3\\tools\\")
from feature_format import featureFormat, targetFeatureSplit

PERF_FORMAT_STRING = "\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\tRecall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"


# In[75]:


def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(n_splits=folds, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv.split(features, labels):  
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."


# In[76]:


CLF_PICKLE_FILENAME = "my_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"


# In[77]:


def dump_classifier_and_data(clf, dataset, feature_list):
    with open(CLF_PICKLE_FILENAME, "w") as clf_outfile:
        pickle.dump(clf, clf_outfile)
    with open(DATASET_PICKLE_FILENAME, "w") as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)
    with open(FEATURE_LIST_FILENAME, "w") as featurelist_outfile:
        pickle.dump(feature_list, featurelist_outfile)


# In[78]:


dump_classifier_and_data(clf, my_dataset, features_list)


# In[79]:


def load_classifier_and_data():
    with open(CLF_PICKLE_FILENAME, "r") as clf_infile:
        clf = pickle.load(clf_infile)
    with open(DATASET_PICKLE_FILENAME, "r") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    with open(FEATURE_LIST_FILENAME, "r") as featurelist_infile:
        feature_list = pickle.load(featurelist_infile)
    return clf, dataset, feature_list


# In[80]:


def main():
    ### load up student's classifier, dataset, and feature_list
    clf, dataset, feature_list = load_classifier_and_data()
    ### Run testing script
    test_classifier(clf, dataset, feature_list)


# In[81]:


if __name__ == '__main__':
    main()

