#!/usr/bin/python

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

import warnings
warnings.filterwarnings('ignore')

import os

os.getcwd()

os.chdir("C:\\Users\\josep\\ud120-projects\\final_project")

os.listdir()

original = "final_project_dataset.pkl"
destination = "final_project_dataset.pkl_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))

with open("final_project_dataset.pkl_unix.pkl", "rb") as data_file:
    data_dict = pkl.load(data_file)

# make a pandas dataframe to explore

enron_df = pd.DataFrame.from_dict(data_dict,orient='index')


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

data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

# List of people with no total payment data

for entry in data_dict:
    if data_dict[entry]['total_payments'] == 'NaN':
        print(entry)
        
# List of people with no total payment data and no stock option data

for entry in data_dict:
    if data_dict[entry]['total_payments'] == 'NaN' and data_dict[entry]['total_stock_value'] == 'NaN':
        print(entry)
        

data_dict.pop('LOCKHART EUGENE E', 0)
data_dict.pop('CHAN RONNIE', 0)

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

# Saving the new dataset without the outliers

my_dataset = data_dict

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
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
 'loan_advances_feature'
 'long_term_incentive',
 'restricted_stock',
 'restricted_stock_deferred',
 'salary',
 'shared_receipt_with_poi',
 'to_messages',
 'total_payments',
 'total_stock_value'
]

distributions = [
    ('Data after standard scaling',
        StandardScaler().fit_transform(features)),
    ('Data after min-max scaling',
        MinMaxScaler().fit_transform(features)),
    ('Data after max-abs scaling',
        MaxAbsScaler().fit_transform(features)),
    ('Data after power transformation (Yeo-Johnson)',
     PowerTransformer(method='yeo-johnson').fit_transform(features)),
    ('Data after quantile transformation (gaussian pdf)',
        QuantileTransformer(output_distribution='normal')
        .fit_transform(features)),
    ('Data after quantile transformation (uniform pdf)',
        QuantileTransformer(output_distribution='uniform')
        .fit_transform(features))
]

# list of classifiers

classifiers = [
    ('SVC', SVC()),
    ('DTC', DecisionTreeClassifier()),
    ('RFC', RandomForestClassifier()),
    ('ADA', AdaBoostClassifier()),
    ('GBC', GradientBoostingClassifier())
]

# param grid for different classifiers
param_grid_list = [('SVC', {'C': [1,10,50,100],
                            'kernel': ['linear', 'poly','rbf']}),
                   ('DTC',{'criterion': ['gini','entropy'], 
                           'splitter': ['best','random'], 
                           'min_samples_split' : [2,5,8, 10],
                           'max_depth': [10,25,50,100,200]}),
                   ('RFC', {'n_estimators': [10,50,100], 
                            'criterion': ['gini','entropy'],
                            'max_depth': [3,5,8],
                            'max_features' : ['log2','sqrt']}),
                   ('ADA', {'n_estimators': [10,50,100], 
                            'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                            'algorithm' : ['SAMME', 'SAMME.R']}),
                   ('GBC', {'loss': ['deviance'],
                            'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                            'max_depth': [3,5,8],
                            'max_features' : ['log2','sqrt'],
                            'n_estimators': [10,50,100]})
                  ]

# loop scale features

result_list = []
for x, features in enumerate(distributions):
    # get scaled features from distribution
    scaled_features = distributions[x][1]
    #divide train and test set
    features_train, features_test, labels_train, labels_test = train_test_split(
        scaled_features, labels, test_size=0.33, random_state=42)
    print('###########################################################')
    print('###################### feature ############################')
    print(features[0])
    # loop classifiers
    for y, classifier in classifiers:
        print('################### classifier ############################')
        print(y, classifier)
        print('###########################################################')
        #find matching parameter grid
        for z, items in param_grid_list:
            if y == z:
                print(y, z, items)
                # classifier using GridSearchCV optimizing for f1 score
                clf = GridSearchCV(classifier, items, scoring='f1_weighted', n_jobs=-2)
                clf.fit(features_train, labels_train)
                print('###########################################################')
                print('parameters:' + str(clf.best_params_))
                print('###########################################################')
                print('accuracy training set: {0:.3g}'.format(clf.score(features_train, labels_train)))             
                print('accuracy test set: {0:.3g}'.format(clf.score(features_test, labels_test)))                    
                print('precision_score : {0:.3g}'.format(precision_score(labels_test,clf.predict(features_test))))
                print('recall_score : {0:.3g}'.format(recall_score(labels_test,clf.predict(features_test))))          
                print('f1-score : {0:.3g}'.format(f1_score(labels_test,clf.predict(features_test))))                
                result_list.append({'scaling' : features[0],
                                    'classifier' : classifier.__class__.__name__,
                                    'best_params' : clf.best_params_,
                                    'accuracy_training_set': clf.score(features_train, labels_train),            
                                    'accuracy_test_set': clf.score(features_test, labels_test),                 
                                    'precision_score': precision_score(labels_test,clf.predict(features_test)),
                                    'recall_score': recall_score(labels_test,clf.predict(features_test)),       
                                    'f1_score' : f1_score(labels_test,clf.predict(features_test))})
                # save intermediate results
                result_df = pd.DataFrame(result_list)
                result_df.to_pickle('result_df.pkl')
                
#list to df and pickle for later use
result_df = pd.DataFrame(result_list)
result_df.to_pickle('result_df.pkl')


import pickle
import sys
from sklearn.model_selection import StratifiedShuffleSplit
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

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
                print("Warning: Found a predicted label not == 0 or 1.")
                print("All predictions should take value 0 or 1.")
                print("Evaluating performance for processed predictions:")
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print(clf)
        print(PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
        print(RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
        print("")
    except:
        print("Got a divide by zero when trying out:", clf)
        print("Precision or recall may be undefined due to a lack of true positive predicitons.")
        
        

def dump_classifier_and_data(clf, dataset, feature_list):
    with open(CLF_PICKLE_FILENAME, "wb") as clf_outfile:
        pickle.dump(clf, clf_outfile)
    with open(DATASET_PICKLE_FILENAME, "wb") as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)
    with open(FEATURE_LIST_FILENAME, "wb") as featurelist_outfile:
        pickle.dump(feature_list, featurelist_outfile)
        
CLF_PICKLE_FILENAME = "my_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"

dump_classifier_and_data(clf, my_dataset, features_list)
