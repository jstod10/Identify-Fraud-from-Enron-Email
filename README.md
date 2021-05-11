### Project Goal

The goal of this project was to examine data related to executives working at the infamous company Enron.  The dataset contains information on 146 individuals, with each record containing 21 features.  The purpose of the project was to utilize machine learning to determine which set of features could most effectively predict attributes a Person of Interest (POI) might display. 18 POIs have previously been identified

Initial examination of the dataset revealed two troublesome records:

* "The Travel Agency in the Park"
* "Total"

I removed those using the .pop() method, then created a for loop to iterate through the remaining entries to check for records with similar concerns and/or little to no data.  I ended up removing two more individual records.  The final dataset contained 142 entries. 

### Features Used

The feature list I ended up sing included the following:

('poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 
'salary', 'deferred_income','long_term_incentive',
'restricted_stock','total_payments',
'shared_receipt_with_poi', 'loan_advances')

#### Selection Process

I started by removing the 'email_address' and 'other' features, as the information contained in those was irrelevant and/or undefined.  I tried to create a new feature 'loan_advances_feature' that combined the 'loan_advances' and 'from_this_person_to_poi' features.  However, that feature proved no benefit so I ended up not using it in my final feature set. 

I used DecisionTreeClassifier first to evaluate different sets of features.  The feature list with the 2 features removed (as outlined above) proved to be the most accurate, although not quite to my liking.  I then utilized Several scaling options (Standard, MinMax, MaxAbs, PowerTranformer, and QuantileTransformer (Gaussian and Uniform)) along with Select K Best to isolate the most important features fromt the dataset.  I utilized the results from that query to use the top five features using a feature score of 10.0+ as criteria.  

Next, I used DecisionTreeClassifier to test feature sets using the top 4, 5, and 6 top features, along with my newly created feature to see which perfomed the best.  I wasn't happy with the results right away, so I ended up settling on using the top 10 features in my final query. 

### Algorithms

I applied the following algorithms during testing:

* SVC
* DecisionTreeClassifier
* RandomForest
* AdaBoost
* GradientBoosting

Several test runs revealed DecisionTree as the top performer of the group, with GradientBoosting and SVC coming in next.  DecisionTree consistently returned accuracy of 90%+ on training sets and 85%+ on test sets.  Precision and Recall both consisently came in at roughly 0.429 and 0.5, respectively.

AdaBoost did not show up in my top 5 performers during all testing.

### Tuning

The definition of tuning, according to [https://en.wikipedia.org/wiki/Hyperparameter_optimization#:~:text=In%20machine%20learning%2C%20hyperparameter%20optimization,typically%20node%20weights)%20are%20learned.]:

In machine learning, hyperparameter optimization or tuning is the problem of choosing a set of optimal hyperparameters for a learning algorithm. A hyperparameter is a parameter whose value is used to control the learning process. By contrast, the values of other parameters (typically node weights) are learned.

The same kind of machine learning model can require different constraints, weights or learning rates to generalize different data patterns. These measures are called hyperparameters, and have to be tuned so that the model can optimally solve the machine learning problem. Hyperparameter optimization finds a tuple of hyperparameters that yields an optimal model which minimizes a predefined loss function on given independent data. The objective function
takes a tuple of hyperparameters and returns the associated loss.  Cross-validation is often used to estimate this generalization performance.


When tuning is not done well, it leads to ineffecient, inconsistent, and irrelevant results.  To validate my findings, I created a for loop to test different combination of classifiers with recommended parameters for each, then looped through scale features to look for the top 5 performing parameters. 

### Validation

A validation dataset is a sample of data held back from training your model that is used to give an estimate of model skill while tuning model’s hyperparameters.

When validation is done wrong, it can lead to innacurate findings.  It's important to find desirable symmetry between training and test sets. 

To validate my findings, I created a for loop to test different combinations of classifiers and scaling to isolate the best performers.  I then ran several test runs and took note of the top 5 results and looked for consistency. 

### Testing

Testing revealed accuracy in the range of 85%. 

Precision and Recall came in at 0.429 and 0.5, respectively.  
