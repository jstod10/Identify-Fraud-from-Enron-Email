Project Goal
The goal of this project was to examine data related to executives working at the infamous company Enron. The dataset contains information on 146 individuals, with each record containing 21 features. The purpose of the project was to utilize machine learning to determine which set of features could most effectively predict attributes a Person of Interest (POI) might display. 18 POIs have previously been identified

Initial examination of the dataset revealed two troublesome records:

"The Travel Agency in the Park"
"Total"
I removed those using the .pop() method, then created a for loop to iterate through the remaining entries to check for records with similar concerns and/or little to no data. I ended up removing two more individual records. The final dataset contained 142 entries.

Features Used
The feature list I ended up using included the following:

('poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'percent_poi_emails')

The following is a list of features and the number of entries missing values for each:

FEATURE COUNT
deferral_payments 104
loan_advances 139
restricted_stock_deferred 126
deferred_income 95
exercised_stock_options 41
long_term_incentive 77
director_fees 127
to_messages 56
email_address 31
from_poi_to_this_person 56
from_messages 56
from_this_person_to_poi 56
shared_receipt_with_poi 56
salary 48
total_payments 19
bonus 61
expenses 48
other 51
restricted_stock 34
total_stock_value 17

Selection Process
I started by removing the 'email_address' and 'other' features, as the information contained in those was irrelevant and/or undefined. I tried to create a new feature 'percent_poi_emails' that combined the 'from_poi_to_this_person' and 'from_this_person_to_poi' features. The purpose of this feature was to identify the percentage of emails connected to POIs.

I used DecisionTreeClassifier first to evaluate different sets of features. The feature list with the 2 features removed (as outlined above) proved to be the most accurate, although not quite to my liking. I then utilized Select K Best to isolate the most important features fromt the dataset. I utilized the results from that query to use the top five features using a minimum feature score of 10.0 as criteria.

Next, I used DecisionTreeClassifier to test feature sets using the top 4, 5, and 6 top features, along with my newly created feature to see which perfomed the best. Using the top 4 features, along with my newly created feature, proved to be the best solution.

Algorithms
I applied the following classifiers during testing and tuned the parameters for each manually:

SVC
DecisionTreeClassifier
NaiveBayes - Gaussian
AdaBoost
LogisticRegression
Several test runs revealed DecisionTree as the top performer of the group

Performance tests using DecisionTree and my selected feature set were as follows:

Accuracy: 0.81
Precision: 0.40
Recall: 0.50
Tuning
The definition of tuning, according to [https://en.wikipedia.org/wiki/Hyperparameter_optimization#:~:text=In%20machine%20learning%2C%20hyperparameter%20optimization,typically%20node%20weights)%20are%20learned.]:

In machine learning, hyperparameter optimization or tuning is the problem of choosing a set of optimal hyperparameters for a learning algorithm. A hyperparameter is a parameter whose value is used to control the learning process. By contrast, the values of other parameters (typically node weights) are learned.

The same kind of machine learning model can require different constraints, weights or learning rates to generalize different data patterns. These measures are called hyperparameters, and have to be tuned so that the model can optimally solve the machine learning problem. Hyperparameter optimization finds a tuple of hyperparameters that yields an optimal model which minimizes a predefined loss function on given independent data. The objective function takes a tuple of hyperparameters and returns the associated loss. Cross-validation is often used to estimate this generalization performance.

When tuning is not done well, it leads to ineffecient, inconsistent, and irrelevant results.

To tune the parameters, I created a for loop, incorporating GridSearchCV for parameter tuning to test different combination of classifiers with recommended parameters for each.

Validation
A validation dataset is a sample of data held back from training your model that is used to give an estimate of model skill while tuning model’s hyperparameters. Validation also serves as a check on overfitting.

When validation is done wrong, it can lead to innacurate findings. A classic mistake in validation is when we use the same data for both training and testing. It is important to train on a valid sample and then test on a different set of data to test the efficacy of chosen parameters. It's important to find desirable symmetry between training and test sets, especially in regards to accuracy - the closer the two results, the more effective the chosen method.

To validate my findings initially, I created a for loop to test different combinations of classifiers and scaling to isolate the best performers using cross validation. I then ran several test runs and took note of the top 5 results and looked for consistency.

Final validation was performed by splitting the data into training and testing sets and comparing the results of each. This code was detailed in the tester.py script.

Testing
3 metrics were used to evaluate different settings:

Accuracy
Precision
Recall
Accuracy is the liklihood of identifying a POI using the feature selection criteria I outlined above.

Precision is the proportion of poisitive identifications that were actually correct.

Recall is the proportion of actual positives that were identified correctly.

Testing using the testing.py script revealed:

Accuracy of 0.78

Precision and Recall came in at 0.30560 and 0.31400, respectively.

These results are encouraging, as there is a nice balance between precision and recall, which proves the effectiveness of the model.
