## Best Decision-Tree Model Finder

import numpy as np
from numpy import genfromtxt

import matplotlib.pyplot as plt

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load Data
X_train = genfromtxt('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/X_train.csv', delimiter=',')
#X_train = X_train[0:735,:]
y_train = genfromtxt('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/y_train.csv', delimiter=',')
#y_train = y_train[0:735]
X_test = genfromtxt('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/X_test.csv', delimiter=',')
#X_test = X_test[0:735,:]
y_test = genfromtxt('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/y_test.csv', delimiter=',')
#y_test = y_test[0:735]

#select t-domain and mean,max,min features
features = np.array([1,2,3,10,11,12,13,14,15,41,42,43,50,51,52,53,54,55,81,82,83,90,91,92,93,94,95,121,122,123,130,131,132,133,134,135,161,162,163,170,171,172,173,174,175,201,204,205,214,217,218,227,230,231,240,243,244,253,256,257])
X_train = X_train[:,features-1]
X_test = X_test[:,features-1]

# Tuning loop
def plot_grid_search(df, grid_param_1, grid_param_2, name_param_1, name_param_2):
    scores_mean = df['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = df['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(22, 22)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_1 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_2, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')

#List Hyperparameters that we want to tune.
max_depth = list(range(1,20))
min_samples_leaf = list(range(1,20))
parameters = dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

dt_model = DecisionTreeClassifier(criterion="gini", random_state=42)
clf = GridSearchCV(dt_model, parameters, cv=10, return_train_score=False)#, verbose=10)

#Fit the model
best_model = clf.fit(X_train, y_train)

plot_grid_search(pd.DataFrame(clf.cv_results_), max_depth, min_samples_leaf, 'max_depth', 'min_samples_leaf')

# Print The value of best Hyperparameters
print('Best max_depth:', best_model.best_estimator_.get_params()['max_depth'])
print('Best min_samples_leaf:', best_model.best_estimator_.get_params()['min_samples_leaf'])
print('Best score:', best_model.best_estimator_.score(X_test, y_test))