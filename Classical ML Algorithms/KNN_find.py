## K-Nearest Neighbour Algorithm
#I am going to develop KNN algorithm in Python using: https://medium.datadriveninvestor.com/k-nearest-neighbors-in-python-hyperparameters-tuning-716734bc557f

from math import sqrt
import numpy as np
from random import randrange
from sklearn import preprocessing

from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

import pandas as pd

## Load Data
X_train = np.genfromtxt('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/X_train.csv', delimiter=',')
#X_train = X_train[0:735,:]
y_train = np.genfromtxt('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/y_train.csv', delimiter=',')
#y_train = y_train[0:735]
X_test = np.genfromtxt('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/X_test.csv', delimiter=',')
#X_test = X_test[0:295,:]
y_test = np.genfromtxt('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/y_test.csv', delimiter=',')
#y_test = y_test[0:295]

#select t-domain and mean,max,min features
features = np.array([1,2,3,10,11,12,13,14,15,41,42,43,50,51,52,53,54,55,81,82,83,90,91,92,93,94,95,121,122,123,130,131,132,133,134,135,161,162,163,170,171,172,173,174,175,201,204,205,214,217,218,227,230,231,240,243,244,253,256,257])
X_train = X_train[:,features-1]
X_test = X_test[:,features-1]

#scaling
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
scaler = preprocessing.StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)

#Transform to one-hot encoding
y_train = to_categorical(y_train-1, 6)
y_test = to_categorical(y_test-1, 6)

## Tuning Model
def plot_grid_search(df, grid_param_1, grid_param_2, name_param_1, name_param_2):
    df = df[df['param_p'] == 1]
  
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
leaf_size = list(range(2,30)) 
n_neighbors = list(range(15,30))
p=[1,2]
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

knn = KNeighborsClassifier()
clf = GridSearchCV(knn, hyperparameters, cv=10)

## Fit the model
best_model = clf.fit(X_train, y_train)

#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

plot_grid_search(pd.DataFrame(clf.cv_results_), leaf_size, n_neighbors, 'leaf_size', 'n_neighbors')

df = pd.DataFrame(clf.cv_results_)
df = df[(df['param_p']==1) & (df['param_leaf_size']<=10) & (df['param_n_neighbors']>=20)]
plot_grid_search(df, list(range(2,11)), list(range(20,30)), 'leaf_size', 'n_neighbors')

## Result
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_pred, multi_class="ovr")