## Convolutional Neural Network + Long Short-Termed Memory Model Finder

import numpy as np
from numpy import genfromtxt
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import RandomizedSearchCV
from matplotlib import pyplot

def load_file(filepath):
    array = genfromtxt(filepath, delimiter=',')
    return array

# load a list of files into a 3D array of [samples, timesteps, features]
def load_group(filenames, prefix='drive/MyDrive/Colab Notebooks/UCI HAR Dataset/'):
    loaded = load_file(prefix + filenames[0])
    loaded = loaded.reshape(-1, 128, 1)
    for name in filenames[1:]:
        data = load_file(prefix + name)
        data = data.reshape(-1, 128, 1)
        loaded = np.concatenate((loaded, data), axis=-1) 
    return loaded

def load_dataset_group(group, prefix='drive/MyDrive/Colab Notebooks/UCI HAR Dataset/'):
    filenames = list()
    #filenames += ['total_acc_x_'+group+'.csv', 'total_acc_y_'+group+'.csv', 'total_acc_z_'+group+'.csv']
    filenames += ['body_acc_x_'+group+'.csv', 'body_acc_y_'+group+'.csv', 'body_acc_z_'+group+'.csv']
    filenames += ['body_gyro_x_'+group+'.csv', 'body_gyro_y_'+group+'.csv', 'body_gyro_z_'+group+'.csv']
    # load input data
    X = load_group(filenames)
    # load class output
    y = load_file(prefix+'y_'+group+'.csv')
    return X, y

def scale_data(X_train, X_test):
    cut = int(X_train.shape[1] / 2)
    X_long = X_train[:, -cut:, :]
    X_long = X_long.reshape((X_long.shape[0] * X_long.shape[1], X_long.shape[2]))
    flat_X_train = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
    flat_X_test = X_test.reshape((X_test.shape[0] * X_test.shape[1], X_test.shape[2]))
    scaler = preprocessing.StandardScaler().fit(X_long)
    flat_X_train = scaler.transform(flat_X_train)
    flat_X_test = scaler.transform(flat_X_test)
    X_train = flat_X_train.reshape((X_train.shape))
    X_test = flat_X_test.reshape((X_test.shape))
    return X_train, X_test

#load the dataset
X_train, y_train = load_dataset_group('train')
X_test, y_test = load_dataset_group('test')
#X_train, X_test, y_train, y_test= X_train[0:735,:,:], X_test[0:295,:,:], y_train[0:295], y_test[0:295]

#standardize data
X_train, X_test = scale_data(X_train, X_test)

#zero-offset class values
y_train = y_train - 1
y_test = y_test - 1
#one hot encode y
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

def get_model(dense_layers=1, layer_size=100):
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=11, activation='relu', input_shape=(n_timesteps,n_features))) #1D o 2D
    for i in range(0, 3):
        model.add(Conv1D(filters=256, kernel_size=11, activation='relu'))
    for i in range(1, 2):
        model.add(LSTM(units=250, return_sequences=True))
    model.add(LSTM(units=250))
    model.add(Dropout(0.2))
    for i in range(0, dense_layers):
        model.add(Dense(layer_size, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model

print("[INFO] initializing model...")
model = KerasClassifier(build_fn=get_model, verbose=0)

# define a grid of the hyperparameter search space
#filters = [192, 256]
#kernel_size = [5, 7, 11]
#conv_layers = range(1,5)
#lstm_layers = range(1,5)
#lstm_sizes = range(50, 250, 50)
#dropout = [0.2, 0.3, 0.4, 0.5]
dense_layers = range(1,5)
layer_size = range(50, 250, 50)
#learnRate = 0.0001#[1e-3, 1e-4]
batchSize = [24, 32]
epochs = [20, 25]#range(15, 25, 5)
# create a dictionary from the hyperparameter grid
grid = dict(
    #filters=filters,
    #kernel_size=kernel_size,
    #conv_layers=conv_layers,
    #lstm_layers=lstm_layers,
    #lstm_sizes=lstm_sizes,
    #dropout=dropout,
    dense_layers=dense_layers,
    layer_size=layer_size,
    #learnRate=learnRate,
    batch_size=batchSize,
    epochs=epochs
)

print("[INFO] performing random search...")
searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=3,	param_distributions=grid, scoring="accuracy")
searchResults = searcher.fit(X_train, y_train)
# summarize grid search information
bestScore = searchResults.best_score_
bestParams = searchResults.best_params_
print("[INFO] best score is {:.2f} using {}".format(bestScore, bestParams))

print("[INFO] evaluating the best model...")
bestModel = searchResults.best_estimator_
accuracy = bestModel.score(X_test, y_test)
print("accuracy: {:.2f}%".format(accuracy * 100))