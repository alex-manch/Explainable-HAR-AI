## One dimensional Convolutional Neural Network Finder

import numpy as np
from numpy import genfromtxt
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

## Load Data
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

#transform to one-hot encoding
y_train = to_categorical(y_train-1)
y_test = to_categorical(y_test-1)

## Build the model
n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

def get_model(filters1=64, conv_layers=1, filters2=64, kernel_size1=3, kernel_size2=3, dense_layer=1, layer_size=100, dropout=0.2, learnRate=0.01):
    model = Sequential()
    model.add(Conv1D(filters=filters1, kernel_size=kernel_size1, activation='relu', input_shape=(n_timesteps,n_features)))
    for i in range(0, conv_layers):
        model.add(Conv1D(filters=filters2, kernel_size=kernel_size2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    for i in range(0, dense_layer):
        model.add(Dense(layer_size, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learnRate), metrics=['accuracy'])
  
## Hyper-parameter tuning
print("[INFO] initializing model...")
model = KerasClassifier(build_fn=get_model, verbose=0)

# define a grid of the hyperparameter search space
filters1 = [8, 16, 32, 64, 128, 256]
conv_layers = range(1,5)
filters2 = [8, 16, 32, 64, 128, 256]
kernel_size1 = [2, 3, 5, 7, 11]
kernel_size2 = [2, 3, 5, 7, 11]
dense_layer = range(1,5)
layer_size = range(50, 900, 34)
dropout = [0.2, 0.3, 0.4, 0.5]
learnRate = [1e-2, 1e-3, 1e-4]
batchSize = [4, 8, 16, 32]
epochs = range(5, 25, 2)
# create a dictionary from the hyperparameter grid
grid = dict(
    filters1=filters1,
    conv_layers=conv_layers,
    filters2=filters2,
    kernel_size1=kernel_size1,
    kernel_size2=kernel_size2,
    dense_layer=dense_layer,
    layer_size=layer_size,
    dropout=dropout,
	learnRate=learnRate,
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

## Test the model
print("[INFO] evaluating the best model...")
bestModel = searchResults.best_estimator_
accuracy = bestModel.score(X_test, y_test)
print("accuracy: {:.2f}%".format(accuracy * 100))