## Convolutional Neural Network + Long Short-Termed Memory Model Loop

import numpy as np
from numpy import genfromtxt
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense, LSTM, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.vis_utils import plot_model
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

def get_model(filters=256, kernel_size=11, conv_layers=3, lstm_layers=1, lstm_sizes=50, dropout=0.2, dense_layers=1, layer_size=100, learnRate=0.01):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(n_timesteps,n_features))) #1D o 2D
    for i in range(0, conv_layers):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
    for i in range(1, lstm_layers):
        model.add(LSTM(units=lstm_sizes, return_sequences=True))
    model.add(LSTM(units=lstm_sizes))
    model.add(Dropout(dropout))
    for i in range(0, dense_layers):
        model.add(Dense(layer_size, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learnRate), metrics=['accuracy'])
    return model

print("[INFO] initializing model...")

# define a grid of the hyperparameter search space
filter = 256#[128, 192, 256] filterS
kernel_size = 11#[5, 7, 11] sizeS
conv_layer = 3#range(1,5)
lstm_layers = range(1,4)
lstm_sizes = range(50, 250, 50)
dropouts = [0.2, 0.3, 0.4, 0.5]
dense_layer = 1#range(1,5)
layer_size = 50#range(50, 250, 50)
learnRate = 0.0001 #learnRates = [1e-2, 1e-3, 1e-4]
batch_size = 30 #batch_sizes = [4, 8, 16, 32]
epoch = 20 #epochs = range(15, 25, 5)

acc, mx, mxf, mxk, mxc, mxl, mxls, mxd, mxdl, mxl, mxlR, mxe, mxb = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
mxlR, mxb, mxe = learnRate, batch_size, epoch
verbose = 0

print("[INFO] performing loop...")
while (acc < 0.95):
    #for dense_layer in dense_layers:
        #for layer_size in layer_sizes:
  for dropout in dropouts:
    for lstm_layer in lstm_layers:
      for lstm_size in lstm_sizes:
                        #for conv_layer in conv_layers:
                        #for filter in filters:
                                #for kernel_size in kernel_sizes:
                                    #for learnRate in learnRates:
        model = get_model(filter, kernel_size, conv_layer, lstm_layer, lstm_size, dropout, dense_layer, layer_size, learnRate)
                                        #for epoch in epochs:
                                            #for batch_size in batch_sizes:
        model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_split=0.3, verbose=verbose)
        print('[DATA] f:'+str(filter)+', k:'+str(kernel_size)+', c:'+str(conv_layer)+', ll:'+str(lstm_layer)+', ls:'+str(lstm_size)+', d:'+str(dropout)+', dl:'+str(dense_layer)+', l:'+str(layer_size)+', lR:'+str(learnRate)+', e:'+str(epoch)+', b:'+str(batch_size))
        test_scores = model.evaluate(X_test, y_test, verbose=2)
        acc = test_scores[1]
        if acc > mx: #in case i don't find a model that reaches 95% accuracy on the test data, I still want to save the model that reaches the best value
            mx = acc
            #mxf = filter
            #mxk = kernel_size
            #mxc = conv_layer
            mxll = lstm_layer
            mxls = lstm_size
            mxd = dropout
            #mxdl = dense_layer
            #mxl = layer_size
            #mxlR = learnRate
            #mxe = epoch
            #mxb = batch_size
            print('[INFO] NEW MAX FOUND!')
            with open('readme.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
                f.write("Test loss:" + str(test_scores[0]))
                f.write(" Test accuracy:" + str(test_scores[1]) + '\n')
                f.write("using f:"+str(mxf)+', k:'+str(mxk)+', c:'+str(mxc)+', ll:'+str(mxll)+', ls:'+str(mxls)+', d:'+str(mxd)+', dl:'+str(mxdl)+', l:'+str(mxl)+', lR:'+str(mxlR)+', e:'+str(mxe)+', b:'+str(mxb))
model.summary()
test_scores = model.evaluate(X_test, y_test, verbose=2)

print(mx)
print("[INFO] best accuracy is {:.2f}".format(test_scores[1]))
print("using f:"+str(mxf)+', k:'+str(mxk)+', c:'+str(mxc)+', ll:'+str(mxll)+', ls'+str(mxls)+', d:'+str(mxd)+', dl:'+str(mxdl)+', l:'+str(mxl)+', lR:'+str(mxlR)+', e:'+str(mxe)+', b:'+str(mxb))
