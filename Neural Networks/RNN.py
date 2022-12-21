## Recurrent Neural Network

import numpy as np
from numpy import genfromtxt
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

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

#zero-offset class values
y_train = y_train - 1
y_test = y_test - 1
#one hot encode y
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

## Build and Train Model
model = keras.Sequential()
model.add(layers.Bidirectional(keras.layers.LSTM(units=128, input_shape=[X_train.shape[1], X_train.shape[2]])))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(units=128, activation='relu'))
model.add(layers.Dense(y_train.shape[1], activation='softmax'))

epochs = 30
batch_size = 16

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(learning_rate=0.0001),
    metrics=["accuracy"])
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

#Visualizing training process
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='test accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='testing loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

## Validation
model.summary()
plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True) #plots the model as a graph

test_scores = model.evaluate(X_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

y_pred = model.predict(X_test)

fig, ax = plt.subplots()
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
ax.matshow(cm, cmap=plt.cm.Blues)
fig.set_size_inches(18.5, 10.5)

plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted values')
plt.ylabel('True values')

for i in range(6):
    for j in range(6):
        c = cm[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
    
## Explain the predictions using LIME
feature_names = open('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/features.txt', 'r').read().splitlines()
activity_labels = open('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/activity_labels.txt', 'r').read().splitlines()
explainer = lime.lime_tabular.RecurrentTabularExplainer(X_test, feature_names=feature_names, class_names=activity_labels, mode='classification')

#explain a random instance
i = np.random.randint(0, X_test.shape[0])
exp = explainer.explain_instance(X_test[i], model.predict, num_features=561, top_labels=1)

model.predict(X_test)[i]
y_test[i]
exp.save_to_file('lime_RNN.html') #download the file and open it