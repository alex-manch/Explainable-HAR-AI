## Convolutional Neural Network + Long Short-Termed Memory Model

import numpy as np
from numpy import genfromtxt
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import confusion_matrix, classification_report

import lime
import lime.lime_tabular
import warnings
from lime import submodular_pick

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
	#filenames += ['body_gyro_x_'+group+'.csv', 'body_gyro_y_'+group+'.csv', 'body_gyro_z_'+group+'.csv']
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

activity_labels = open('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/activity_labels.txt', 'r').read().splitlines()

feature_names = list(['body_acc_x', 'body_acc_y', 'body_acc_z'])#, 'body_gyro_x', 'body_gyro_y', 'body_gyro_z'])

## Fit and Evaluate Model
n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
filters, kernel_size, conv_layers, lstm_layers, lstm_sizes, dropout, dense_layers, layer_size, learnRate = 256, 11, 3, 2, 200, 0.3, 1, 100, 1e-4

model = Sequential()
model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(n_timesteps,n_features)))
for i in range(0, conv_layers):
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
for i in range(1, lstm_layers):
    model.add(LSTM(units=lstm_sizes, return_sequences=True))
model.add(LSTM(units=lstm_sizes))
model.add(Dropout(dropout))
for i in range(0, dense_layers):
    model.add(Dense(layer_size, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))

model.summary()
plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

verbose, epochs, batch_size = 10, 20, 32
# fit network
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learnRate), metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.3, verbose=verbose)

filename = 'cnn_acc_model.sav'
pickle.dump(model, open(filename, 'wb'))

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

## Results
model.summary()
test_scores = model.evaluate(X_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

y_pred = model.predict(X_test)

fig, ax = plt.subplots()
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
ax.matshow(cm, cmap=plt.cm.Blues)
fig.set_size_inches(18.5, 10.5)

plt.title('CNN Confusion Matrix')
plt.xlabel('Predicted values')
plt.ylabel('True values')

for i in range(6):
    for j in range(6):
        c = cm[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
        
p = np.argmax(y_pred, axis=1)
t = np.argmax(y_test, axis=1)

print(classification_report(t, p, target_names=activity_labels).format(1.6))

y_pos = np.arange(len(activity_labels))
performance = [0.96, 0.95, 0.93, 0.88, 0.89, 0.89]

plt.figure(figsize=(16,8))
plt.bar(y_pos, performance, align='center')
plt.xticks(y_pos, activity_labels)
plt.ylabel('Usage')
plt.title('Programming language usage')

plt.show()

## Explaom the prediction using LIME
model = pickle.load(open(filename, 'rb'))

explainer = lime.lime_tabular.RecurrentTabularExplainer(X_train, feature_names=feature_names, class_names=activity_labels, mode='classification')

#Explain a random instance
i = np.random.randint(0, X_test.shape[0])
exp = explainer.explain_instance(X_test[i], model.predict, num_features=X_test.shape[-1], top_labels=1)
y_pred[i]
y_test[i]
exp.save_to_file('lime_CNN.html') #download the file and open it

## SubmodularPick-LIME
y_test_ = np.argmax(y_test, axis=1)+1
random_samples = np.empty((0, 128, 3), int)

for i in range(1,7):
    activity_samples = X_test[y_test_ == i]
    random_indices = np.random.choice(activity_samples.shape[0], 25, replace=False)
    random_samples = np.vstack([random_samples, activity_samples[random_indices]])

sp_obj = submodular_pick.SubmodularPick(explainer, random_samples, model.predict, sample_size=150, num_features=20, num_exps_desired=12)

[exp.as_pyplot_figure(label=exp.available_labels()[0]) for exp in sp_obj.sp_explanations]