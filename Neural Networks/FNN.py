## Feedforward Neural Network
from numpy import genfromtxt
import numpy as np
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

X_train = genfromtxt('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/X_train.csv', delimiter=',')
#X_train = X_train[0:735,:]
y_train = genfromtxt('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/y_train.csv', delimiter=',')
#y_test = y_test[0:295]
X_test = genfromtxt('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/X_test.csv', delimiter=',')
#X_test = X_test[0:295,:]
y_test = genfromtxt('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/y_test.csv', delimiter=',')
#y_test = y_test[0:295]

#scaling
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
scaler = preprocessing.StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)

#Transform to one-hot encoding
y_train = to_categorical(y_train-1, 6)
y_test = to_categorical(y_test-1, 6)

## Build Neural Network
model= Sequential([
                   Dense(257, input_shape=(561,), activation='relu'),
                   Dropout(0.5), #Dropout rate between the fully connected layers (useful to reduce overfitting)
                   Dense(141, activation='relu'),
                   Dropout(0.5),
                   Dense(6, activation='softmax')  #this makes the sum of class probabilities = 1
])

model.summary()
plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True) #plots the model as a graph

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(learning_rate=0.0001),
    metrics=["accuracy"]
    )
    
## Train Neural Network
epochs = 20
batch_size = 8
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='testing accuracy')
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

## Test NN
model.summary()
test_scores = model.evaluate(X_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])   

## Explain the predictions using LIME
feature_names = open('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/features.txt', 'r').read().splitlines()

activity_labels = open('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/activity_labels.txt', 'r').read().splitlines()

explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=activity_labels, mode='classification')

#explaining instance
i = np.random.randint(0, X_test.shape[0])
exp = explainer.explain_instance(X_test[i], model.predict, num_features=561, top_labels=1)
model.predict(X_test)[i]
y_test[i]
exp.save_to_file('lime_FNN.html') #download the file and open it