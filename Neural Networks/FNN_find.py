## Best Feedforward Neural Network Finder

from numpy import genfromtxt
from sklearn import preprocessing
from numpy import genfromtxt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

X_train = genfromtxt('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/X_train.csv', delimiter=',')
#X_train = X_train[0:735,:]
y_train = genfromtxt('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/y_train.csv', delimiter=',')
#y_train = y_train[0:735]
X_test = genfromtxt('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/X_test.csv', delimiter=',')
#X_test = X_test[0:735,:]
y_test = genfromtxt('drive/MyDrive/Colab Notebooks/UCI HAR Dataset/y_test.csv', delimiter=',')
#y_test = y_test[0:735]

#scaling
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
scaler = preprocessing.StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)
#Transform to one-hot encoding
y_train = to_categorical(y_train-1, 6)
y_test = to_categorical(y_test-1, 6)

## Bulding and Training Model
def get_mlp_model(hiddenLayerOne=561, hiddenLayerTwo=256,	dropout=0.2, learnRate=0.01):
    model = Sequential()
    model.add(Dense(hiddenLayerOne, activation="relu", input_shape=(561,)))
    model.add(Dropout(dropout)) #Dropout rate between the fully connected layers (useful to reduce overfitting)
    model.add(Dense(hiddenLayerTwo, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(6, activation="softmax"))
    model.compile(optimizer=Adam(learning_rate=learnRate), loss="categorical_crossentropy", metrics=["accuracy"])
    return model
    
#Hyperparameter tuning
# wrap our model into a scikit-learn compatible classifier
print("[INFO] initializing model...")
model = KerasClassifier(build_fn=get_mlp_model, verbose=0)

# define a grid of the hyperparameter search space
hiddenLayerOne = range(200, 561, 19)
hiddenLayerTwo = range(6, 561, 15)
learnRate = [1e-2, 1e-3, 1e-4]
dropout = [0.2, 0.3, 0.4, 0.5]
batchSize = [4, 8, 16, 32]
epochs = range(5, 50, 5)
# create a dictionary from the hyperparameter grid
grid = dict(
	hiddenLayerOne=hiddenLayerOne,
	learnRate=learnRate,
	hiddenLayerTwo=hiddenLayerTwo,
	dropout=dropout,
	batch_size=batchSize,
	epochs=epochs
)

# initialize a random search with a 3-fold cross-validation and then start the hyperparameter search process
print("[INFO] performing random search...")
searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=3,
	param_distributions=grid, scoring="accuracy")
searchResults = searcher.fit(X_train, y_train)
# summarize grid search information
bestScore = searchResults.best_score_
bestParams = searchResults.best_params_
print("[INFO] best score is {:.2f} using {}".format(bestScore,
	bestParams))
    
## Test the model
# extract the best model, make predictions on our data, and show a classification report
print("[INFO] evaluating the best model...")
bestModel = searchResults.best_estimator_
accuracy = bestModel.score(X_test, y_test)
print("accuracy: {:.2f}%".format(accuracy * 100))