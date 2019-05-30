# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def labelEncode(end, arrayOfIndices, targetArray):
    for i in range(0, end):
        print(i)
        labelencoder = LabelEncoder()
        targetArray[:, arrayOfIndices[i]] = labelencoder.fit_transform(targetArray[:, arrayOfIndices[i]])


          
        
# Importing the dataset
dataset = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
X = dataset.iloc[:, 2:80].values
y = dataset.iloc[:, 80].values
XDataframe = pd.DataFrame(X)
stringIndices = []
for j in range(0, len(X[0])):
    for i in range(0, 1460):
        if(isinstance(X[i][j], str)):
            stringIndices.append(j)

stringIndices = list(dict.fromkeys(stringIndices))
        
for i in range(0, 1460):
    for j in stringIndices:
        if (isinstance(X[i][j], str)):
            continue
        X[i][j] = 'Unknown'
labelEncode(len(stringIndices), stringIndices, X)        


for j in range(0, len(X[0])):
    for i in range(0, 1460):
        if(np.isnan(X[i][j])):
            X[i][j] = 0

XDataframe = pd.DataFrame(X)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


# Until here data is cleaned from strings and nan has been set to either
# Unknonwn or 0


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #     
onehotencoder = OneHotEncoder(categorical_features = stringIndices)
X = onehotencoder.fit_transform(X).toarray()

XDataframe = pd.DataFrame(X)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)