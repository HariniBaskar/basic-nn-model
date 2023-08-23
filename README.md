# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network with multiple hidden layers and multiple nodes in each hidden layer is known as a deep learning system or a deep neural network. Here the basic neural network model has been created with one input layer, one hidden layer and one output layer.The number of neurons(UNITS) in each layer varies the 1st input layer has 16 units and hidden layer has 8 units and output layer has one unit.

In this basic NN Model, we have used "relu" activation function in input and hidden layer, relu(RECTIFIED LINEAR UNIT) Activation function is a piece-wise linear function that will output the input directly if it is positive and zero if it is negative.

## Neural Network Model

![exp1](https://github.com/HariniBaskar/basic-nn-model/assets/93427253/60aed9ab-aa09-4320-b0cd-bb7fb628cce9)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('Experiment1').sheet1
data = worksheet.get_all_values()
df = pd.DataFrame(data[1:], columns=data[0])
df = df.astype({'Input1':'float'})
df = df.astype({'Output':'float'})
df.head()

X = df[['Input1']].values
y = df[['Output']].values
X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
X_train1

aimodel=Sequential([
    Dense(7,activation='relu'),
    Dense(6,activation='relu'),
    Dense(1)
])
aimodel.compile(optimizer='rmsprop',loss='mse')
aimodel.fit(X_train1,y_train,epochs=2000)
aimodel.fit(X_train1,y_train,epochs=2000)

loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()

X_test1 = Scaler.transform(X_test)
ai.evaluate(X_test1,y_test)

X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai.predict(X_n1_1)
```

## Dataset Information

![exp11](https://github.com/HariniBaskar/basic-nn-model/assets/93427253/199521ca-7fcf-430f-8639-4c6363ab82ab)

## OUTPUT

### Training Loss Vs Iteration Plot

![exp12](https://github.com/HariniBaskar/basic-nn-model/assets/93427253/b2420eb5-05c0-462f-baad-1ba18a7f0be4)


### Test Data Root Mean Squared Error

![exp13](https://github.com/HariniBaskar/basic-nn-model/assets/93427253/4718b208-7a3d-4764-beb2-7380ded6688e)

### New Sample Data Prediction

![exp14](https://github.com/HariniBaskar/basic-nn-model/assets/93427253/ceae7531-4350-4e00-88eb-05c848fb812d)

## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully.
