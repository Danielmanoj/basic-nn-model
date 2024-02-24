# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by human brain neurons). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps establish a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the libraries that we will use Import the dataset and check the types of the columns Now build your training and test set from the dataset Here we are making the neural network 2 hidden layers with 1 output and input layer and an activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model


![image](https://github.com/Danielmanoj/basic-nn-model/assets/69635071/97bd7920-1b2b-435e-88fa-ed2651916dce)


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
### Name: MANOJ G
### Register Number: 212222240060
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
worksheet = gc.open('clinical').sheet1
data = worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'actual':'float'})
dataset1 = dataset1.astype({'predicted':'float'})
dataset1.head()
X = dataset1[['actual']].values
y = dataset1[['predicted']].values
X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai_brain = Sequential([
    Dense(6,activation = 'relu'),
    Dense(6,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(X_train1,y_train,epochs = 4000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)



```
## Dataset Information


![image](https://github.com/Danielmanoj/basic-nn-model/assets/69635071/4d8f6638-a937-46af-a6cb-e14421f5354c)


## OUTPUT

### Training Loss Vs Iteration Plot


![image](https://github.com/Danielmanoj/basic-nn-model/assets/69635071/b34c4320-15b1-4045-8272-8cc6b1b0a217)


### Test Data Root Mean Squared Error


![image](https://github.com/Danielmanoj/basic-nn-model/assets/69635071/d7c21acf-d567-4ee7-913c-aa66033c8c6b)


### New Sample Data Prediction

![image](https://github.com/Danielmanoj/basic-nn-model/assets/69635071/569904d8-9747-4299-b6e2-cc58c500ce48)


## RESULT

A neural network regression model for the given dataset has been developed Sucessfully.
