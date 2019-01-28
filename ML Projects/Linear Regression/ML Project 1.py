#!/usr/bin/env python
# coding: utf-8

# Mohak Kant - R11481106
# Machine Learning Project 1 - Linear Regression
# 26th January, 2019

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_excel('proj1Dataset.xlsx');
dataset.head()

# Convert Data to Numpy Arrays
predictors = np.array(dataset['Weight'])
targets = np.array(dataset['Horsepower'])


# Clean Up Data: Ommiting Non Finite Values or missing data
if(np.sum(~np.isfinite(targets))):
    clean_indices = np.argwhere(np.isfinite(targets))
    predictors = predictors[clean_indices]
    targets = targets[clean_indices]
# Normalize Data
min_x = np.min(predictors)
min_t = np.min(targets)
std_x = np.std(predictors)
std_t = np.std(targets)
predictors = (predictors - min_x) / std_x
targets = (targets - min_t) / std_t


# Design Matrix X
array_of_ones = np.ones(len(predictors)).reshape(400,1)
X = np.append(predictors,array_of_ones,axis=1)

# Design Matrix t
t = targets

# Closed Form Predicted Values
predictions_closed_form = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),t)


x_mesh = np.linspace(np.min(X),np.max(X),10000)
y_closed_form = x_mesh*predictions_closed_form[0] + predictions_closed_form[1]

def calculate_output(w,X):
    return np.dot(w.T,X)
def cost_function(w,X,t):
    y = calculate_output(w,X)
    cost = np.sum(np.power(y - t,2))
    return cost

def calculate_gradient(w,X,t):
    y = calculate_output(w,X)
    dw1 = np.sum(-2*np.dot(X,(t-y).T))
    dw0 = np.sum(-2*(t-y).T)
    dw = [dw1, dw0]
    return np.array(dw).reshape(2,1)


def train(w,X,t,epochs,learning_rate,printCost=False,plotCost=False):
    costArr = []
    for i in range(epochs):
        costArr.append(cost_function(w,X,t))
        if printCost:
            print('Cost: ' + str(costArr[i]))
        grad = calculate_gradient(w,X2,t2)
        w -= learning_rate*grad
    if plotCost:
        plt.plot(list(range(epochs)),costArr)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost Function')
        


# Initialze Random Weigths and other parameters
w = np.random.randn(2,1)
learning_rate = 1e-4
epochs = 250
X2 = X.T # To insure correct dimensions
t2 = t.T # To insure correct dimensions

# Train the Model
train(w,X2,t2,epochs,learning_rate,0,0)



y = x_mesh*w[0] + w[1]

plt.subplots(2,1,figsize=(15,7))
plt.subplot(121)
# plt.scatter(dataset['Weight'],dataset['Horsepower'])
plt.scatter(predictors,targets,marker='x')
plt.plot(x_mesh,y_closed_form,color='r')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.title('(Normalized Data) Closed Form Solution' + ' \nWeights: \nw1= ' + str(predictions_closed_form[0]) + '\nw0= ' + str(predictions_closed_form[1]))
# plt.show()
plt.subplot(122)
plt.plot(x_mesh,y,color='g')
# plt.scatter(dataset['Weight'],dataset['Horsepower'])
plt.scatter(predictors,targets,marker='x')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.title('(Normalized Data) Gradient Descent Solution' + ' \nWeights: \nw1= ' + str(w[0]) + '\nw0= ' + str(w[1]))



print('Closed Form Weights: \n' + str(predictions_closed_form))
print('Gradient Decent Weights: \n' + str(w))

plt.show()
