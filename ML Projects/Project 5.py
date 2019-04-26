#!/usr/bin/env python
# coding: utf-8

# In[115]:


# Import Libraries
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
# A class that models the Neural Net with L-layers and
# N neurons in each layer. It also contains the functions
# for training, testing, and optimizing the Neural Network

np.random.seed(100)

class DeepNN:

    # Constructor to build the structure of the Neural Network
    # It accepts the layers in the format of [2,3,1] -> 2 Neuron Input Layer,
    # 3 Neuron Hidden Layer and 1 Neuron output layer
    def __init__(self, layers, activations):
        ############################### Initialize the number of layers and neurons
        self.layers = layers
        self.num_layers = len(layers)
        self.hidden_layers = len(layers) - 2
        self.input_neurons = layers[0]
        self.output_neurons = layers[-1]

        ########## Intialize parameters for Forward Propogation
        # Initialize Weights
        self.epsilon = 0.12  # Random Weight Initialization Factor
        self.weights = []
        for i in range(self.num_layers-2):
            self.weights.append(np.random.randn(layers[i]+1, layers[i+1]+1)*2*self.epsilon - self.epsilon)
                        # We add a +1 to incorporate for weights from the +1 neuron for the bias
        self.weights.append(np.random.randn(layers[-2]+1, layers[-1])*2*self.epsilon - self.epsilon)

        self.a = [] # To keep track of activations
        self.z = [] # To keep track of layer values
        self.activations = activations # Activations for each layer

        ######### Intialize parameters for Backward Propogation
        self.delta = []
        self.gradient = []

        # Initialize Scaling
        self.scaler = preprocessing.StandardScaler()

    ################################### Define Some Activation Functions and their derivatives ##################
    def sigmoid(self,z):
        return 1.0/(1.0 + np.exp(-z))

    def sigmoidPrime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def reLU(self,x):
        return np.maximum(x, 0)

    def reLUPrime(self,x):
        return np.where(x > 0, 1.0, 0.0)

    def softmax(self,x):
        return np.exp(x)/np.sum(np.exp(x), axis = 0)

    def tanh(self,z):
        return np.tanh(z)

    def tanh_prime(self,x):
        return 1 - np.tanh(x)**2

    def identity(self,x):
        return x

    def identity_prime(self,x):
        return 1

    ######################################### Cost Functions #############################################
    # Least Squares
    def least_squares_cost(self,t):
        return 0.5*np.mean( (t-self.a[-1])**2 )

    # Cross Entropy Log Loss Function
    def log_loss(self,t):
        return np.mean( np.nan_to_num( -1*t*np.log(self.a[-1]) - (1-t)*np.log(1-self.a[-1]) ) )

    ######################################### Forward Feed ##############################################
    def forwardFeed(self, X):
        a = [X] # Keep Track of activations
        z = []

        # Add Bias
        c = np.ones([1,a[0].shape[0]]).reshape(a[0].shape[0],1)
        a[0] = np.concatenate((c,a[0]), axis=1)
#         print(a)
        for i in range(self.num_layers-1):
#             print(a[i])
            z.append(np.dot(a[i],self.weights[i]))
            if(self.activations[i] == 'sigmoid'):
                a.append(self.sigmoid(z[i]))
            elif(self.activations[i] == 'reLU'):
                a.append(self.reLU(z[i]))
            elif(self.activations[i] == 'tanh'):
                a.append(self.tanh(z[i]))
            elif(self.activations[i] == 'softmax'):
                a.append(self.softmax(z[i]))
            elif(self.activations[i] == 'identity'):
                a.append(self.identity(z[i]))
        self.a = a
        self.z = z

    def backPropogate(self,y):

        delta = []
        gradient = []
        weights_flipped = self.weights[::-1]
        z_flipped = self.z[::-1]
        activations_flipped = self.a[::-1]
        activation_func_flipped = self.activations[::-1]
        delta.append(activations_flipped[0] - y)
        for i in range(0,self.num_layers-2):
                if(activation_func_flipped[i] == 'sigmoid'):
                    delta.append( np.dot(delta[i], weights_flipped[i].T ) * self.sigmoidPrime(z_flipped[i+1]) )
                elif(activation_func_flipped[i] == 'reLU'):
                    delta.append( np.dot(delta[i], weights_flipped[i].T ) * self.reLUPrime(z_flipped[i+1]) )
                elif(activation_func_flipped[i] == 'tanh'):
                    delta.append( np.dot(delta[i], weights_flipped[i].T ) * self.tanh_prime(z_flipped[i+1]) )
                elif(activation_func_flipped[i] == 'identity'):
                    delta.append( np.dot(delta[i], weights_flipped[i].T ) * self.identity_prime(z_flipped[i+1]) )
                elif(activation_func_flipped[i] == 'softmax'):
                    delta.append( np.dot(delta[i], weights_flipped[i].T))

        delta = delta[::-1]

        for i in range(len(delta)):
            gradient.append( np.dot(self.a[i].T, delta[i]) )
            
        self.delta = delta
        self.gradient = gradient

    def learn(self, epochs, learning_rate, X, y, cost_func, metrics_at=10, optimizer = '', batch_size=10, scaler_type='standard_scaler', split=False, test_size=0.25, beta = 0.9, beta_2 = 0.99, epsilon=10e-8):
        start = time.time()
        
        cost=[]
        
        v_n = []
        s_n = []
        for j in range(len(self.weights)):
            v_n.append(np.zeros(self.weights[j].shape))
            s_n.append(np.zeros(self.weights[j].shape))
            
        if scaler_type == 'min_max_scaler':
            self.scaler = preprocessing.MinMaxScaler()
        X = self.scaler.fit_transform(X)

        if split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
            X = X_train
            y = y_train

        
        for i in range(epochs):
                if optimizer == 'mini_batch':
                    # Batch size of 1 makes it SGD
                    no_of_batches = int(len(X)/batch_size)
                    for k in range(no_of_batches):
                        if k==no_of_batches-1:
                            X_sgd = X[k*batch_size:]
                            y_sgd = y[k*batch_size:]
                        else:
                            X_sgd = X[k*batch_size:(k+1)*batch_size]
                            y_sgd = y[k*batch_size:(k+1)*batch_size]
                        self.forwardFeed(X_sgd)
                        self.backPropogate(y_sgd)
                        for j in range(len(self.gradient)):
                            v_n[j] = beta * v_n[j] + (1-beta) * self.gradient[j]
                            self.weights[j] = self.weights[j] - learning_rate * v_n[j]
                elif optimizer == 'rmsprop':
                    # Batch size of 1 makes it SGD
                    no_of_batches = int(len(X)/batch_size)
                    for k in range(no_of_batches):
                        if k==no_of_batches-1:
                            X_sgd = X[k*batch_size:]
                            y_sgd = y[k*batch_size:]
                        else:
                            X_sgd = X[k*batch_size:(k+1)*batch_size]
                            y_sgd = y[k*batch_size:(k+1)*batch_size]
                            
                        self.forwardFeed(X_sgd)
                        self.backPropogate(y_sgd)
                        for j in range(len(self.gradient)):
                            s_n[j] = beta_2 * s_n[j] + (1-beta_2) * np.power(self.gradient[j],2)
                            self.weights[j] = self.weights[j] - learning_rate * self.gradient[j]/np.sqrt(s_n[j]+epsilon)
                elif optimizer == 'adam':
                    # Batch size of 1 makes it SGD
                    no_of_batches = int(len(X)/batch_size)
                    for k in range(no_of_batches):
                        if k==no_of_batches-1:
                            X_sgd = X[k*batch_size:]
                            y_sgd = y[k*batch_size:]
                        else:
                            X_sgd = X[k*batch_size:(k+1)*batch_size]
                            y_sgd = y[k*batch_size:(k+1)*batch_size]
                        self.forwardFeed(X_sgd)
                        self.backPropogate(y_sgd)
                        for j in range(len(self.gradient)):
                            v_n[j] = beta * v_n[j] + (1-beta) * self.gradient[j]
                            s_n[j] = beta_2 * s_n[j] + (1-beta_2) * np.power(self.gradient[j],2)
                            
                            # Beta Correction
                            v_n[j] = v_n[j] / (1-np.power(beta,k)+epsilon)
                            s_n[j] = s_n[j] / (1-np.power(beta_2,k)+epsilon)
                            
                            self.weights[j] = self.weights[j] - learning_rate * v_n[j]/np.sqrt(s_n[j]+epsilon)

                else:
                    self.forwardFeed(X)
                    self.backPropogate(y)
                    for j in range(len(self.gradient)):
                        self.weights[j] = self.weights[j] - learning_rate * self.gradient[j]

                if(i%metrics_at == 0):
                    self.forwardFeed(X)
                    print('Effective epoch: ', i/metrics_at + 1)
                    if(cost_func == 'log_loss'):
                        cost.append(self.log_loss(y))
                        print('Accuracy: ', np.mean(np.round(self.think(X))==y) * 100, '%')
                    elif(cost_func == 'least_squares'):
                        cost.append(self.least_squares_cost(y))
                    print('Cost: ', cost[i], '\n')

        

        end = time.time()
        
        
        if(cost_func == 'log_loss' and split):
            print('Testing Accuracy: ', np.mean(np.round(self.think(X_test))==y_test) * 100, '%')
        plt.plot(cost)
        print('Time Taken: ', end-start, ' seconds')
        return self.weights

    def think(self,X):

        a = [X] # Keep Track of activations
        z = []

        # Add Bias
        c = np.ones([1,a[0].shape[0]]).reshape(a[0].shape[0],1)
        a[0] = np.concatenate((c,a[0]), axis=1)
#         print(a)
        for i in range(self.num_layers-1):
#             print(a[i])
            z.append(np.dot(a[i],self.weights[i]))
            if(self.activations[i] == 'sigmoid'):
                a.append(self.sigmoid(z[i]))
            elif(self.activations[i] == 'reLU'):
                a.append(self.reLU(z[i]))
            elif(self.activations[i] == 'tanh'):
                a.append(self.tanh(z[i]))
            elif(self.activations[i] == 'softmax'):
                a.append(self.softmax(z[i]))
            elif(self.activations[i] == 'identity'):
                a.append(self.identity(z[i]))
                
        return a[-1]



########################## Part a) - Generate Dataset
X = np.array([[-1, -1],[1, 1],[-1, 1],[1, -1]],dtype='float')
t = np.array([1, 1, 0, 0],dtype='float').reshape(4,1)


NN = DeepNN([2,2,1], activations=['tanh','tanh','sigmoid'])

w = NN.learn(epochs=1500, learning_rate=0.1, X=X, y=t, cost_func='log_loss', metrics_at=1, split=False)
print('Accuracy: ',np.mean(np.round(NN.think(X))==t) * 100)


x_mesh = np.linspace(-1.2,1.2,30).reshape(30,1)
y_mesh = np.linspace(-1.2,1.2,30).reshape(30,1)

X_mesh, Y_mesh = np.meshgrid(x_mesh,x_mesh)
X_test = np.hstack((x_mesh,y_mesh))



# Classify Points for Decision Boundary
predicitions = np.zeros([len(x_mesh),len(y_mesh)])
for i in range(len(X_mesh)):
    for j in range(len(Y_mesh)):
        pnt = np.array([X_mesh[i,j],Y_mesh[i,j]]).reshape(1,2)
        predicitions[i,j] = NN.think(pnt)[0,0]



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X_mesh, Y_mesh, predicitions-0.475,1000)

x_xor = [-1, 1, -1, 1]
y_xor = [-1, 1, 1, -1]
t_xor = [0.5, 0.5, -0.5, -0.5]
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(x_xor, y_xor, t_xor, c='r')
plt.show()



