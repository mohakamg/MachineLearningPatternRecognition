import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd


X_train_100 = np.vstack((pd.read_csv('N_train_100.txt').columns.values.astype(np.float),np.array(pd.read_csv('N_train_100.txt'))))
t_train_100 = X_train_100[:, -1]
X_train_100 = X_train_100[:,:-1]

X_train_1000 = np.vstack((pd.read_csv('N_train_1000.txt').columns.values.astype(np.float),np.array(pd.read_csv('N_train_1000.txt'))))
t_train_1000 = X_train_1000[:, -1]
X_train_1000 = X_train_1000[:,:-1]

X_test = np.vstack((pd.read_csv('N_test.txt').columns.values.astype(np.float),np.array(pd.read_csv('N_test.txt'))))
t_test = X_test[:, -1]
X_test = X_test[:,:-1]


# Naive Bayes
def naive_bayes_classifier(train_data, Priors, test_data, labels):
    
    n = len(train_data)
    c1_train = train_data[:int(n*Priors[0]),:]
    c2_train = train_data[int(n*Priors[0]):,:]
    mean_class1 = np.mean(c1_train, axis=0)
    mean_class2 = np.mean(c2_train, axis=0)
    mean = np.vstack((mean_class1,mean_class2))
    
    var_class1 = np.var(c1_train, axis=0)
    var_class2 = np.var(c2_train, axis=0)
    var = np.vstack((var_class1,var_class2))
    
    [row, col] = test_data.shape
    naive_bayes1 = np.ones([1, row])
    naive_bayes2 = np.ones([1, row])
    
    for i in range(col):
        naive_bayes1 *= norm.pdf(test_data[:,i], mean[0,i], np.sqrt(var[0,i]))*Priors[0]
        naive_bayes2 *= norm.pdf(test_data[:,i], mean[1,i], np.sqrt(var[1,i]))*Priors[1]
        
    naive_bayes1 = naive_bayes1.reshape(row,1)
    naive_bayes2 = naive_bayes2.reshape(row,1)
        
    posterior_probs = []
    for i in range(row):
        if(naive_bayes1[i] > naive_bayes2[i]):
            posterior_probs.append(1)
        else:
            posterior_probs.append(0)
            
    posterior_probs = np.array(posterior_probs)
    accuracy = np.mean(np.array(posterior_probs) == labels)
    test_error = np.sum(np.array(posterior_probs) != labels)
    return posterior_probs, accuracy, test_error



P = np.array([1/2, 1/2])

# Bayes Classifer
def bayes_classifier(train_data, Priors, test_data, labels):
    n = len(train_data)
    c1_train = train_data[:int(n*Priors[0]),:]
    c2_train = train_data[int(n*Priors[0]):,:]
    mean_class1 = np.mean(c1_train, axis=0)
    mean_class2 = np.mean(c2_train, axis=0)
    
    mean = np.array([mean_class1,mean_class2])
    
    cov_matrix1 = np.cov(c1_train.T) 
    cov_matrix2 = np.cov(c2_train.T)
    cov_matrix = np.array([cov_matrix1, cov_matrix2])
    [row, col] = test_data.shape
    
    posterior_probs = np.zeros([1,row])[0]
    for i in range(row):
        g1 = -0.5 * (test_data[i] - mean[0]) @ np.linalg.inv(cov_matrix[0]) @ (test_data[i] - mean[0]).T - col/2 * np.log(2*np.pi) - 0.5*np.log(np.linalg.det(cov_matrix[0]))
        g2 = -0.5 * (test_data[i] - mean[1]) @ np.linalg.inv(cov_matrix[1]) @ (test_data[i] - mean[1]).T - col/2 * np.log(2*np.pi) - 0.5*np.log(np.linalg.det(cov_matrix[1]))
        if g1 >= g2:
            posterior_probs[i] = 1
        else:
            posterior_probs[i] = 0
            
   
    accuracy = np.mean(np.array(posterior_probs) == labels)
    test_error = np.sum(np.array(posterior_probs) != labels)
    return posterior_probs, accuracy, test_error


# Bayes Classifer Using Orignal Parameters
def bayes_classifier_original_params(mean, cov_matrix, Priors, test_data, labels):

    [row, col] = test_data.shape
    
    posterior_probs = np.zeros([1,row])[0]
    for i in range(row):
        g1 = -0.5 * (test_data[i] - mean[0]) @ np.linalg.inv(cov_matrix[0]) @ (test_data[i] - mean[0]).T - col/2 * np.log(2*np.pi) - 0.5*np.log(np.linalg.det(cov_matrix[0]))
        g2 = -0.5 * (test_data[i] - mean[1]) @ np.linalg.inv(cov_matrix[1]) @ (test_data[i] - mean[1]).T - col/2 * np.log(2*np.pi) - 0.5*np.log(np.linalg.det(cov_matrix[1]))
        if g1 >= g2:
            posterior_probs[i] = 1
        else:
            posterior_probs[i] = 0
            
   
    accuracy = np.mean(np.array(posterior_probs) == labels)
    test_error = np.sum(np.array(posterior_probs) != labels)
    return posterior_probs, accuracy, test_error



m1 = [0]*5
m2 = [1]*5
m = np.array([m1, m2])
S1 = [[0.8, 0.2, 0.1, 0.05, 0.01],
     [0.2, 0.7, 0.1, 0.03, 0.02],
     [0.1, 0.1, 0.8, 0.02, 0.01],
     [0.05, 0.03, 0.02, 0.9, 0.01],
     [0.01, 0.02, 0.01, 0.01, 0.8]]
S2 = [[0.9, 0.1, 0.05, 0.02, 0.01],
     [0.1, 0.8, 0.1, 0.02, 0.02],
     [0.05, 0.1, 0.7, 0.02, 0.01],
     [0.02, 0.02, 0.02, 0.6, 0.02],
     [0.01, 0.02, 0.01, 0.02, 0.7]]
S = np.array([S1,S2])



naive_bayes_classifier_100 = naive_bayes_classifier(X_train_100, P, X_test, t_test)
naive_bayes_classifier_1000 = naive_bayes_classifier(X_train_1000, P, X_test, t_test)
bayes_classifier_100 = bayes_classifier(X_train_100, P, X_test, t_test)
bayes_classifier_1000 = bayes_classifier(X_train_1000, P, X_test, t_test)
original_params = bayes_classifier_original_params(m, S, P, X_test, t_test)



print("Naive Bayes Classier Test Error (N = 100): ", naive_bayes_classifier_100[2])
print("Naive Bayes Classier Test Error (N = 1000): ", naive_bayes_classifier_1000[2])
print("Bayes Classier using MLE Test Error (N = 100): ", bayes_classifier_100[2])
print("Bayes Classier using MLE Test Error (N = 1000): ", bayes_classifier_1000[2])
print("Bayes Classier using original MLE Parameters Test Error: ", original_params[2])





