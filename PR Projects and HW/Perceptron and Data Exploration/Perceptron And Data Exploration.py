# Mohak Kant, R11481106

########### NOTE: CHECK TERMINAL/CONSOLE FOR OUTPUTS ##########

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load Data
data = pd.read_excel('Proj1DataSet.xlsx')

#Edit Column Names
data.columns = ['Sepal Length', 'Sepal Width', 'Pedal Length', 'Pedal Width', 'species']

# Data Exploration
classes = data['species'].unique()
print('Classes', classes)
number_of_classes = len(data['species'].unique())
print('Number of Classes = ', number_of_classes)

features = list(data.columns.values)[0:-1]
number_of_features = len(features)
print('Number of Features = ', number_of_features)

# Divide into categories
setosa = data.loc[data['species'] == 'setosa']
versicolor = data.loc[data['species'] == 'versicolor']
virginica = data.loc[data['species'] == 'virginica']

# Compute Statistics
max_sepal_length = np.max(data['Sepal Length'])
max_sepal_width = np.max(data['Sepal Width'])
max_pedal_length = np.max(data['Pedal Length'])
max_pedal_width = np.max(data['Pedal Width'])
print('Max Sepal Length: ', max_sepal_length)
print('Max Sepal Width: ', max_sepal_width)
print('Max Pedal Length: ', max_pedal_length)
print('Max Pedal Width: ', max_pedal_width)

min_sepal_length = np.min(data['Sepal Length'])
min_sepal_width = np.min(data['Sepal Width'])
min_pedal_length = np.min(data['Pedal Length'])
min_pedal_width = np.min(data['Pedal Width'])
print('Min Sepal Length: ', min_sepal_length)
print('Min Sepal Width: ', min_sepal_width)
print('Min Pedal Length: ', min_pedal_length)
print('Min Pedal Width: ', min_pedal_width)

mean_sepal_length = np.mean(data['Sepal Length'])
mean_sepal_width = np.mean(data['Sepal Width'])
mean_pedal_length = np.mean(data['Pedal Length'])
mean_pedal_width = np.mean(data['Pedal Width'])
print('Mean Sepal Length: ', mean_sepal_length)
print('Mean Sepal Width: ', mean_sepal_width)
print('Mean Pedal Length: ', mean_pedal_length)
print('Mean Pedal Width: ', mean_pedal_width)

var_sepal_length = np.var(data['Sepal Length'])
var_sepal_width = np.var(data['Sepal Width'])
var_pedal_length = np.var(data['Pedal Length'])
var_pedal_width = np.var(data['Pedal Width'])
print('Variance Sepal Length: ',var_sepal_length)
print('Variance Sepal Width: ',var_sepal_width)
print('Variance Pedal Length: ',var_pedal_length)
print('Variance Pedal Width: ',var_pedal_width)

prior_setosa = len(setosa)/len(data)
prior_versicolor = len(versicolor)/len(data)
prior_virginica = len(virginica)/len(data)

# Interclass Variance
sw_sepal_length =  prior_setosa*(np.var(setosa['Sepal Length'])) + prior_versicolor*(np.var(versicolor['Sepal Length'])) + prior_virginica*(np.var(virginica['Sepal Length']))
sw_sepal_width =  prior_setosa*(np.var(setosa['Sepal Width'])) + prior_versicolor*(np.var(versicolor['Sepal Width'])) + prior_virginica*(np.var(virginica['Sepal Width']))
sw_pedal_length =  prior_setosa*(np.var(setosa['Pedal Length'])) + prior_versicolor*(np.var(versicolor['Pedal Length'])) + prior_virginica*(np.var(virginica['Pedal Length']))
sw_pedal_width =  prior_setosa*(np.var(setosa['Pedal Width'])) + prior_versicolor*(np.var(versicolor['Pedal Width'])) + prior_virginica*(np.var(virginica['Pedal Width']))
print('Interclass Variance Sepal Length: ', sw_sepal_length)
print('Interclass Variance Sepal Width: ', sw_sepal_width)
print('Interclass Variance Pedal Length: ', sw_pedal_length)
print('Interclass Variance Pedal Width: ', sw_pedal_width)

# Between Class Variance
sb_sepal_length =  prior_setosa*(np.mean(setosa['Sepal Length']) - mean_sepal_length)**2 + prior_versicolor*(np.mean(versicolor['Sepal Length']) - mean_sepal_length)**2 + prior_virginica*(np.mean(virginica['Sepal Length']) - mean_sepal_length)**2
sb_sepal_width =  prior_setosa*(np.mean(setosa['Sepal Width']) - mean_sepal_width)**2 + prior_versicolor*(np.mean(versicolor['Sepal Width']) - mean_sepal_width)**2 + prior_virginica*(np.mean(virginica['Sepal Width']) - mean_sepal_width)**2
sb_pedal_length =  prior_setosa*(np.mean(setosa['Pedal Length']) - mean_pedal_length)**2 + prior_versicolor*(np.mean(versicolor['Pedal Length']) - mean_pedal_length)**2 + prior_virginica*(np.mean(virginica['Pedal Length']) - mean_pedal_length)**2
sb_pedal_width =  prior_setosa*(np.mean(setosa['Pedal Width']) - mean_pedal_width)**2 + prior_versicolor*(np.mean(versicolor['Pedal Width']) - mean_pedal_width)**2 + prior_virginica*(np.mean(virginica['Pedal Width']) - mean_pedal_width)**2
print('B/w Class Vairance Sepal Length: ', sb_sepal_length)
print('B/w Class Vairance Sepal Width: ', sb_sepal_width)
print('B/w Class Vairance Pedal Length: ', sb_pedal_length)
print('B/w Class Vairance Pedal Width: ', sb_pedal_width)

# Data Visualization
    # Pedal Length x Sepel Length
plt.subplots(2,3,figsize=(15,15))
plt.subplot(2,2,1)
plt.scatter(setosa['Sepal Length'], setosa['Pedal Length'], marker='x')
plt.scatter(versicolor['Sepal Length'], versicolor['Pedal Length'], marker='o')
plt.scatter(virginica['Sepal Length'], virginica['Pedal Length'], marker='*')
plt.xlabel('Sepal Length (xm)')
plt.ylabel('Pedal Length (xm)')
plt.title('Pedal Lenght x Sepal Length')
plt.legend(['Setosa','Virsicolor','Virginica'])

    # Pedal Width x Sepel Width
plt.subplot(222)
plt.scatter(setosa['Sepal Width'], setosa['Pedal Width'], marker='x')
plt.scatter(versicolor['Sepal Width'], versicolor['Pedal Width'], marker='o')
plt.scatter(virginica['Sepal Width'], virginica['Pedal Width'], marker='*')
plt.xlabel('Sepal Width (xm)')
plt.ylabel('Pedal Width (xm)')
plt.title('Pedal Width x Sepal Width')
plt.legend(['Setosa','Virsicolor','Virginica'])

    # Pedal Width x Sepel Length
plt.subplot(223)
plt.scatter(setosa['Sepal Length'], setosa['Pedal Width'], marker='x')
plt.scatter(versicolor['Sepal Length'], versicolor['Pedal Width'], marker='o')
plt.scatter(virginica['Sepal Length'], virginica['Pedal Width'], marker='*')
plt.xlabel('Sepal Length (xm)')
plt.ylabel('Pedal Width (xm)')
plt.title('Pedal Width x Sepal Length')
plt.legend(['Setosa','Virsicolor','Virginica'])

    # Pedal Length x Sepel Width
plt.subplot(224)
plt.scatter(setosa['Sepal Width'], setosa['Pedal Length'], marker='x')
plt.scatter(versicolor['Sepal Width'], versicolor['Pedal Length'], marker='o')
plt.scatter(virginica['Sepal Width'], virginica['Pedal Length'], marker='*')
plt.xlabel('Sepal Width (xm)')
plt.ylabel('Pedal Length (xm)')
plt.title('Pedal Lenght x Sepal Width')
plt.legend(['Setosa','Virsicolor','Virginica'])


# In[5]:


# Pedal Length x Pedal Width
plt.figure()
plt.scatter(setosa['Pedal Width'], setosa['Pedal Length'], marker='x')
plt.scatter(versicolor['Pedal Width'], versicolor['Pedal Length'], marker='o')
plt.scatter(virginica['Pedal Width'], virginica['Pedal Length'], marker='*')
plt.xlabel('Pedal Width (xm)')
plt.ylabel('Pedal Length (xm)')
plt.title('Pedal Lenght x Pedal Width')
plt.legend(['Setosa','Virsicolor','Virginica'])
plt.rcParams["figure.figsize"] = [10,10]



# Convert Targets to zero's and Ones
data.loc[data['species']=='setosa', 'speicesN']=1
data.loc[data['species']=='virginica', 'speicesN']=2
data.loc[data['species']=='versicolor', 'speicesN']=3

columnsTitles=["Sepal Length","Sepal Width","Pedal Length","Pedal Width","speicesN","species"]
data=data.reindex(columns=columnsTitles)

# Correlation Co-efficients
# plt.subplots(1,2,figsize=(15,15))
# plt.subplot(121)
plt.matshow(data[:-1].corr(),cmap='seismic')
plt.xticks(range(len(data.columns)-1), data.columns[:-1])
plt.yticks(range(len(data.columns)-1), data.columns[:-1])
plt.colorbar()
plt.rcParams["figure.figsize"] = [25,25]


# Classes Vs Features Plot
    # Sepl vs Class
plt.subplots(2,2,figsize=(12,12))
plt.subplot(2,2,1)
plt.scatter(setosa['Sepal Length'], np.ones([len(setosa),1]), marker='x')
plt.scatter(versicolor['Sepal Length'], 2*np.ones([len(setosa),1]), marker='o')
plt.scatter(virginica['Sepal Length'], 3*np.ones([len(setosa),1]), marker='*')
plt.title('Sepl vs Class')
plt.legend(['Setosa','Virsicolor','Virginica'])
plt.xticks(np.arange(0,9,1))

    # SepW vs Class
plt.subplot(2,2,2)
plt.scatter(setosa['Sepal Width'], np.ones([len(setosa),1]), marker='x')
plt.scatter(versicolor['Sepal Width'], 2*np.ones([len(setosa),1]), marker='o')
plt.scatter(virginica['Sepal Width'], 3*np.ones([len(setosa),1]), marker='*')
plt.title('SepW vs Class')
plt.legend(['Setosa','Virsicolor','Virginica'])
plt.xticks(np.arange(0,9,1))

    # Petl vs Class
plt.subplot(2,2,3)
plt.scatter(setosa['Pedal Length'], np.ones([len(setosa),1]), marker='x')
plt.scatter(versicolor['Pedal Length'], 2*np.ones([len(setosa),1]), marker='o')
plt.scatter(virginica['Pedal Length'], 3*np.ones([len(setosa),1]), marker='*')
plt.title('Petl vs Class')
plt.legend(['Setosa','Virsicolor','Virginica'])
plt.xticks(np.arange(0,9,1))

    # PetW vs Class
plt.subplot(2,2,4)
plt.scatter(setosa['Pedal Width'], np.ones([len(setosa),1]), marker='x')
plt.scatter(versicolor['Pedal Width'], 2*np.ones([len(setosa),1]), marker='o')
plt.scatter(virginica['Pedal Width'], 3*np.ones([len(setosa),1]), marker='*')
plt.title('PetW vs Class')
plt.legend(['Setosa','Virsicolor','Virginica'])
plt.xticks(np.arange(0,9,1))


def predict_misclassification(X,t,w):
    pred = X@w
    pred[pred<=0] = -1
    pred[pred > 0 ] = 1
    missclassifications = np.where(pred!=t)[0]

    return missclassifications

def batch_perceptron(X,t,rho,epochs):

    w = np.zeros([X.shape[1],1])
    w_next = np.random.randn(X.shape[1],1)
    curr_epoch = 0

    while(curr_epoch < epochs and (np.absolute(w_next-w)>10e-6).any()):
        w_next = w
        missclassifications = predict_misclassification(X,t,w)
        for j in missclassifications:
            w = w + rho*t[j]*X[j,:].reshape(len(w),1)

        curr_epoch += 1

    print('Max Epochs: ', epochs)
    print('Number of Epochs ran for: ', curr_epoch)

    if(curr_epoch < epochs-1):
        print('Converged')
    else:
        print('Not Converged')
    return w

# Classification Tasks
    # Create Desing Matrices
X = np.array(setosa.iloc[:,:-1])
t = np.ones([len(X),1])

X = np.append(X,np.array(versicolor.iloc[:,:-1]),axis=0)
t = np.append(t,-1*np.ones([len(np.array(versicolor.iloc[:,:-1])),1]),axis=0)

X = np.append(X,np.array(virginica.iloc[:,:-1]),axis=0)
t = np.append(t,-1*np.ones([len(np.array(virginica.iloc[:,:-1])),1]),axis=0)

X = np.hstack((X,np.ones(len(X)).reshape(len(X),1)))

# 1) Setosa vs Versi+Vergi - All Features

# 1.1 Batch Perceptron
print("################# Setosa vs Versi+Vergi - All Features #####################")

w_perceptron1 = batch_perceptron(X,t,0.1,1000)
print('Batch Perceptron Weights: ', w_perceptron1)

# 1.2 Least Squares


w1 = np.linalg.pinv(X) @ t
print('Weights LS: ', w1)

print('LS Misclassifications:', len(predict_misclassification(X,t,w1)))



# 2) Setosa vs Versi+Vergi - Features 3 and 4

print("################# Setosa vs Versi+Vergi - Features 3 and 4 #####################")

X2 = X[:,2:5]

# 2.1 Batch Perceptron
w_perceptron2 = batch_perceptron(X2,t,0.1,1000)
print('Batch Perceptron Weights: ', w_perceptron2)

# 2.2 Least Squares
w2 = np.linalg.pinv(X2) @ t

plt.figure()
plt.scatter(setosa['Pedal Length'], setosa['Pedal Width'], marker='x')
plt.scatter(versicolor['Pedal Length'], versicolor['Pedal Width'], marker='o')
plt.scatter(virginica['Pedal Length'], virginica['Pedal Width'], marker='*')
plt.xlabel('Pedal Length (cm)')
plt.ylabel('Pedal Width (cm)')
plt.title('Setosa vs Versi+Vergi - Features 3 and 4')
plt.rcParams["figure.figsize"] = [9,9]


x_mesh = np.linspace(0,7.5,150).reshape(150,1)
y_ls = -(x_mesh*w2[0] + w2[2])/w2[1]
plt.plot(x_mesh,y_ls)
print('Weights LS: ', w2)

y_perc = -(x_mesh*w_perceptron2[0] + w_perceptron2[2])/w_perceptron2[1]
plt.plot(x_mesh,y_perc)
plt.ylim([0,5])

plt.legend([ 'Least Squares', 'Perceptron','Setosa','Virsicolor','Virginica'])

print('LS Misclassifications:', len(predict_misclassification(X2,t,w2)))


# Classification Tasks

# 3) Vergi vs Versi+setosa - All Features
    # Create Desing Matrices
X = np.array(virginica.iloc[:,:-1])
t = np.ones([len(X),1])

X = np.append(X,np.array(versicolor.iloc[:,:-1]),axis=0)
t = np.append(t,-1*np.ones([len(np.array(versicolor.iloc[:,:-1])),1]),axis=0)

X = np.append(X,np.array(setosa.iloc[:,:-1]),axis=0)
t = np.append(t,-1*np.ones([len(np.array(setosa.iloc[:,:-1])),1]),axis=0)

X = np.hstack((X,np.ones(len(X)).reshape(len(X),1)))

# 3.1 Batch Perceptron
print("################# Vergi vs Versi+setosa - All Features #####################")

w_perceptron3 = batch_perceptron(X,t,0.1,10000)
print('Batch Perceptron Weights: ', w_perceptron3)

# 3.2 Least Squares
w3 = np.linalg.pinv(X) @ t
print('Weights LS: ', w3)

print('LS Misclassifications:', len(predict_misclassification(X,t,w3)))


# 4) Verginica vs Setosa+Versi- Features 3 and 4

print("################# Verginica vs Setosa+Versi- Features 3 and 4 #####################")

X2 = X[:,2:5]
# 4.1 Batch Perceptron
w_perceptron4 = batch_perceptron(X2,t,0.1,10000)
print('Batch Perceptron: ', w_perceptron4)

# 4.2 Least Squares
w4 = np.linalg.pinv(X2) @ t
print('Weights LS: ', w2)

plt.figure()
plt.scatter(setosa['Pedal Length'], setosa['Pedal Width'], marker='x')
plt.scatter(versicolor['Pedal Length'], versicolor['Pedal Width'], marker='o')
plt.scatter(virginica['Pedal Length'], virginica['Pedal Width'], marker='*')
plt.xlabel('Pedal Length (cm)')
plt.ylabel('Pedal Width (cm)')
plt.title('Verginica vs Setosa+Versi- Features 3 and 4')

plt.rcParams["figure.figsize"] = [12,12]


x_mesh = np.linspace(0,7.5,150).reshape(150,1)
y = -(x_mesh*w4[0] + w4[2])/w4[1]
plt.plot(x_mesh,y)
plt.ylim([0,5])

y_perc = -(x_mesh*w_perceptron4[0] + w_perceptron4[2])/w_perceptron4[1]
plt.plot(x_mesh,y_perc)
plt.ylim([0,5])

plt.legend([ 'Least Squares', 'Perceptron','Setosa','Virsicolor','Virginica'])
print('LS Misclassifications:', len(predict_misclassification(X2,t,w4)))

# Multiclass Least Squares
X = np.array(setosa.iloc[:,:-1])
t1 = np.repeat(np.array([1, 0, 0]).reshape(1,3),50,axis=0)

X = np.append(X,np.array(versicolor.iloc[:,:-1]),axis=0)
t2 = np.repeat(np.array([0, 1, 0]).reshape(1,3), 50,axis=0)

X = np.append(X,np.array(virginica.iloc[:,:-1]),axis=0)
t3 = np.repeat(np.array([0, 0, 1]).reshape(1,3), 50,axis=0)

X = np.hstack((X,np.ones(len(X)).reshape(len(X),1)))

X = X[:,2:5]

t = np.array([t1,t2,t3])
t = t.reshape(150,3)



print("################# Multiclass LS- Features 3 and 4 #####################")

w_mc = (np.linalg.pinv(X)@t)
print("Multiclass LS, weights = \n", w_mc)

w_setosa = w_mc[:,0].reshape(3,1)
w_versi = w_mc[:,1].reshape(3,1)
w_verginica = w_mc[:,2].reshape(3,1)

w1 = w_setosa - w_versi
w2 = w_setosa - w_verginica
w3 = w_verginica - w_versi

plt.figure()
plt.scatter(setosa['Pedal Length'], setosa['Pedal Width'], marker='x')
plt.scatter(versicolor['Pedal Length'], versicolor['Pedal Width'], marker='o')
plt.scatter(virginica['Pedal Length'], virginica['Pedal Width'], marker='*')
plt.xlabel('Pedal Length (cm)')
plt.ylabel('Pedal Width (cm)')
plt.title('Multiclass LS - Features 3 and 4')
plt.legend(['Setosa','Virsicolor','Virginica'])

y1 = -(x_mesh*w1[0] + w1[2])/w1[1]
y2 = -(x_mesh*w2[0] + w2[2])/w2[1]
y3 = -(x_mesh*w3[0] + w3[2])/w3[1]

plt.plot(x_mesh,y1)
plt.plot(x_mesh,y2)
plt.plot(x_mesh,y3)
plt.ylim([0,5])

plt.rcParams["figure.figsize"] = [12,12]


# Get Predictions
y = []
for i in range(len(t)):
    pred = X[i]@w_mc
    if(np.sum(pred>0)==3):
        y.append(1)
    else:
        y.append(np.argmax((np.round(pred))))
y = np.array(y).reshape(150,1)

# Create Targets
tt = np.zeros([150,1])
tt[0:51] = 0
tt[51:101] = 1
tt[101:150] = 2

print("Total Misclassifications (Multiclass LS): ", np.sum(y!=tt))


plt.show()
