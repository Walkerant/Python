import os
import numpy as np
import matplotlib.pyplot as plt

#change the directory
os.chdir('E:\Code\Python\MachineLearning\LinearRegression')

data = np.loadtxt('ex1data2.txt',delimiter = ',')
X = data[:,:2]
y = data[:,-1]

#define some parameters
m = X.shape[0]
n = X.shape[1]
theta = np.ones(n + 1)
numIter = 1000
lossArr = np.zeros(numIter)

def featureScale(data):
    data_mean = np.mean(data,axis = 0)
    data_std = np.mean(data,axis = 0)
    return ((data - data_mean)/data_std)



alpha = input('please enter alpha:')
alpha = float(alpha)
#insert the first column in the data
X = np.c_[np.ones(m),featureScale(X)]
# print(X)

y /= np.max(y)
# y = featureScale(y)

#define hypothsis
def hypothsis(X_train,theta):
    return np.dot(X_train,theta.T)

#define cost function
def costFunction(X_train,y):
    h = hypothsis(X,theta)
    return np.sum((h - y)**2) / 2*m

#define Gradient descent algrothm
def updateTheta(alpha,train_X):
    global theta
    theta -= (alpha / m) * (np.dot(train_X.T,hypothsis(train_X,theta) - y))

#use normal equation resolve
#define the inverse matrix of a matrix
inv = np.linalg.inv
def normal_equation(train_X,train_y):
    x2 = np.dot(train_X.T,train_X)
    x_ = inv(x2)
    return np.dot(np.dot(x_,train_X.T),train_y)

normal_equation_theta1,normal_equation_theta2,normal_equation_theta3 = normal_equation(X,y)



for i in range(numIter):
    lossArr[i] = costFunction(X,y)
    print('%d:%.2f' % (i,lossArr[i]))
    updateTheta(alpha,X)

print(theta)
print(normal_equation(X,y))

plt.plot(lossArr)
plt.axhline(y = lossArr[-1],color = 'r',linewidth = 2)
plt.annotate(('min loss:%.2f' % lossArr[-1]), xy = (1.5, lossArr[-1] + 30))
ax = plt.gca()
ax.set_xscale('log')
plt.title('multiple variables LinearRegression \nwith Gradient descent and normal_equation')
plt.annotate(('theta \nGradient descent:%.2f %.2f %.2f\nnormal  equation:%.2f %.2f %.2f'
%(theta[0],theta[1],theta[2],
normal_equation_theta1,normal_equation_theta2,normal_equation_theta3)),
xy = (10,500))
plt.xlabel('number of Iteration %d' % numIter)
plt.ylabel('Cost from number of Iteration')

plt.show()
