import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('E:\Code\Python\MachineLearning\LinearRegression')
#load data
data = np.loadtxt('ex1data1.txt',delimiter = ',')
X = data[:,0]
y = data[:,1]
theta = np.zeros(2)
m = np.shape(X)[0]
allIter = 10000
allCost = np.zeros(allIter)

#define hyophsis function
def hypothsis(x,theta):
    return theta[0] + x * theta[1]

#define costFunction
def costFunction(train_X,train_y,theta):
    costSum = 0
    for i in range(m):
        costSum += (hypothsis(train_X[i],theta) - train_y[i])**2
        #print(i,'cost:',costSum)
    print('costSum:',costSum / (2 * m))
    return costSum

#for every iter,update theta by using Gradient descent
def updateTheta(alpha):
    costSum1 = 0
    costSum2 = 0
    global theta
    for i in range(m):

        costSum1 += ((hypothsis(X[i],theta) - y[i])/ m)
        costSum2 += ((hypothsis(X[i],theta) - y[i]) * X[i]/ m)

    #updata theta
    theta[0] -= (costSum1 * alpha)
    theta[1] -= (costSum2 * alpha)

    print('Gradient:',theta[0],theta[1])

alpha = 0.023

for numIter in range(allIter):
    allCost[numIter] =  costFunction(X,y,theta)
    updateTheta(alpha)

print('hypothsis is %f *X + %f' % (theta[0],theta[1]))

print('Plot')
plt.plot(allCost)
ax = plt.gca()#plt.gca is meaning plt get current axes in the figure.
ax.set_xscale('log')
plt.title('LinearRegression')
plt.xlabel('the number of Iters %d' % allIter)
plt.ylabel('Cost')
plt.show()
