
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#generate some sparse data to play with
#data is sparse signal corrupted with an additive noise

np.random.seed(42)
n_sample,n_features = 50,200
X = np.random.randn(n_sample,n_features)
coef = 3 * np.random.randn(n_features)
inds = np.arange(n_features)
np.random.shuffle(inds)#make the quene no noder
coef[inds[10:]] = 0#sparse coef
y = np.dot(X,coef)

#add noise
y += 0.01 * np.random.normal((n_sample,))

#spilt data in train set and test set
n_sample = X.shape[0]
X_train,y_train = X[:n_sample/2],y[:n_sample/2]
X_test,y_test = X[n_sample/2:],y[n_sample/2:]

#Lasso
from sklearn.linear_model import Lasso
alpha = 0.1
lasso = Lasso(alpha = alpha)
y_pred_lasso = lasso.fit(X_train,y_train).predict(X_test)
r2_score_lasso = r2_score(y_test,y_pred_lasso)
print(lasso)
print("r^2 on test data:%f" % r2_score_lasso)

#ElasticNet
from sklearn.linear_model import ElasticNet
enet = ElasticNet(alpha = alpha,l1_ratio = 0.7)

y_pred_enet = enet.fit(X_train,y_train).predict(X_test)
r2_score_enet = r2_score(y_test,y_pred_enet)

print(enet)
print("r^2 on test data:" % r2_score_enet)

#plot
plt.plot(enet.coef_,color = 'lightgreen',linewidth = 2,label = 'ElasticNet coefficient')
plt.plot(lasso.coef_,color = 'gold',linewidth = 2,label = 'lasso coefficient')
plt.plot(coef,'--',color = 'navy',label = 'original coefficient')
plt.legend(loc = 'best')
plt.title('Lasso R^2:%f,ElasticNet R^2:%f' % (r2_score_lasso,r2_score_enet))
plt.show()
