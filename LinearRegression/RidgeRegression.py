import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

X = 1. / (np.arange(1,11) + np.arange(0,10)[:,np.newaxis])
y = np.ones(10)

n_alpha = 200
alphas = np.logspace(-2,2,n_alpha)
clf = linear_model.Ridge(fit_intercept = False)

coefs = []
for a in alphas:
    clf.set_params(alpha = a)
    clf.fit(X,y)
    coefs.append(clf.coef_)

#可视化
ax = plt.gca()#plt.gca is meaning plt get current axes in the figure.
ax.plot(alphas,coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficeints as a function of regulation')
plt.axis('tight')
plt.show()
