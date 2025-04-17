import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.datasets import fetch_openml
basketball = fetch_openml(name="baskball", version=1)

# Loading the data
import pandas as pd
df = pd.DataFrame(basketball.data)
df.head()

x = np.array(df['age'])
y = np.array(df['time_played'])

# Fitting Models with Different Orders
import numpy.polynomial.polynomial as poly

d1 = 1
d2 = 5
beta1 = poly.polyfit(x,y,d1)
beta2 = poly.polyfit(x,y,d2)
xp = np.linspace(20,40,100)
yp_hat1 = poly.polyval(xp,beta1)
yp_hat2 = poly.polyval(xp,beta2)
# Make a scatterplot and superimpose prediction curves for d1 and d2
# Add grid lines, axis labels, and a legend
plt.plot(xp,yp_hat1,'r-',linewidth=2,label='Predicted (d=1)')
plt.plot(xp,yp_hat2,'g-',linewidth=2,label='Predicted (d=5)')
plt.scatter(x,y,label='Data')
plt.xlim(20,40)
plt.ylim(5,45)
plt.xlabel('age')
plt.ylabel('time played')
plt.legend(loc='upper right')
plt.grid(True)

# K-fold Cross-Validation

# Create a k-fold object
k = 10
kfo = model_selection.KFold(n_splits=k,shuffle=True)
# Try model orders d between 0 and 7
dtest = dtest = np.arange(0,7)
nd = len(dtest)
nRSScv = np.zeros((nd,k))
# Loop over the folds
for itsplit, Ind in enumerate(kfo.split(x)):
    # Get the training data in the split
    Itr, Its = Ind
    xtr = x[Itr]
    ytr = y[Itr]
    xts = x[Its]
    yts = y[Its]
    # Loop over the model order
    for it, d in enumerate(dtest):
        # Fit data on training folds
        beta_hat = poly.polyfit(xtr,ytr,d)
        # Measure nRSS on test fold
        yhat = poly.polyval(xts,beta_hat)
        nRSScv[it,itsplit] = np.mean((yhat-yts)**2)

# minimize mean cv-nrss
nRSS_mean = np.mean(nRSScv,axis=1)
nRSS_se = np.std(nRSScv,axis=1,ddof=1)/np.sqrt(k)
imin = np.argmin(nRSS_mean)
print("The model order that minimizes mean CV-nRSS is {0:d}".format(dtest[imin]))

# one-standard-error rule
nRSS_tgt = nRSS_mean[imin] + nRSS_se[imin]
I = np.where(nRSS_mean <= nRSS_tgt)[0]
iose = I[0]
dose = dtest[iose]
print("The model order estimated by the one-standard-error rule is %d" % dose)

# the mean CV-nRSS curve with errorbars
plt.errorbar(dtest, nRSS_mean, yerr=nRSS_se, fmt='-')
# model order yielding minimum mean CV-nRSS
plt.plot([dtest[imin],dtest[imin]],[0,100],'r--')
# target nRSS
plt.plot([dtest[0],dtest[imin]], [nRSS_tgt, nRSS_tgt], '--')
# model order estimated by the one-standard-error rule
plt.plot([dose,dose], [0,100], 'g--')

xp = np.linspace(0,100,100)
# minimizes mean CV-nRSS
betamin = poly.polyfit(x,y,dtest[imin])
yp_hatmin = poly.polyval(xp,betamin)
plt.plot(xp,yp_hatmin,'r-',linewidth=2,label='Predicted CV-nRSS')
# one-standard-error rule
betaose = poly.polyfit(x,y,dose)
yp_hatose = poly.polyval(xp,betaose)
plt.plot(xp,yp_hatose,'g-',linewidth=2,label='Predicted OSE')
# plot data
plt.scatter(x,y,label='Data')
plt.xlim(20,40)
plt.ylim(5,45)
plt.xlabel('age')
plt.ylabel('time played')
plt.legend(loc='upper right')
plt.grid(True)