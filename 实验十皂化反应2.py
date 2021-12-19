import matplotlib.pyplot as plt
import numpy
import numpy as np
import matplotlib
from matplotlib import rcParams
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from scipy.linalg import solve
matplotlib.rcParams['text.usetex'] = True
np.set_printoptions(suppress=True)
from sympy import *
from pylab import *
import math
config = {
    "font.family":'serif',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)

#计算r值
def computeCorrelation(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)
    print("r：", SSR / SST, "r-squared：", (SSR / SST) ** 2)
    return


print(matplotlib.matplotlib_fname())
def runplt(size=None):
    plt.figure(figsize=(10,6))
    plt.title(r'The Relationship Of $\kappa_{0} - \kappa_{t} \ (30^{\circ}\rm{C})$')
    plt.ylabel(r'$\kappa_{t}\ / \rm{S}\cdot\rm{m}^{-1}$')
    plt.xlabel(r'$\displaystyle\frac{\kappa_{0} - \kappa_{t}}{t}\ / \rm{S}\cdot\rm{m}^{-1}\cdot\rm{min}^{-1}$')
    # plt.axis([np.log10(400), np.log10(100000),-1.80, 0])
    # plt.axis([])
    return plt
print(matplotlib.matplotlib_fname())
runplt()

kapa = np.array([9.24,8.10,7.40,6.92,6.57,6.30,6.10,5.93,5.79,5.67,5.57,5.48,5.41,5.34,5.28])
kapa = kapa / 10
kapa0 = 1.139
c = 0.04976
t = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

x = (kapa0 - kapa)
x = x / t
plt.scatter(x,kapa,c='tab:brown',marker='o',label = r'$k_{303.15\rm{K}} = 8.849$',zorder=3)

A = np.polyfit(x,kapa,1)
B = np.poly1d(A)
print(B)
D = B.deriv()
k = 1 / (D * c)
print('答案就是')
print(k)
y_err = x.std() * np.sqrt(1/len(x) +
                          (x - x.mean())**2 / np.sum((x - x.mean())**2))

plt.fill_between(x, B(x) - y_err, B(x) + y_err, alpha=0.2)
tick_params(direction='in')
tick_params(top='on',bottom='on',left='on',right='on')
plt.plot(x,B(x),ls='-',label=r'$\kappa_{t} = 2.271 \displaystyle\frac{\kappa_{0} - \kappa_{t}}{t} + 0.4371$',zorder=2)
plt.legend(loc='lower right')
plt.grid(zorder=0)
plt.savefig('速率常数2.pdf',bbox_inches='tight')
plt.show()
np.savetxt("new.csv", [kapa,(kapa0-kapa),x], delimiter=',')