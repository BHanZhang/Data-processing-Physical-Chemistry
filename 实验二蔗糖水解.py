import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import rcParams
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
np.set_printoptions(suppress=True)
from sympy import *
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
    plt.figure(figsize=(12,7.5))
    plt.title(r'The Figure of $\ln{(\alpha_{t} - \alpha_{\infty})}-t$')
    plt.xlabel(r'$t \  / \ \rm{min}$')
    plt.ylabel(r'$\ln{(\alpha_{t} - \alpha_{\infty})}$')
    # plt.axis([0, 4.5,0.03, 0.07])
    # plt.axis([])
    return plt

t = [3,5,7,9,11,13,15,17,19,25,35,50,70,90]

a = [12.15,11.70,11.35,10.95,10.65,10.25,9.85,9.60,9.25,8,6.4,4.15,1.25,-0.6]
a = np.array(a)
a = a + 0.20
print('a修正')
print(a)

ai = [-4.60]
ai = np.array(ai)
a1 = a - ai
print('at-ainfty')
print(a1)
a2 = np.log(a1)
print('ln')
print(a2)


A = np.polyfit(t,a2,1)
B = np.poly1d(A)
print('图像的拟合函数：')
print(B)
D = B.deriv()
print('求导之后得到斜率：')
print(D)
computeCorrelation(t, a2)

plt=runplt()
plt.grid(zorder=0)
plt.scatter(t,a2,c='purple',marker='o',label='$Original \ Datas$',zorder=3)
plt.plot(np.arange(0,95,1),B(np.arange(0,95,1)),ls='--',c='orange',label=r'$After \ linear \ fitting:\ln{(\alpha_{t} - \alpha_{\infty})} =-0.01583t + 2.918\quad R^{2} = 0.993$',zorder=2)
for X, y in zip(t, a2):
    plt.text(X+1, y, (X,np.around(y,2)),ha='left', va='bottom', fontsize=6)
plt.legend(loc='upper right')
plt.savefig('ffigg.pdf',bbox_inches='tight')
plt.show()