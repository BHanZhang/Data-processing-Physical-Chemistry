import matplotlib.pyplot as plt
import numpy
import numpy as np
import matplotlib
from matplotlib import rcParams
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from scipy.linalg import solve
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
    plt.figure(figsize=(16,10))
    plt.title(r'The Figure(Benzoic acid) of $\theta - t$')
    plt.xlabel(r'$t \ / \  \rm{s}$')
    plt.ylabel(r'$\theta \ / \ ^{\circ}$C')
    # plt.axis([0, 4.5,0.03, 0.07])
    # plt.axis([])
    return plt

# theta = [0.009,0.016,0.019,0.023,0.027,0.029,0.129,0.477,0.693,0.821,0.902,0.960,1.002,1.027,1.053,1.075,1.094,1.110,1.122,1.134,1.146,1.155,1.163,1.172,1.178,1.185,1.191]
theta = [0.002,0.004,0.006,0.007,0.008,0.009,0.055,0.364,0.762,1.068,1.270,1.385,1.469,1.533,1.575,1.609,1.638,1.660,1.681,1.697,1.711,1.723,1.725,1.733,1.740,1.746,1.751,1.756]
theta = np.array(theta)
theta = theta + 20.932

t = [60,120,180,240,300,330,360,390,420,450,480,510,540,570,600,630,660,690,720,750,780,810,840,870,900,930,960,990]

print(theta[6])
print(t[6])
print(theta[21])

A = np.polyfit(t[0:5],theta[0:5],1)
B = np.poly1d(A)
print('点火前：')
print(B)

C = np.polyfit(t[20:28],theta[20:28],1)
D = np.poly1d(C)
print('点火后：')
print(D)

E = np.polyfit(t[4:20],theta[4:20],13)
F = np.poly1d(E)

#算算J点
Iy = (theta[6] + theta[20])/2
# print(Iy)
Ix = np.roots(F-Iy)
print(Ix)

#算算A点
Ax = Ix[9].real
Ay = B(Ax)
#算算C点
Cx = Ix[9].real
Cy = D(Cx)

U = Cy - Ay
print(r'$\Delta T $为')
print(U)

plt=runplt()
plt.grid(zorder=0)
plt.scatter(t,theta,c='purple',marker='o',label='original datas',zorder=3)
plt.scatter(Ix[9].real,Iy,c='green',marker='o',zorder=3)
plt.scatter(Ax,Ay,c='green',marker='o',zorder=3)
plt.scatter(Cx,Cy,c='green',marker='o',zorder=3)
#text
plt.text(Ix[9].real+5,Iy,(np.around(Ix[9].real,3),np.around(Iy,3)),ha='left', va='top', fontsize=10)
plt.text(Ix[9].real-5,Iy,'$I$',ha='right', va='center', fontsize=15)
plt.text(Ax+5,Ay,(np.around(Ix[9].real,3),np.around(Ay,3)),ha='left', va='top', fontsize=10)
plt.text(Ix[9].real-5,Ay+0.015,'$A$',ha='right', va='center', fontsize=15)
plt.text(Cx+5,Cy,(np.around(Ix[9].real,3),np.around(Cy,3)),ha='left', va='top', fontsize=10)
plt.text(Ix[9].real-5,Cy,'$C$',ha='right', va='center', fontsize=15)
c2 = np.linspace(360,810,40000)
s2 = make_interp_spline(t[6:21],theta[6:21])(c2)
c = np.linspace(300,390,900)
s = make_interp_spline(t[4:8],theta[4:8])(c)
plt.plot(c2,s2,ls='-',c='orange',zorder=2)
plt.plot(numpy.arange(60,Ix[9].real,1),B(numpy.arange(60,Ix[9].real,1)),ls='-',c='violet',label=r'Before ignition : $ \theta = 2.500\times 10^{-5} t + 20.930$',zorder=2)
plt.plot(c[300:600],s[300:600],ls='-',c='orange',label=r'Reaction',zorder=2)
plt.plot(numpy.arange(Ix[9].real,t[27],1),D(numpy.arange(Ix[9].real,t[27],1)),ls='-',c='violet',label=r'After ignition : $ \theta =  0.0002083t + 22.480$',zorder=2)
plt.vlines(Ix[9].real, Ay, Cy)
# plt.plot(t[5:20],F(t[5:20]),ls='-',c='GREEN',label=r'After ignition : $ \theta = 0.00025t + 21.880$',zorder=2)
plt.legend(loc='upper left')
plt.savefig('Fig2.pdf')
plt.show()
