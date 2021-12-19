import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import rcParams
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
np.set_printoptions(suppress=True)


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
    plt.figure(figsize=(8,5))
    plt.title(r'The Figure of $\ \frac{c\Lambda_{m}}{c^{\ominus}}$ - $\frac{1}{\Lambda_{m}}$')
    plt.xlabel(r'$\frac{1}{\Lambda_{m}}\ $/ mol$\cdot \rm{S}^{-1}\cdot \rm{m}^{2}$')
    plt.ylabel(r'$\frac{c\Lambda_{m}}{c^{\ominus}}\ $/ $\rm{S}\cdot\rm{m}^{2}\cdot\rm{mol}^{-1}$')
    # plt.axis([0, 4.5,0.03, 0.07])
    # plt.axis([])
    return plt



c = [0.01,0.02,0.03,0.04,0.05]
c = np.array(c)
c = 1000*c

k = [156.1,224,276,320,357]
kh = 2.50/10000
k = np.array(k)
k = k/10000

lc = 390.72/10000
k = k-kh
l = k/c
y = c*l
y = y/1000
x = 1/l

A = np.polyfit(x,y,1)
B = np.poly1d(A)
print('图像的拟合函数：')
print(B)
D = B.deriv()
print('求导之后得到斜率：')
print(D)
print('标准电离平衡常数为斜率除以无限稀薄摩尔电导率的平方：')
lc2 = lc*lc
D = D/lc2
print(D)

plt=runplt()
plt.grid(zorder=0)
plt.scatter(x,y,c='purple',marker='o',label='original datas',zorder=3)
c2 = np.linspace(600,1430,500)
s2 = make_interp_spline(x,B(x))(c2)
plt.plot(c2,s2,ls='--',c='orange',label=r'Linear Fitting',zorder=2)
plt.legend(loc='upper left')
plt.savefig('Map.pdf',bbox_inches='tight')
plt.show()