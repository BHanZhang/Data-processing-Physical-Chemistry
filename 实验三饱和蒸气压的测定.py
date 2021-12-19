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
from scipy import stats
config = {
    "font.family":'serif',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)

#计算r值
def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    #a、b、r
    print("使用scipy库：a：",slope,"b：", intercept,"r：", r_value,"r-squared：", r_value**2)

print(matplotlib.matplotlib_fname())
def runplt(size=None):
    plt.figure(figsize=(8,5))
    plt.title(r'The Relationship of $p - T$')
    plt.xlabel(r'$T \ / \  \rm{K}$')
    plt.ylabel(r'$p \ / \ \rm{Pa}$')
    # plt.axis([0, 4.5,0.03, 0.07])
    # plt.axis([])
    return plt

def runplt1(size=None):
    plt.figure(figsize=(8,5))
    plt.title(r'The Relationship of $\ln{p} - 1/T$')
    plt.xlabel(r'$1/T \ / \  \rm{K}^{-1}$')
    plt.ylabel(r'$\ln{p} \ / \ \rm{Pa}$')
    # plt.axis([0, 4.5,0.03, 0.07])
    # plt.axis([])
    return plt

#真空度(pa)
p1 = [-94.25,-91.59,-88.95,-85.35,-83.34,-81.81,-77.23]
p1 = np.array(p1)
p = p1 + 102.83
p = p * 1000

#热力学温度(K)
T = [25,30,34,38,40,42,45]
T = np.array(T)
T = T + 273.15
T1 = 1 / T
#拟合
def funlog(T, a, b, d):
    return a + 10 ** (b / T + d)
popt, pcov = curve_fit(funlog, T, p,maxfev=50000)
r = funlog(T, *popt)
print(popt)

p2 = popt[0] + 10 ** (popt[1] / T + popt[2])

runplt()
plt.grid(zorder=0)
plt.scatter(T,p,c='purple',marker='o',label='original datas',zorder=3)
plt.plot(T,p2,ls='-',c='orange',zorder=2)
plt.legend(loc='upper left')
plt.savefig('饱和蒸气压1.pdf',bbox_inches='tight')
plt.show()

A = np.polyfit(T1,np.log(p),1)
B = np.poly1d(A)
print(B)
print(5137*8.314)

runplt1()
plt.grid(zorder=0)
plt.scatter(T1,np.log(p),c='purple',marker='o',label='original datas',zorder=3)
plt.plot(T1,B(T1),ls='-',c='orange',label= r'$\ln{p} = \frac{-5137}{T}+ 26.28$',zorder=2)
plt.legend(loc='upper right')
plt.savefig('饱和蒸气压2.pdf',bbox_inches='tight')
rsquared(T1,np.log(p))
plt.show()

numpy.savetxt("new.csv", p1 * 1000, delimiter=',')
