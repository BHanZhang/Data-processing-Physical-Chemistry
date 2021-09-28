import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import rcParams
from scipy.interpolate import make_interp_spline
np.set_printoptions(suppress=True)
config = {
    "font.family":'serif',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)

print(matplotlib.matplotlib_fname())
def runplt(size=None):
    plt.figure(figsize=(10,6))
    plt.title(r'The figure of $E-T$')
    plt.xlabel(r'$T\ $\ K')
    plt.ylabel(r'$E\ $\ V')
    # plt.axis([0, 6.5, 0, 0.35])
    # plt.axis([])
    return plt

T = [25,30,35,40,45]
T = np.array(T)
T = T + 273.15

E = [1094.10,1091.15,1089.03,1086.83,1085.41]
E = np.array(E)
E = E * 0.001

A = np.polyfit(T,E,4)
B = np.poly1d(A)
C = B(T)
print(B)
D = B.deriv()
print('求导之后')
print(D)
print('25摄氏度时原电池电动势')
print(B(298.15))
print('其导数')
print(D(298.15))
plt=runplt()
plt.grid(zorder=0)
plt.scatter(T,E,c='purple',marker='o',label='original datas',zorder=3)

T2 = np.linspace(298.15,318.15,500)
E2 = make_interp_spline(T,C)(T2)
plt.plot(T2,E2,ls='--',c='orange',label='Polynomial fitting results(4 polynomial fitting)',zorder=2)
plt.text(302.9,1.09285,r'Fitting formula:$E(T)=1.18\times 10^{-7} T^{4} - 0.0001455 T^{3} + 0.06726 T^{2} - 13.82 T + 1066$')
plt.text(302.9,1.09225,r'After derivation:$\dfrac{{\rm{d}}E(T)}{{\rm{d}}T} = 4.72\times 10^{-7} T^{3} - 0.0004364 T^{2} + 0.1345 T - 13.82$')
#结果运行即可得到（即B、D是结果，已经print出来）
plt.legend(loc='upper right')
plt.savefig('6.pdf')
plt.show()
