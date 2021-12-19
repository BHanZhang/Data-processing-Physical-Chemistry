import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
from pathlib import Path
from scipy.interpolate import make_interp_spline
np.set_printoptions(suppress=True)
fpath = Path(mpl.get_data_path(), "/System/Library/Fonts/Supplemental/Times New Roman.ttf")

#沸点
t = np.array([78.00,76.04,74.21,72.30,71.18,70.10,68.75,67.72,66.77,65.65,65.22,64.80,64.60,64.53,64.50,65.10,67.04,68.12,71.21,74.04,75.29,77.01,78.60,80.45])
T = t + 273.15

#液相浓度

cl = np.array([0,0.018,0.036,0.051,0.078,0.093,0.116,0.136,0.152,0.238,0.259,0.359,0.424,0.524,0.546,0.702,0.732 ,0.765 ,0.799 ,0.851 ,0.937 ,0.946 ,0.955 ,1.000 ])
cg = np.array([0,0.076,0.146,0.287,0.312,0.344,0.395,0.451,0.463,0.489,0.492,0.520,0.524,0.550,0.557,0.539 ,0.565 ,0.590 ,0.623 ,0.751 ,0.816 ,0.870 ,0.955 ,1.000 ])

def runplt(size=None):
    plt.figure(figsize=(12,7.5))
    plt.title(r'Ethanol-cyclohexane phase diagram$\ (p = constant)$', font=fpath,fontsize=13)
    plt.xlabel(r'$x_{\rm{cyclohexane}}$', font=fpath,fontsize=13)
    plt.ylabel(r'$T \ / \ $K', font=fpath,fontsize=13)
    plt.axis([0, 1,337, 354])
    # plt.axis([])
    return plt


A = np.polyfit(cl[0:15],T[0:15],6)
B = np.poly1d(A)
C = np.polyfit(cg[0:15],T[0:15],6)
D = np.poly1d(C)
z1 = np.roots(-D+B)
print(z1)
print(D(z1[0]))
E = np.polyfit(cl[13:24],T[13:24],2)
F = np.poly1d(E)
G = np.polyfit(cg[13:24],T[13:24],5)
K = np.poly1d(G)

z2 = np.roots(F-K)
print(z2)
print(K(z2[3]))
z = z1[0].real + z2[3].real
z = z/2

Tl = D(z1[0]) + K(z2[3])
Tl =Tl /2
print(z)
print(Tl)


# C = np.polyfit(cg,T,5)
# D = np.poly1d(C)


# c1 = np.linspace(0,cl[14],500)
# s1 = make_interp_spline(cl[0:14],T[0:14])(c1)


plt = runplt()
plt.grid(zorder=0)
plt.scatter(cl,T,c='purple',marker='o',label='Liquid phase points',zorder=3)
plt.scatter(cg,T,c='blue',marker='o',label='Gas phase points',zorder=3)
plt.plot(np.arange(0,cl[14],0.01),B(np.arange(0,cl[14],0.01)),ls='-',c='orange',label='Liquid phase Curve',zorder=2)
plt.plot(np.arange(0,cg[14],0.01),D(np.arange(0,cg[14],0.01)),ls='-',c='green',label='Gas phase Curve',zorder=2)

plt.plot(np.arange(cl[14],1.0,0.001),F(np.arange(cl[14],1.0,0.001)),ls='-',c='orange',zorder=2)
plt.plot(np.arange(cg[15]+0.002,1.01,0.001),K(np.arange(cg[15]+0.002,1.01,0.001)),ls='-',c='green',zorder=2)
plt.text(0.5,352,('g'),ha='left', va='top', font=fpath, fontsize=20)
plt.text(0.1,339.4,('l'),ha='left', va='top', font=fpath, fontsize=20)
plt.text(0.9,339.4,('l'),ha='left', va='top', font=fpath, fontsize=20)
plt.text(0.21,342,('g+l'),ha='left', va='top', font=fpath, fontsize=20)
plt.text(0.67,342,('g+l'),ha='left', va='top', font=fpath, fontsize=20)

# plt.plot(np.arange(0,1,0.01),D(np.arange(0,1,0.01)),ls='-',c='green',zorder=2)
plt.legend(loc='upper left')
plt.savefig('相图.pdf',bbox_inches='tight')
plt.show()


np.savetxt("new.csv", cg, delimiter=',')


