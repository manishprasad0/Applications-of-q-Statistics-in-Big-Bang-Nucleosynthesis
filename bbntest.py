import matplotlib.pyplot as plt
import numpy as np
NA = 6.02214179e23# Avogadro number / mol^-1
kB = 8.617343e-11# Boltzmann constant / MeV/K
from scipy.special import gamma

#print(test(10.0))
# all available indices
index = ['neutron', 'proton', 'deutron', 'tritium',
         'helium3', 'helium4', 'lithium7', 'beryllium7']
label = ['n', 'p', 'd', 't', r'He$^3$',
         r'He$^4$', r'Li$^7$', r'Be$^7$']
#E=10.0
#A=S_factor(10.0)
""" # initialize() has been executed with default arguments
T,X = BBN(5e-10, index, atol=1e-13)
plt.axis([1, 1e-2, 1e-13, 2])
plt.loglog(T, X.T)
plt.xlabel('T = temperature / MeV')
plt.ylabel('X = mass fraction')
plt.legend(label)
plt.show() """

""" import numpy as np
t0=0.00731342084953392 
t1=13214.359161699398
n_step=256
t = np.geomspace(t0, t1, n_step)
#t[[0,-1]]
plt.loglog(t,X.T)
plt.ylim(bottom=1e-13)
plt.show() """