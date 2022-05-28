import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

q=1.2
x = np.linspace(0, 10, 1000)
bq=np.sqrt(q-1)*(0.25)*(3*q-1)*(1+q)*gamma(0.5+(1/(q-1)))/gamma(1/(q-1))

Fq=bq*np.sqrt(x)*(1-(q-1)*x)**(1/(q-1))
Fmb=np.sqrt(x)*np.exp(-x)
plt.plot(x,Fq)
plt.plot(x,Fmb)
plt.plot(5,0,'-ro')
plt.legend(["Tsallis", "Maxwell Boltzmann"], loc ="upper right")

plt.annotate('Cutoff for q=1.2 at (5,0)', xy =(5,0), xytext =(4,0.1), arrowprops = {"arrowstyle":"->"})
plt.grid()
plt.xlabel('E in MeV')
plt.ylabel('Normalized Probability')
plt.title('Tsallis and Maxwell Boltzmann Distributions')
plt.show()