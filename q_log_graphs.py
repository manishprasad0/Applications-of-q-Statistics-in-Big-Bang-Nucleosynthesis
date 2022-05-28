import matplotlib.pyplot as plt
import numpy as np

q=[0,1,2,2.5,3]
x = np.linspace(-1, 2, 1000)

for i in range(5):
    if q[i]==1:
        plt.plot(x,np.exp(-x), label='q=1')
    else:
        plt.plot(x,((1+(1-q[i])*x)**(1/(q[i]-1))), label='q={}'.format(q[i]))
#plt.plot(x,((1+(1-3)*x)**(1/(3-1))))
plt.ylim(0, 3)
plt.legend(loc='upper right')
plt.grid()
plt.xlabel(r'$\beta\varepsilon_i$')
plt.ylabel(r'$Z_{q}p_i$')
plt.title(r'$Variation\; of\; Z_{q}p_i\; with\; \beta\varepsilon_i$')
plt.show()