# reference:
#   R. A. Alpher, J. W. Follin and R. C. Herman
#     Physical Review 92 (1953) 1347
#   E. W. Kolb and M. S. Turner
#     "The Early Universe" chapter 4

import numpy as np
from scipy.integrate import solve_ivp,quad
from scipy.special import roots_laguerre
from constants import me,rad_const,mp,mn,Grav,hbar,tau_n

N = 64 # number of nodes for gaussian quadrature
node,weight = roots_laguerre(N)

def expansion(T0, T1, N_nu=3, n_step=256):
    """ adiabatic expansion of the early universe
    T0 = initial temperature /MeV at time t=0
    T1 = final temperature / MeV
    n_step = number of steps at which data are saved
             with logarithmic interval of T
    N_nu = number of neutrino generation
    return T, T_nu, t where
      T = photon temperature / MeV
      T_nu = neutrino temperature / MeV
      t = age of the universe /sec from T=T0
      each shape (n_step,)
    """
    global a_nu # neutrino radiation constant
    a_nu = rad_const*0.875*N_nu
    T = np.geomspace(T0, T1, n_step)
    s = solve_ivp(expansion_eq, T[[0,-1]], [T0,0], t_eval=T)
    return s.t, s.y[0], s.y[1]

G8P3 = np.sqrt(8*np.pi*Grav/3)/hbar;

def expansion_eq(T, y):
    """
    T  = photon temperature / MeV
    y[0] = T_nu = neutrino temperature / MeV
    y[1] = t = age of the universe from T=T0 / sec
    return [d(T_nu)/dT, dt/dT]
    """
    T_nu = y[0] # neutrino temperature
    E_nu = a_nu * T_nu**4 # neutrino energy density
    E_r = rad_const * T**4 # photon energy density
    P_r = E_r/3 # photon pressure
    c_r = 4*E_r # photon specific heat * T
    E_e, P_e, c_e = electron_gas(T)
    E = E_e + E_r
    P = P_e + P_r
    c = (c_e + c_r)/T
    H = G8P3*np.sqrt(E + E_nu) # dln(a)/dt = expansion rate
    dy = c/(E+P)/3 # -dln(a)/dT = dln(T_nu)/dT
    return [dy*T_nu, -dy/H]

def electron_gas(T):
    """ ideal gas of electrons and positrons
    T = temperature / MeV
    return E,P,c where
      E = energy density of gas
      P = pressure of gas
      c = dE/dlnT = (specific heat of gas)*T
        in units MeV^4/(hbar*c)^3
    """
    a = np.expand_dims(me/T, -1)
    x = node
    x2 = x*x
    y = np.sqrt(x*x + a*a)
    z = np.exp(y)
    f = np.asarray([y, x2/y/3, y*y*z/(z+1)])
    f*= x2/(z+1)*np.exp(x)
    f = np.dot(f, weight) # integral [0,infty]
    f*= T**4*2/np.pi**2 # 4*(4pi)/(2pi)^3,
    # 4 = (electron spin) + (positron spin)
    return f

q = (mn-mp)/me
k = 1/quad(lambda x: np.sqrt(x**2-1)*x*(q-x)**2, 1, q)[0]

def weak_rate(T, T_nu, tau_n=tau_n):
    """ proton-neutron weak interaction
    T = photon temperature / MeV
    T_nu = neutrino temperature / MeV
    tau_n = neutron mean lifetime / sec
    return pn,np where
      pn = proton to neutron conversion rate / sec^-1
      np = neutron to proton conversion rate / sec^-1
    """
    mT = me/T
    a = np.expand_dims(mT, -1)
    b = np.expand_dims(T/T_nu, -1)
    c = np.expand_dims(q*mT, -1)

    x = node
    y = x+a
    z = np.exp(y)
    z1,z2 = np.exp((y+c)*b), np.exp((y-c)*b)
    y1,y2 = (y+c)**2/(z1+1), (y-c)**2/(z2+1)
    f = np.asarray([y1*z + y2*z2, y2*z + y1*z1])
    f *= y*np.sqrt(x*(x+2*a))/(z+1)*np.exp(x)
    f = np.dot(f, weight)# integral [0,infty]
    return f/mT**5*k/tau_n

