# reference:
#   P. J. E. Peebles
#     The Astrophysical Journal 146 (1966) 542
#   E. W. Kolb and M. S. Turner
#     "The Early Universe" chapter 4

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from constants import hbar,c,zeta3,mn,mp,tau_n
from expansion import expansion,weak_rate
from nuclear import element,neutron,proton,reaction,reaction_rate

n_index = element.index(neutron)
p_index = element.index(proton)

N = len(element) # number of chemical elements
M = len(reaction) # number of nuclear reactions

index = np.full((2,2,M), -1, dtype=np.int) # table of nuclear reactions
bind = np.empty(M) # binding energy
balance = np.empty(M) # balancing factor for backward reaction
A = [e.mass_num for e in element] # mass number of elements
e_name = [e.name for e in element] # name of elements
single = [len(r['product'])==1 for r in reaction] # single product or not

for k,r in enumerate(reaction):
    for i,q in enumerate(r.values()):
        for j,p in enumerate(q):
            index[i,j,k] = element.index(p)
    m = [[p.mass for p in q] for q in r.values()]
    g = [[p.spin for p in q] for q in r.values()]
    bind[k] = np.sum(m[0]) - np.sum(m[1])
    balance[k] = np.prod(g[0])/np.prod(g[1])*(
                 np.prod(m[0])/np.prod(m[1]))**1.5

balance[single] /= (2*np.pi)**1.5*(hbar*c)**3

def initialize(T_init=1e1, T_final=1e-2, N_nu=3, tau_n=tau_n):
    """ thermodynamics of the early universe
    T_init,T_final = temperature / MeV
    N_nu = number of neutrino generation
    tau_n = neutron mean lifetime / sec
    """
    global T0,T1,T,T_nu,time,p_n,n_p,interp
    T0,T1 = T_init, T_final
    T, T_nu, time = expansion(10*T0, T1, N_nu) # cosmic expansion
    p_n, n_p = weak_rate(T, T_nu, tau_n) # proton-neutron conversion rate
    interp = interp1d(time, [T, T_nu, p_n, n_p],
                      'cubic', fill_value='extrapolate')
initialize()

def rate(T, T_nu):
    """
    T = temperature / MeV
    T_nu = neutrino temperature / MeV
    return [r1,-r2] (shape (2,M)) where
      r1 = forward reaction rate / sec^-1
      r2 = backward reaction rate / sec^-1
    """
    n = nhc3*T_nu**3 # number density of nucleons / cm^-3
    r = reaction_rate(T)
    r[index[0,0]==index[0,1]] /= 2 # halve rate for identical particles
    r1 = r*n
    r2 = r1*balance*np.exp(-bind/T)
    r2[single] *= T**1.5/n
    return np.asarray([r1, -r2])

def reaction_eq(t, y):
    """ right hand side of differential equations
    t = time / sec (unused)
    y = number density of element / that of nucleon
    return f = dy/dt
    """
    T, T_nu, p_n, n_p = interp(t)
    R = rate(T, T_nu)
    X = np.prod(np.append(y,1)[index], axis=1)
    X = np.einsum('ij,ij->j', R, X)
    f = np.zeros(N+1)
    f[n_index] = y[p_index]*p_n - y[n_index]*n_p
    f[p_index] = -f[n_index]
    for k,X in enumerate(X):
        for j in range(2):
            f[index[0,j,k]] -= X # reactor decrease
            f[index[1,j,k]] += X # product increase
    return f[:-1]

def jac(t,y):
    """ jacobian of reaction_eq
    used in implicit solver for stiff equation """
    T, T_nu, p_n, n_p = interp(t)
    R = rate(T, T_nu)
    X = R[:,np.newaxis,:] * np.append(y,1)[index]
    f = np.zeros((N+1,N+1))
    f[n_index, n_index] = -n_p
    f[n_index, p_index] =  p_n
    f[p_index, n_index] =  n_p
    f[p_index, p_index] = -p_n
    for k in range(M):
        for i in range(2):
            for j in range(2):
                for l in range(2):
                    f[index[0,l,k],index[i,j,k]] -= X[i,1-j,k]
                    f[index[1,l,k],index[i,j,k]] += X[i,1-j,k]
    return f[:-1,:-1]

def bbn(eta, index, n_step=256, **kw):
    """ Big Bang Nucleosynthesis
    eta = baryon to photon ratio (Kolb & Turner eq.3.104)
    index = list of element's name (string)
            (available indices are given in nuclear.py)
    n_step = number of steps at which data are evaluated
             with logarithmic interval in T
    kw = keyword arguments passed to solve_ivp, e.g.
      atol = tolerance for absolute error (default 1e-6)
      rtol = tolerance for relative error (default 1e-3)
    return T,X where
      T = temperature / MeV (shape(n_step,)) in [T_init, T_final]
      X = mass fraction of elements (shape(len(index),n_step))
    """
    global nhc3
    nhc3 = 2.75*eta*2*zeta3/np.pi**2/(hbar*c)**3
    t0,t1 = np.interp([T0,T1], T[::-1] , time[::-1])
    t = np.geomspace(t0, t1, n_step)
    y = np.zeros(N)
    y[n_index] = 1/(np.exp((mn-mp)/T0) + 1)
    y[p_index] = 1 - y[n_index] # initial condition
    s = solve_ivp(reaction_eq, t[[0,-1]], y, 'Radau',
                  t_eval=t, jac=jac, **kw)
    X = (s.y.T * A).T[[e_name.index(i) for i in index]]
    return np.interp(t, time, T), X
