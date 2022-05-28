import numpy as np
from constants import kB,NA,amu

class particle:
    def __init__(self, mass, spin, name):
        self.mass = mass
        self.spin = spin
        self.name = name
        self.mass_num = int(np.round(mass/amu))
        
    def __hash__(self):
        return hash((self.mass, self.spin))

proton = particle(938.272013, 2, 'proton')
neutron = particle(939.565346, 2, 'neutron')
deutron = particle(1875.612793, 3, 'deutron')
tritium = particle(2808.920906, 2, 'tritium')
helium3 = particle(2808.391383, 2, 'helium3')
helium4 = particle(3727.379109, 1, 'helium4')
lithium7 = particle(6533.833166, 4, 'lithium7')
beryllium7 = particle(6534.184060, 4, 'beryllium7')

reaction = [
    {'reactor':[neutron,proton], 'product':[deutron], 'use':True},
    {'reactor':[deutron,proton], 'product':[helium3], 'use':True},
    {'reactor':[deutron,deutron], 'product':[helium3,neutron], 'use':True},
    {'reactor':[deutron,deutron], 'product':[tritium,proton], 'use':True},
    {'reactor':[helium3,neutron], 'product':[tritium,proton], 'use':True},
    {'reactor':[tritium,deutron], 'product':[helium4,neutron], 'use':True},
    {'reactor':[helium3,deutron], 'product':[helium4,proton], 'use':True},
    {'reactor':[helium3,helium4], 'product':[beryllium7], 'use':True},
    {'reactor':[helium4,tritium], 'product':[lithium7], 'use':True},
    {'reactor':[beryllium7, neutron], 'product':[lithium7,proton], 'use':True},
    {'reactor':[lithium7, proton], 'product':[helium4,helium4], 'use':True }
]

mask = [r['use'] for r in reaction]
reaction = [{k:r[k] for k in ['reactor','product']}
            for r in reaction if r['use']] # remove unused reactions
element = list({p for r in reaction
                for q in r.values() for p in q})


def reaction_rate(T):
    """
    T = temperature / MeV
    return <sigma v> / cm^3 s^-1
      averaged over Maxwell distribution of v
      where sigma = cross section of reaction 
    reference: M. S. Smith, L. H. Kawano and R. A. Malaney
        The Astrophysical Journal Supplement 85 (1993) 219
    """
    T9 = T/kB*1e-9
    T912 = np.sqrt(T9)
    T932 = T9*T912
    T913 = T9**(1/3)
    T923 = T913**2
    T943 = T923**2
    T953 = T9*T923
    T9f = T9/(1 + 0.1071*T9)
    T9f13 = T9f**(1/3)
    T9f56 = T9f**(5/6)
    T9e = T9/(1 + 0.1378*T9)
    T9e13 = T9e**(1/3)
    T9e56 = T9e**(5/6)
    T9a = T9/(1 + 13.076*T9)
    T9a32 = T9a**1.5
    T9d = T9/(1 + 0.759*T9)
    T9d13 = T9d**(1/3)
    T9d56 = T9d**(5/6)

    return np.asarray([
        # n + p -> d + gamma
        4.742e+4*(1. - .8504*T912 + .4895*T9 - .09623*T932
                  + 8.471e-3*T9*T9 -2.80e-4*T9*T932),
        # p + d -> 3He + gamma
        2.65e+3/T923*np.exp(-3.720/T913)*(
            1. + .112*T913 + 1.99*T923
            + 1.56*T9 + .162*T943 + .324*T953),
        # d + d -> n + 3He
        3.95e+8/T923*np.exp(-4.259/T913)*(
            1. + .098*T913 + .765*T923 + .525*T9
            + 9.61e-3*T943 + .0167*T953)-(-0.015025080019524184 +0.26888597843943923*T9)*1e7,
        # d + d -> p + t
        4.17e+8/T923*np.exp(-4.258/T913)*(
            1. + .098*T913 + .518*T923 + .355*T9
            - .010*T943 - .018*T953)-(-0.0181475219826851 +0.33241893276000606*T9+0.8051993459837072*T9*T9-0.44992147555103057*T9*T9*T9)*1e7,
        # n + 3He -> p + t
    	7.21e+8*(1. - .508*T912 + .228*T9),
        # d + t -> n + 4He
    	1.063e+11/T923*np.exp(-4.559/T913 - (T9/.0754)**2)*(
            1. + .092*T913 - .375*T923 - .242*T9
            + 33.82*T943 + 55.42*T953
            ) + 8.047e+8/T923*np.exp(-0.4857/T9),
        # 3He + d -> 4He + p
    	5.021e+10/T923*np.exp(-7.144/T913 - (T9/.270)**2)*(
            1. + .058*T913 + .603*T923 + .245*T9
            + 6.97*T943 + 7.19*T953
            ) + 5.212e+8/T912*np.exp(-1.762/T9),
        # 3He + 4He -> 7Be + gamma
	4.817e+6/T923*np.exp(-14.964/T913)*(
            1. + .0325*T913 - 1.04e-3*T923 - 2.37e-4*T9
            - 8.11e-5*T943 - 4.69e-5*T953
            ) + 5.938e+6*T9f56/T932*np.exp(-12.859/T9f13),
        # 4He + t -> 7Li + gamma
    	3.032e+5/T923*np.exp(-8.090/T913)*(
            1. + .0516*T913 + .0229*T923 + 8.28e-3*T9
            - 3.28e-4*T943 - 3.01e-4*T953
            ) + 5.109e+5*T9e56/T932*np.exp(-8.068/T9e13)-(0.05880133242722607 -0.7004637134775736*T9-0.26197283995779763*T9*T9+7.254907923628384*T9*T9*T9),
        # 7Be + n -> 7Li + p
    	2.675e+9*(1. - .560*T912 + .179*T9 - .0283*T932
                  + 2.214e-3*T9*T9 - 6.851e-5*T9*T932
            ) + 9.391e+8*T9a32/T932 + 4.467e+7/T932*np.exp(-0.07486/T9),
        # 7Li + p -> 4He + 4He
    	1.096e+9/T923*np.exp(-8.472/T913) \
        - 4.830e+8*T9d56/T932*np.exp(-8.472/T9d13) \
    	+ 1.06e+10/T932*np.exp(-30.442/T9) \
        + 1.56e+5/T923*np.exp((-8.472/T913) - (T9/1.696)**2)*(
            1. + .049*T913 - 2.498*T923 + .860*T9
            + 3.518*T943 + 3.08*T953
            ) + 1.55e+6/T932*np.exp(-4.478/T9)
        ])[mask]/NA

def S_factor(E):
    """
    T = temperature / MeV
    return <sigma v> / cm^3 s^-1
      averaged over Maxwell distribution of v
      where sigma = cross section of reaction 
    reference: M. S. Smith, L. H. Kawano and R. A. Malaney
        The Astrophysical Journal Supplement 85 (1993) 219
    """
    T=1
    T9 = T/kB*1e-9
    T912 = np.sqrt(T9)
    T932 = T9*T912
    T913 = T9**(1/3)
    T923 = T913**2
    T943 = T923**2
    T953 = T9*T923
    T9f = T9/(1 + 0.1071*T9)
    T9f13 = T9f**(1/3)
    T9f56 = T9f**(5/6)
    T9e = T9/(1 + 0.1378*T9)
    T9e13 = T9e**(1/3)
    T9e56 = T9e**(5/6)
    T9a = T9/(1 + 13.076*T9)
    T9a32 = T9a**1.5
    T9d = T9/(1 + 0.759*T9)
    T9d13 = T9d**(1/3)
    T9d56 = T9d**(5/6)

    return np.asarray([
        # n + p -> d + gamma
        (7.31638*10**(-20) + 2.35455*10**(-20)*E**(1/2) - 1.55683*10**(-18)*E+ 5.93351*10**(-18)*E**(3/2) - 9.25443*10**(-18)*E*E 
        + 6.6732*10**(-18)*E**(5/2)- 1.82393*10**(-18)*E*E*E),
        # p + d -> 3He + gamma
        (0.214*10**(-6) + 0.556*10**(-5)*E + 0.551*10**(-5)*E*E - 0.157*10**(-5)*E*E*E  ) ,
        # d + d -> n + 3He
        0.0522 + 0.370*E - 0.196*E*E + 0.0672*E*E*E - 0.00885*E*E*E*E,
        # d + d -> p + t
        (0.0542 + 0.205*E - 0.024*E*E),
        # n + 3He -> p + t
    	(0.706*10**9 - 0.149*(10**10)* E**(1/2) + 0.521*(10**10)*E - 0.239*(10**11)*E**(3/2)+ 0.617*(10**11)*E*E 
        - 0.449*(10**11)*E**(5/2) - 0.540*(10**11)*E*E*E + 0.951*(10**11)*E**(7/2)- 0.375*(10**11)*E*E*E*E),
        # d + t -> n + 4He
    	(26 - 0.361*E + 248*E*E)/(1+((E-0.0479)/0.0392)**2),
        # 3He + d -> 4He + p
    	(19.5 - 22.7*E + 61.8*E*E- 19.5*E*E*E + 4.05*E*E*E*E)/(1+((E-0.201)/0.132)**2),
        # 3He + 4He -> 7Be + gamma
	    0.107*10-2 + np.exp(-0.552*E)*(-0.582*10**(-3) - 0.606*10**(-3)*E - 0.154*10**(-3)*E*E),
        # 4He + t -> 7Li + gamma
    	(0.10- 0.15*E + 0.13*E*E)*10**(-6),
        # 7Be + n -> 7Li + p
    	(0.470*(10**10) - 0.202*(10**11)*E**(1/2) + 0.349*(10**11)*E - 0.253*(10**11)*E**(3/2)+ 0.660*(10**10)*E*E +0.109*(10**10)/(1+((E-0.317)/0.114)**2)),
        # 7Li + p -> 4He + 4He
    	(0.0609 + 0.173*E - 0.319*E*E + 0.217*E*E*E)
        ])[mask]/NA
