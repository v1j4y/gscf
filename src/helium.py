import numpy as np
from scipy.integrate import odeint
from scipy import integrate
from scipy import interpolate
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from copy import deepcopy

def vectorfieldVH(w, t, p):
    """
    Defines the differential equations for the coupled spring-mass system.

    Arguments:
    w :  vector of the state variables:
              w = [x1,y1,x2,y2]
    t :  time
    p :  vector of the parameters:
              p = [m1,m2,k1,k2,L1,L2,b1,b2]
    """
    x1, y1 = w
    _, _, nrf, tck,_,_,_,_, z = p

    # Create f = (x1',y1'):
    f = [y1,
         -nrf(t,tck)*nrf(t,tck)/t
         ]
    return f

def vectorfieldHe(w, t, p):
    """
    Defines the differential equations for the coupled spring-mass system.

    Arguments:
    w :  vector of the state variables:
              w = [x1,y1,x2,y2]
    t :  time
    p :  vector of the parameters:
              p = [m1,m2,k1,k2,L1,L2,b1,b2]
    """
    x1, y1 = w
    e1, l1, urf, tckur, z = p

    # Create f = (x1',y1'):
    f = [y1,
         2*(l1*(l1+1)/(t)**2 - z/t - e1 + urf(t,tckur)/t)*x1
         ]

    return f

def calcEnergy(ei,urf,tckur,nrf,tck,t=None,stoptime=60.0,numpoints=3200):
    E = 2*ei
    if t is None:
        t = [stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)]
    h = t[1]-t[0]
    VHl = np.array([urf(x,tckur)/x for x in t])
    Nr2 = np.array([(nrf(x,tck))**2 for x in t])
    eH = integrate.simps(VHl*Nr2, x=t)
    print(eH)
    E = E - eH
    return(E)

def vectorfieldX(w, t, p):
    """
    Defines the differential equations for the coupled spring-mass system.

    Arguments:
    w :  vector of the state variables:
              w = [x1,y1,x2,y2]
    t :  time
    p :  vector of the parameters:
              p = [m1,m2,k1,k2,L1,L2,b1,b2]
    """
    x1, y1 = w
    e1, l1, urf, tckur, uxrf, tckurx, nrf, tck, z = p

    # Create f = (x1',y1'):
    f = [y1,
         2*(l1*(l1+1)/(t)**2 - z/(t) - e1 + 2*urf(t,tckur)/t + uxrf(t,nrf,tck) )*x1
         ]

    return f

def calcEnergyVx(ei,urf,tckur,urxf,tckurx,nrf,tck,t=None,stoptime=60.0,numpoints=3200):
    E = 2*ei
    if t is None:
        t = [stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)]
    VHl = np.array([urf(x,tckur)/x for x in t])
    Vxl = np.array([urxf(x,nrf,tck) for x in t])
    ur2 = np.array([(nrf(x,tck))**2 for x in t])
    eH = integrate.simps(VHl*ur2, x=t)
    ex = integrate.simps(Vxl*ur2, x=t)
    print((eH, (ex/2)))
    E = E - eH + (ex/2)
    return(E)

def urxf(x,nrf,tck):
    numer = 3.*nrf(x,tck)*nrf(x,tck)
    denom = 2.*np.pi*np.pi*x*x
    return(-np.power(numer/denom,1/3))

