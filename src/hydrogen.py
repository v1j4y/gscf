from imports_all import *

def vectorfield(w, t, p):
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
    e1, l1, urf, tckur, _, _, _, _, z = p

    # Create f = (x1',y1'):
    f = [y1,
         2*(l1*(l1+1)/(t)**2 - z/(t) - e1 + urf(t,tckur)/t)*x1
         ]

    return f

def solve_SLP(fn=None, tckfn=None, fnx=None, tckfnx=None, fnc=None, tckfnc=None, e1=-0.5, l1=0, z=1., t=None, numpoints=1600, stoptime=15.0, xlim=0, ylim=-1.0E-6, vectorfield=None, isWF=True):

    if fn is None:
        def fn(x,tckfn):
            return(0.)

    if fnx is None:
        def fnx(x,tckfnx):
            return(0.)

    if fnc is None:
        def fnc(x,tckfnc):
            return(0.)

    if vectorfield is None:
        print("[solve_SLP] Error: Have to supply a vectorfield")
        return(0,0,0)

    # Parameter values
    # Initial conditions
    # x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
    x1 = xlim
    y1 = ylim

    # ODE solver parameters
    abserr = 1.0e-8
    relerr = 1.0e-6

    # Create the time samples for the output of the ODE solver.
    # I use a large number of points, only because I want to make
    # a plot of the solution that looks nice.
    if t is None:
        t = [stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)]

    # Reverse the list to converge from the right
    t_rev = t[::-1]

    # Pack up the parameters and initial conditions:
    p = [e1, l1, fn, tckfn, fnx, tckfnx, fnc, tckfnc, z]
    w0 = [x1, y1]

    # Call the ODE solver.
    wsol = odeint(vectorfield, w0, t_rev, args=(p,),
                  atol=abserr, rtol=relerr)

    x1 = wsol[:,0]

    # Reverse the result back
    x1 = x1[::-1]

    if isWF:
        # Normalize wavefunction
        norm = integrate.simps(x1**2, x=t)
        x1 = x1/np.sqrt(norm)

    tckfnout = interpolate.splrep(t,x1)

    def fnout(x, tck):
        return interpolate.splev(x, tckfnout)
    return(x1,fnout,tckfnout)

def shoot(E, t, l=0, z=1., fn=None, tckfn=None, fnx=None, tckfnx=None, fnc=None, tckfnc=None, xlim=0, ylim=-1.E-6, vectorfield=None, isWF=True):
   if vectorfield is None:
      print("[shoot] Error: Have to supply a vectorfield")
      return(0,0,0,0)
   u,fnout,tckfnout= solve_SLP(fn=fn, tckfn=None, fnx=fnx, tckfnx=tckfnx, fnc=fnc, tckfnc=tckfnc, e1=E, l1=l, z=z, t=t, xlim=xlim, ylim=ylim, vectorfield=vectorfield, isWF=isWF)
   u = u/t**l

   # Extrapolate u to the origin r=0.
   return u[0] - t[0] * (u[1] - u[0])/(t[1] - t[0]), u, fnout, tckfnout

rr = np.logspace(-6, 5, 500)
numpoints=400
stoptime=15.0
rr = np.array([stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)])
EE = [-1.1]
u0s = [
    shoot(EE[0], rr, l=0, vectorfield=vectorfield)[0] for E in EE
]

path = "/home/chilkuri/Documents/codes/python/gscf/Fig-tmp.png"

plt.clf()
fig, ax = plt.subplots()

rr = np.array([stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)])
EE = [-0.5]
u0s,u,_,_ = shoot(EE[0], rr, l=0, vectorfield=vectorfield, isWF=True)

plt.plot(rr,u)
plt.grid()
#ax.set_yscale('log')
plt.xlabel("r")
plt.ylabel("u(r)")

plt.savefig(path)

# "path" variable must be set by block that
# expands this org source code block
"[["+path+"]]"

# "path" variable must be set by block that
# expands this org source code block
"[["+path+"]]"

path = "/home/chilkuri/Documents/codes/python/gscf/Fig-1.png"

plt.clf()
fig, ax = plt.subplots()

numpoints=3200
stoptime=60.0
rr = np.array([stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)])
EE = np.linspace(-1.0, 0.1, 100)
u0s = [
    shoot(E, rr, l=0, vectorfield=vectorfield)[0] for E in EE
]

plt.plot(EE, u0s)
plt.grid()
plt.xlabel("E")
plt.ylabel("u(0)")

plt.savefig(path)

# "path" variable must be set by block that
# expands this org source code block
"[["+path+"]]"

def get_energy_and_density(l,rr,z=1.,E=None, vectorfield=None, urf=None, tckur=None, fnx=None, tckfnx=None, fnc=None, tckfnc=None, xlim=0., ylim=-1.0E-6, isWF=True):
    dE = 0.51 # scan resolution to look for sign changes
    if E is None:
        E = -1.0 # starting energy

    if vectorfield is None:
        print("[get_energy_and_density] Error have to supply a vectorfield")
        return(0)

    if urf is None:
        def urf(x,tckur):
            return(0)

    def fn(e):
        u0s = shoot(e, rr, l=l, z=z, fn=urf, tckfn=tckur, fnx=fnx, tckfnx=tckfnx, fnc=fnc, tckfnc=tckfnc, vectorfield=vectorfield, xlim=xlim, ylim=ylim, isWF=isWF)[0]
        return(u0s)
    E_bound = root_scalar(fn, x0=E-dE, x1=E+dE).root
    _,u_bound,nrf,tck = shoot(E_bound, rr, l=l, z=z, fn=urf, fnx=fnx, tckfnx=tckfnx, fnc=fnc, tckfnc=tckfnc, tckfn=tckur, vectorfield=vectorfield, xlim=xlim, ylim=ylim, isWF=isWF)
    return(E_bound, u_bound, nrf, tck)

path = "/home/chilkuri/Documents/codes/python/gscf/Figs/Fig-1.png"

plt.clf()
fig, ax = plt.subplots()

numpoints=3200
stoptime=60.0
rr = np.array([stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)])
E_bound,_,nrf,tck = get_energy_and_density(0,rr,E=-1.00,vectorfield=vectorfield)
nrf_orig=deepcopy(nrf)
tck_orig=deepcopy(tck)
#E_bound,_,nrf,tck = get_energy_and_density(0,rr,E=-0.15,vectorfield=vectorfield)
#E_bound,_,nrf,tck = get_energy_and_density(0,rr,E=-0.05,vectorfield=vectorfield)

x1n = [nrf(x,tck) for x in rr]
plt.plot(rr, x1n)
plt.grid()
plt.xlabel("r")
plt.ylabel("u(0)")

plt.savefig(path)

# "path" variable must be set by block that
# expands this org source code block
"[["+path+"]]"
