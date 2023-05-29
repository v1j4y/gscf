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

numpoints=400
stoptime=15.0
rr = np.array([stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)])
qmax = 1.
xlim = qmax
ylim = 0
x1,urf,tckur = solve_SLP(fn=nrf, tckfn=tck, t=rr, xlim=xlim, ylim=ylim, vectorfield=vectorfieldVH)

path = "/home/chilkuri/Documents/codes/python/gscf/Figs/Fig-2.png"

plt.clf()
fig, ax = plt.subplots()

numpoints=3200
stoptime=60.0
rr = np.array([stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)])
qmax = 1.
xlim = qmax
ylim = 0.
x1,urf,tckur = solve_SLP(fn=nrf, tckfn=tck, t=rr, xlim=xlim, ylim=ylim, vectorfield=vectorfieldVH, isWF=False)

x1n = [urf(x,tck)  for x in rr]
plt.plot(rr, x1n)
plt.grid()
plt.xlabel("r")
plt.ylabel("U(r)")

plt.savefig(path)

# "path" variable must be set by block that
# expands this org source code block
"[["+path+"]]"

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

stoptime=60.0
numpoints=3200
rr = np.array([stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)])

# Get initial density
E_bound,_,nrf,tck = get_energy_and_density(0,rr,z=2.,E=-1.50,vectorfield=vectorfield)

# Get initial ur
qmax = 1.
xlim = qmax
ylim = 0.
x1,urf,tckur = solve_SLP(fn=nrf, tckfn=tck, t=rr, xlim=xlim, ylim=ylim, vectorfield=vectorfieldVH, isWF=False)
E0 = calcEnergy(E_bound, urf, tckur, nrf, tck)
print(E_bound, E0)

E_conv = []
dE_conv = []
E_conv.append(E0)
dE_conv.append(E0)
cnt = 0
Ediff = 10.
while cnt < 9 and abs(Ediff) > 1.E-4:

    # Get density
    E_bound,_,nrf,tck = get_energy_and_density(0,rr,z=2.,E=-1.50,vectorfield=vectorfield, urf=urf, tckur=tckur)
    # Get ur
    x1,urf,tckur = solve_SLP(fn=nrf, tckfn=tck, t=rr, xlim=xlim, ylim=ylim, vectorfield=vectorfieldVH, isWF=False)
    E1 = calcEnergy(E_bound, urf, tckur, nrf, tck,t=rr)
    #E1 = E_bound
    E_conv.append(E1)
    Ediff = abs(E0-E1)
    dE_conv.append(Ediff)
    print(f"Iter : {cnt} E = {E1} Diff = {Ediff} E_bound={E_bound}")
    E0 = E1

    cnt += 1

path = "/home/chilkuri/Documents/codes/python/gscf/Figs/Fig-tmp3.png"

plt.clf()
fig, ax = plt.subplots()

numpoints=3200
stoptime=60.0
rr = np.array([stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)])
#E_bound,_,nrf,tck = get_energy_and_density(0,rr,E=-0.40,vectorfield=vectorfield)
#E_bound,_,nrf,tck = get_energy_and_density(0,rr,E=-0.15,vectorfield=vectorfield)
#E_bound,_,nrf,tck = get_energy_and_density(0,rr,E=-0.05,vectorfield=vectorfield)

#E_bound,_,nrf,tck = get_energy_and_density(0,rr,E=-2.10,vectorfield=vectorfieldHe)
E_bound,_,nrf,tck = get_energy_and_density(0,rr,z=2.,E=-2.10,vectorfield=vectorfieldHe, urf=urf, tckur=tckur)
print(E_bound)

x1n = [nrf(x,tck) for x in rr]
plt.plot(rr, x1n)
plt.grid()
plt.xlabel("r")
plt.ylabel("u(0)")

plt.savefig(path)

# "path" variable must be set by block that
# expands this org source code block
"[["+path+"]]"

path = "/home/chilkuri/Documents/codes/python/gscf/Figs/Fig-4.png"

plt.clf()
fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0)

axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle('Helium atom ground state energy')

axs[0].plot(range(cnt+1), E_conv, marker='.')
axs[1].plot(range(cnt+1), dE_conv, marker=".", color='r')

lims = [ [-5.5,-2.5],[-5.5,3.2]]

# Hide x labels and tick labels for all but bottom plot.
for (i,ax) in enumerate(axs):
    ax.label_outer()
    ax.set_ylim(lims[i])

axs[0].axhline(y = -2.861, color = 'b', linestyle = '--')
axs[1].axhline(y = 0., color = 'r', linestyle = '--')
axs[0].set(ylabel="E(He) (u.a.)")
axs[1].set(ylabel="$\Delta E (u.a.)$")

plt.savefig(path)

# "path" variable must be set by block that
# expands this org source code block
"[["+path+"]]"

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

stoptime=60.0
numpoints=3200
rr = np.array([stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)])

# Get initial density
E_bound,_,nrf,tck = get_energy_and_density(0,rr,z=2.,E=-2.50,vectorfield=vectorfield)

# Get initial ur
qmax = 1.
xlim = qmax
ylim = 0.
x1,urf,tckur = solve_SLP(fn=nrf, tckfn=tck, t=rr, xlim=xlim, ylim=ylim, vectorfield=vectorfieldVH, isWF=False)
E0 = calcEnergy(E_bound, urf, tckur, nrf, tck)
print(E_bound, E0)

def urxf(x,nrf,tck):
    numer = 3.*nrf(x,tck)*nrf(x,tck)
    denom = 2.*np.pi*np.pi*x*x
    return(-np.power(numer/denom,1/3))

E_conv = []
dE_conv = []
E_conv.append(E0)
dE_conv.append(E0)
cnt = 0
Ediff = 10.
while cnt < 30 and abs(Ediff) > 1.E-4:

    # Get density
    E_bound,_,nrf,tck = get_energy_and_density(0,rr,z=2.,E=-1.00,vectorfield=vectorfieldX, urf=urf, tckur=tckur, fnx=urxf, tckfnx=tckur, fnc=nrf, tckfnc=tck)
    # Get ur
    x1,urf,tckur = solve_SLP(fn=nrf, tckfn=tck, t=rr, xlim=xlim, ylim=ylim, vectorfield=vectorfieldVH, isWF=False)
    E1 = calcEnergyVx(E_bound, urf, tckur, urxf, tckur, nrf, tck, t=rr)
    #E1 = E_bound
    E_conv.append(E1)
    Ediff = abs(E0-E1)
    dE_conv.append(Ediff)
    print(f"Iter : {cnt} E = {E1} Diff = {Ediff} E_bound={E_bound}")
    E0 = E1

    cnt += 1

path = "/home/chilkuri/Documents/codes/python/gscf/Fig-tmp5.png"

plt.clf()
fig, ax = plt.subplots()

numpoints=3200
stoptime=60.0
rr = np.array([stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)])
EE = np.linspace(-2.1, 0.1, 100)
u0s = [
    shoot(E, rr, l=0, z=2., vectorfield=vectorfieldX, fn=urf, tckfn=tckur, fnx=urxf, tckfnx=tckur)[0] for E in EE
]

plt.plot(EE, u0s)
plt.grid()
plt.xlabel("E")
plt.ylabel("u(0)")

plt.savefig(path)

# "path" variable must be set by block that
# expands this org source code block
"[["+path+"]]"

path = "/home/chilkuri/Documents/codes/python/gscf/Figs/Fig-tmp6.png"

plt.clf()
fig, ax = plt.subplots()

numpoints=3200
stoptime=60.0
rr = np.array([stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)])
#E_bound,_,nrf,tck = get_energy_and_density(0,rr,E=-0.40,vectorfield=vectorfield)
#E_bound,_,nrf,tck = get_energy_and_density(0,rr,E=-0.15,vectorfield=vectorfield)
#E_bound,_,nrf,tck = get_energy_and_density(0,rr,E=-0.05,vectorfield=vectorfield)

#E_bound,_,nrf,tck = get_energy_and_density(0,rr,E=-2.10,vectorfield=vectorfieldHe)
#E_bound,_,nrf,tck = get_energy_and_density(0,rr,z=2.,E=-2.00,vectorfield=vectorfieldX, urf=urf, tckur=tckur, fnx=urxf, tckfnx=tckur)
#print(E_bound)
xa  = [np.sqrt(4)*x*np.exp(-x) for x in rr]
x1n = [nrf_orig(x,tck_orig) for x in rr]
plt.plot(rr, x1n)
plt.plot(rr, xa)
plt.grid()
plt.xlabel("r")
plt.ylabel("u(0)")

plt.savefig(path)

# "path" variable must be set by block that
# expands this org source code block
"[["+path+"]]"

path = "/home/chilkuri/Documents/codes/python/gscf/Figs/Fig-6.png"

plt.clf()
fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0)

axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle('Helium atom ground state energy')

axs[0].plot(range(cnt+1), E_conv, marker='.')
axs[1].plot(range(cnt+1), dE_conv, marker=".", color='r')

lims = [ [-5.5,-0.5],[-5.5,4.2]]

# Hide x labels and tick labels for all but bottom plot.
for (i,ax) in enumerate(axs):
    ax.label_outer()
    ax.set_ylim(lims[i])

axs[0].axhline(y = -2.26 , color = 'g', linestyle = '--')
axs[0].axhline(y = -2.72 , color = 'g', linestyle = '--')
axs[0].axhline(y = -2.861, color = 'b', linestyle = '--')
axs[1].axhline(y = 0., color = 'r', linestyle = '--')
axs[0].set(ylabel="E(He) (u.a.)")
axs[1].set(ylabel="$\Delta E (u.a.)$")

plt.savefig(path)

# "path" variable must be set by block that
# expands this org source code block
"[["+path+"]]"
