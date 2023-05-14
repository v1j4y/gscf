
# Table of Contents

1.  [Introduction](#orgf98a791)
2.  [Hydrogen atom](#org562d740)
    1.  [Methodology](#org434f8cb)
        1.  [Shooting method](#org38aea0a)
        2.  [Plots](#org8e90f31)
        3.  [Solution of the SLP](#org7b20f0e)
3.  [Helium atom](#org0525f4d)
    1.  [Poisson equation](#org489c15c)
    2.  [Solution](#org98fc57f)
        1.  [Vector Field](#orga38c6dd)
        2.  [ODE Solution](#org52d8b5b)
        3.  [Testing](#org53b4ba4)
        4.  [Main](#org00b046c)
    3.  [Self-consistent field cycle](#org1413d3f)
        1.  [Hydrogen functions with a potential](#org2938649)
        2.  [Testing](#orgf6d2354)
        3.  [Main](#org6a7b497)
        4.  [Calculate energy](#org2aa7223)
        5.  [SCF cycle code](#org7f6fdb2)
    4.  [Figure](#org4976447)


    import numpy as np
    from scipy.integrate import odeint
    from scipy import integrate
    from scipy import interpolate
    from scipy.optimize import root_scalar
    import matplotlib.pyplot as plt


<a id="orgf98a791"></a>

# Introduction

The Schrodinger equation is a certain class of Boundary Value Problem (BVP).
Solution of the Schrodinger equation results in the eigenvalues($\lambda$) and
eigenvectors ($\psi$) of the Hamiltonian.

$$
\mathcal{L}\psi(x) = \lambda\psi(x)
$$

These type of equations where $\mathcal{L}$ is a second order differential
operator and self-adjoint (i.e. hermitian) is called a Sturm-Liouville problem (SLP).
Sturm-Liouville problems are general eigenvalues problems of the form

$$
-\frac{d}{dt}\left ( p(t) \frac{d y(t)}{dt} \right ) + q(t)y(t) = \lambda g(t)y(t)
$$

posed on an interval $-\infty \le a \le b \le \infty$ which may be finite or infinite. Such
problem&rsquo;s arise in various fields of physics and chemistry and there exist
robust and well studied numerical and analytical methods for their resolution.

In the present manuscript, we shall document numerical methods for the solution
of various types of Hamiltonians and analyze their spectra and wavefunctions.


<a id="org562d740"></a>

# Hydrogen atom

We begin with the most simplest SLP which is the solution
of the Hamiltonian for the hydrogen atom Eq:[8](#org6d2ff4d)

$$
\hat{H}\psi(\mathbf{x}) = \left [ -\Laplace + V \right ]\psi(\mathbf{x})
$$

This is the SLP(Eq:[4](#orgb591c3c)) with $p(t)=g(t)=1$, and $q(t)=V$, the potential acting
on the particle. The Eq:[8](#org6d2ff4d) is in cartesian coordinates $\mathbf{x}$ and
can be transformed to spherical coordinates via a coordinate transformation.

\begin{equation}
\label{org2b5f043}
\begin{align*}
x_1 &= r\sin{\theta}\cos{\phi},\\
x_2 &= r\sin{\theta}\sin{\phi},\\
x_3 &= r\cos{\theta}
\end{align*}
\end{equation}

In spherical coordinates, the operator $\hat{H}$ is transformed to

\begin{equation}
\label{orgd6d611b}
\begin{align*}
\hat{H} &= -\frac{1}{r^2}\frac{\partial}{\partial r} \left( r^2 \frac{\partial}{\partial r} \right) \\
&  -\frac{1}{r^2}\frac{1}{\sin{\theta}}\frac{\partial}{\partial\theta} \left(\sin{\theta}\frac{\partial}{\partial\theta} \right)\\
&  -\frac{1}{r^2}\frac{1}{\sin{\theta}^2}\frac{\partial^2}{\partial\phi^2} + V
\end{align*}
\end{equation}

We can then separate the wavefunction to three independent variables
$\psi(\mathbf{x})=R(r)\Theta(\theta)\Phi(\phi)$ to obtain three separate SLPs

\begin{equation}
\label{orgeb075cd}
\begin{align*}
\left (
-\frac{1}{r^2}\frac{\partial}{\partial r} \left( r^2 \frac{\partial}{\partial r} \right)
+ \frac{l(l+1)}{r^2} + V(r)
 \right)R(r) &= \lambda R(r)\\
\frac{1}{\sin{\theta}}\left (
-\frac{\partial}{\partial \theta} \left( \sin{\theta} \frac{\partial}{\partial \theta} \right)
+ \frac{m^2}{\sin{\theta}}
 \right)\Theta(\theta) &= l(l+1) \Theta(\theta)\\
-\frac{\partial^2}{\partial \phi^2}\Phi(\phi) &= m^2 \Phi(\phi)\\
\end{align*}
\end{equation}

With only $r$ being the dependent variable i.e. the first equation.
Here we can do a further transformation of the dependent variable
$u(r) = r R(r)$ which gives the SLP

\begin{equation}
\label{orgc5a21fc}
\begin{align*}
-\frac{\partial^2 u(r)}{\partial r^2}
+ q(r) u(r) &= \lambda u(r) \\
q(r) &= \frac{l(l+1)}{r^2} + V(r)\\
p(r) &= g(r) = 1
\end{align*}
\end{equation}

These (Eq:[4](#orgc5a21fc)) are the working equations.


<a id="org434f8cb"></a>

## Methodology

First, we transfrom Eq:[4](#orgc5a21fc) into a set of coupled linear
ODEs

\begin{equation}
\label{orgbd68754}
\begin{align*}
y &= \begin{pmatrix} u \\ u' \end{pmatrix}\\
y' &= \begin{pmatrix} u' \\ u'' \end{pmatrix} = \begin{pmatrix} u' \\ \left( \frac{l(l+1)}{r^2} -\frac{1}{r} - E \right) u \end{pmatrix}\\
\end{align*}
\end{equation}

The boundary conditions are at $r=0$ and $r=\infty$ with
$u(r)=0$ and $u(\infty)=0$.

1.  Vector Field

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
            e1, l1 = p
        
            # Create f = (x1',y1'):
            f = [y1,
                 (l1*(l1+1)/t**2 - 2./t - 2.*e1)*x1
                 ]
            return f

2.  Solution

        def solve_SLP(e1=-0.5, l1=0, t=None, numpoints=1600, stoptime=15.0):
            # Parameter values
        
            # Initial conditions
            # x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
            x1 = 0
            y1 =-1.0E-6
        
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
            p = [e1, l1]
            w0 = [x1, y1]
        
            # Call the ODE solver.
            wsol = odeint(vectorfield, w0, t_rev, args=(p,),
                          atol=abserr, rtol=relerr)
        
            x1 = wsol[:,0]
            # Reverse the result back
            x1 = x1[::-1]
        
            # Normalize
            norm = integrate.simps(x1**2, x=t)
            x1 = x1/np.sqrt(norm)
        
            tck = interpolate.splrep(t,x1)
        
            def nrf(x, tck):
                return interpolate.splev(x, tck)
            return(x1,nrf,tck)


<a id="org38aea0a"></a>

### Shooting method

Here we start with $u(\infty)=0$ and integrate towards
$r=0$. This is more stable for the convergence with
respect to the Hydrogen atom.

1.  Code

        def shoot(E, t, l=0):
           u,nrf,tck= solve_SLP(e1=E, l1=l, t=t)
           u = u/t**l
        
           # Extrapolate u to the origin r=0.
           return u[0] - t[0] * (u[1] - u[0])/(t[1] - t[0]), u, nrf, tck

2.  Testing

        rr = np.logspace(-6, 5, 500)
        numpoints=400
        stoptime=15.0
        rr = np.array([stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)])
        EE = [-1.1]
        u0s = [
            shoot(EE[0], 0, rr)[0] for E in EE
        ]

3.  Plot


<a id="org8e90f31"></a>

### Plots

1.  Plotting stuff

        
        # "path" variable must be set by block that
        # expands this org source code block
        "[["+path+"]]"

2.  Main

    ![img](/home/chilkuri/Documents/codes/python/gscf/Fig-1.png)


<a id="org7b20f0e"></a>

### Solution of the SLP

Here we have to search for the value of $E$
for which the BVP has the final conditions satisfied
i.e. $u(r)=0$. This is done using the optimization
routine from `scipy`.

1.  Code

        def get_energy_and_density(l,rr,E=None):
            dE = 0.01 # scan resolution to look for sign changes
            if E is None:
                E = -1.0 # starting energy
        
            def fn(e):
                u0s = shoot(e, rr, l=l)[0]
                return(u0s)
            E_bound = root_scalar(fn, x0=E-dE, x1=E).root
            _,u_bound,nrf,tck = shoot(E_bound, rr, l=l)
            return(E_bound, u_bound, nrf, tck)

2.  Testing

        numpoints=400
        stoptime=15.0
        rr = np.array([stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)])
        E_bound,_,_,_ = get_energy_and_density(0,rr)

3.  Main

    ![img](/home/chilkuri/Documents/codes/python/gscf/Figs/Fig-1.png)


<a id="org0525f4d"></a>

# Helium atom

Here we need to include the Hartree potential $V_H$ which is the
repulsion between the two electrons

\begin{equation}
\label{org9a43e7b}
V_H(\mathbf{r}) = \int dr'^3 n(\mathbf{r}')\frac{1}{\mathbf{r}-\mathbf{r}'}
\end{equation}

Where the $n(\mathbf{r})$ is the density which is given as

$$
n(\mathbf{r}) = 2\sum_i^{N_{occ}} |\psi(\mathbf{r})|^2
$$

where we assume a close shell spin singlet slater determinant.


<a id="org489c15c"></a>

## Poisson equation

In order to calculate the Hartree potential Eq:[6](#org9a43e7b), we shall
transform it into an SLP which we can again solve using the
above methodology the solution of the Hydrogen atom.

\begin{equation}
\label{org609bb1c}
\nabla^2 V_H(\mathbf{r}) = -4 \pi n(\mathbf{r})
\end{equation}

This can again be transformed using the variable substitution
$u(r)=rR(r)$ to a 1D equation.

\begin{equation}
\label{orgccf0374}
\frac{\partial^2 U(r)}{\partial r} = -4\pi r n(r)
\end{equation}

The fact that $n(r)$ is simply $R(r)^2$ by definition and the
fact that $u(r)$ is normalized we can drop off $4\pi$ to finally
obtain

\begin{equation}
\label{org48fa30c}
U''(r) = -\frac{u(r)^2}{r}
\end{equation}

This is the SLP that we need to solve to obtain the
hartree potential $V_H(r)$.


<a id="org98fc57f"></a>

## Solution

The BVP Eq:[9](#org48fa30c) takes the following boundary conditions

\begin{equation}
\begin{align*}
U(0) &= 0\\
U(r_{max}) &= q_{max}
\end{align*}
\end{equation}

where, $q_{max}$ is the total charge. We shall use these conditions
in the shooting method to find the correct Hartree potential.

$$
q_{max} = \int_0^{max} \text{d}r\ u^2(r)
$$


<a id="orga38c6dd"></a>

### Vector Field

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
        nrf, tck = p
    
        # Create f = (x1',y1'):
        f = [y1,
             -nrf(t,tck)*nrf(t,tck)/t
             ]
        return f


<a id="org52d8b5b"></a>

### ODE Solution

    def solve_SLP_VH(nrf, tck, t=None, numpoints=1600, stoptime=15.0, qmax=1):
        # Parameter values
    
        # Initial conditions
        # x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
        x1 = qmax
        y1 =-1.0E-6
    
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
        p = [nrf, tck]
        w0 = [x1, y1]
    
        # Call the ODE solver.
        wsol = odeint(vectorfieldVH, w0, t_rev, args=(p,),
                      atol=abserr, rtol=relerr)
    
        x1 = wsol[:,0]
        # Reverse the result back
        x1 = x1[::-1]
    
        # Normalize
        norm = integrate.simps(x1**2, x=t)
        x1 = x1/np.sqrt(norm)
    
        tckur = interpolate.splrep(t,x1)
    
        def urf(x, tck):
            return interpolate.splev(x, tckur)
        return(x1,urf,tckur)


<a id="org53b4ba4"></a>

### Testing

    numpoints=400
    stoptime=15.0
    rr = np.array([stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)])
    __,urf,tckur = solve_SLP_VH(nrf, tck, t=rr)


<a id="org00b046c"></a>

### Main


<a id="org1413d3f"></a>

## Self-consistent field cycle

In order to find the solution, we need to perform a SCF loop
so that the energy stays constant.

In order to calculate the total energy, we now also need to
incorporate the Hartee potential

\begin{equation}
\label{orgb6de203}
E = 2 \epsilon - \int \text{d}r\ V_H(r) u^2(r)
\end{equation}


<a id="org2938649"></a>

### Hydrogen functions with a potential

1.  Vector Field

        def vectorfieldwithVH(w, t, p):
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
            e1, l1, urf, tckur = p
        
            # Create f = (x1',y1'):
            f = [y1,
                 (l1*(l1+1)/t**2 - 2./t - 2.*e1 + urf(t,tckur)/t)*x1
                 ]
            return f

2.  Solution

        def solve_SLP_withVH(urf, tckur, e1=-0.5, l1=0, t=None, numpoints=1600, stoptime=15.0):
            # Parameter values
        
            # Initial conditions
            # x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
            x1 = 0
            y1 =-1.0E-6
        
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
            p = [e1, l1, urf, tckur]
            w0 = [x1, y1]
        
            # Call the ODE solver.
            wsol = odeint(vectorfieldwithVH, w0, t_rev, args=(p,),
                          atol=abserr, rtol=relerr)
        
            x1 = wsol[:,0]
            # Reverse the result back
            x1 = x1[::-1]
        
            # Normalize
            norm = integrate.simps(x1**2, x=t)
            x1 = x1/np.sqrt(norm)
        
            tck = interpolate.splrep(t,x1)
        
            def nrf(x, tck):
                return interpolate.splev(x, tck)
            return(x1,nrf,tck)

3.  Shooting Code

        def shoot_withVH(E, t, urf, tckur, l=0):
           u,nrf,tck= solve_SLP_withVH(urf, tckur, e1=E, l1=l, t=t)
           u = u/t**l
        
           # Extrapolate u to the origin r=0.
           return u[0] - t[0] * (u[1] - u[0])/(t[1] - t[0]), u, nrf, tck

4.  Final solution

        def get_energy_and_density_withVH(l,rr,urf,tckur,E=None):
            dE = 0.01 # scan resolution to look for sign changes
            if E is None:
                E = -1.0 # starting energy
        
            def fn(e):
                u0s = shoot_withVH(e, rr, urf, tckur, l=l)[0]
                return(u0s)
            E_bound = root_scalar(fn, x0=E-dE, x1=E).root
            _,u_bound,nrf,tck = shoot_withVH(E_bound, rr, urf, tckur, l=l)
            return(E_bound, u_bound, nrf, tck)


<a id="orgf6d2354"></a>

### Testing

    numpoints=400
    stoptime=15.0
    rr = np.array([stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)])
    _,_,nrf1,tck1 = get_energy_and_density_withVH(0, rr, urf, tckur, E=-0.5)


<a id="org6a7b497"></a>

### Main


<a id="org2aa7223"></a>

### Calculate energy

    def calcEnergy(ei,urf,tckur,nrf,tck,t=None,stoptime=60.0,numpoints=3200):
        E = 2*ei
        if t is None:
            t = [stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)]
        h = t[1]-t[0]
        VHl = np.array([urf(x,tckur)/x for x in t])
        Nr2 = np.array([(nrf(x,tck))**2 for x in t])
        eH = integrate.simps(VHl*Nr2, x=t)
        E = E - eH
        return(E)


<a id="org7f6fdb2"></a>

### SCF cycle code

    
    stoptime=60.0
    numpoints=3200
    rr = np.array([stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)])
    
    # Get initial density
    E_bound,_,nrf,tck = get_energy_and_density(0,rr,E=-0.40)
    
    # Get initial ur
    x1,urf,tckur = solve_SLP_VH(nrf, tck, t=rr)
    E0 = calcEnergy(E_bound, urf, tckur, nrf, tck)
    E0 = E_bound
    
    E_conv = []
    dE_conv = []
    E_conv.append(2*E0)
    dE_conv.append(2*E0)
    cnt = 0
    while cnt < 8:
    
        # Get density
        E_bound,x1,nrf,tck = get_energy_and_density_withVH(0, rr, urf, tckur, E=-0.5)
        # Get ur
        x1,urf,tckur = solve_SLP_VH(nrf, tck, t=rr)
        E1 = calcEnergy(E_bound, urf, tckur, nrf, tck,t=rr)
        E1 = E_bound
        E_conv.append(2*E1)
        Ediff = abs(E0-E1)
        dE_conv.append(Ediff)
        print(f"Iter : {cnt} E = {E1} Diff = {Ediff} E_bound={E_bound}")
        E0 = E1
    
        cnt += 1


<a id="org4976447"></a>

## Figure

