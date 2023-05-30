import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('/home/runner/work/gscf/gscf/src')
sys.path.append('/home/runner/work/gscf/gscf')

import numpy as np

from src.hydrogen import get_energy_and_density, vectorfield

def test_hydrogen(numpoints=3200, stoptime=60):
    rr = np.array([stoptime * float(i+0.0001) / (numpoints - 1) for i in range(numpoints)])
    E_bound,_,_,_ = get_energy_and_density(0,rr,vectorfield=vectorfield)
    abs(E_bound - 0.5) <= 1.0E-10
