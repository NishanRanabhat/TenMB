"""
This is a file that tests our codes. We will show how to run a DMRG and TDVP code with pure states for transverse field Ising Hamiltonian.
"""

using .TenMB


#building Ising hamiltonian

#parameters
h = 0.5; #transverse field
N = 20; #system size
spins = "Pauli"; #Pauli spins
longdir = "X"; #longitudinal direction along X axis

Ham = TenMB.Hamiltonian_Ising(h,N,spins,longdir); #Ham is a MPO

