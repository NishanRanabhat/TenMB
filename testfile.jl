"""
This is a file that tests our codes, we call the main module as using .TenMB .
"""

using .TenMB


#building Ising hamiltonian
h = 0.5; #transverse field
N = 20; #system size
spins = "Pauli"; #Pauli spins
longdir = "X"; #longitudinar direction along X axis

Ham = TenMB.Hamiltonian_Ising(h,N,spins,longdir); #Ham is a MPO

