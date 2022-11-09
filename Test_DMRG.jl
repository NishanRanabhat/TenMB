"""
This is a file that tests our codes. We will show how to run a DMRG code with pure states for transverse field Ising Hamiltonian.
We will show the simplest case for one site DMRG algorithm.
"""

using .TenMB


#building Ising hamiltonian

#parameters
h = 0.5; #transverse field
N = 20; #system size
spins = "Pauli"; #Pauli spins
longdir = "X"; #longitudinal direction along X axis

Ham = TenMB.Hamiltonian_Ising(h,N,spins,longdir); #Ham is a MPO


#initialize random state
chi = 16; #bond dimension
d = 2; #local dimension

psi = TenMB.random_psi(N,chi,d); #psi is a MPS

#initialize before DMRG sweeps
M,psi,Env = TenMB.Initialize(N,psi,Ham,"MPS");

# M is the leftmost unnormalized tensor in MPS
# psi is the rest of MPS and is right canonical
# Env is the set of right environment tensors   

#DMRG sweeps
krydim_DMRG = 4; #krylov dimension for DMRG sweeps
maxit = 10; #iteration for eigensolver 
num_sweep_DMRG = 8; #number of DMRG sweeps

for i in 1:num_sweep_DMRG
    M,psi,Env = TenMB.right_sweep_DMRG_one_site(M,psi,Env,Ham,N,krydim_DMRG,maxit,d);
    M,psi,Env = TenMB.left_sweep_DMRG_one_site(M,psi,Env,Ham,N,krydim_DMRG,maxit,d);
end

"""
DMRG algorithm can be tested by calculating energy density (see observables.jl file) after every sweep and
see if it converges to the expected value.
"""