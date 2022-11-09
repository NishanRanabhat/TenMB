"""
This is a file that tests our codes. We will show how to run a TDVP code with pure states for transverse field Ising Hamiltonian.

Setup: In this simplest example we will initialize our state as an fully polarized (in longitudinal direction) MPS and evolve it with
       two site TDVP algorithm. 
"""

using .TenMB


#building Ising hamiltonian

#parameters
h = 0.5; #transverse field
N = 20; #system size
spins = "Pauli"; #Pauli spins
longdir = "X"; #longitudinal direction along X axis

Ham = TenMB.Hamiltonian_Ising(h,N,spins,longdir); #Ham is a MPO


#initialize fully polarized spin state in "up" direction
psi = TenMB.fully_pol_X(N,"up"); #psi is a MPS

#initialize before DMRG sweeps
M,psi,Env = TenMB.Initialize(N,psi,Ham,"MPS");

# M is the leftmost unnormalized tensor in MPS
# psi is the rest of MPS and is right canonical
# Env is the set of right environment tensors   

#DMRG sweeps
dt = 0.01; #trotterized time steps
krydim_TDVP = 14; #krylov dimension for TDVP algorithm
chi_max = 64; #maximum bond dimension
close_cutoff = 0.00000001 #cutoff value in Lanczos exponential solver
ctf_val = 0.00000001 #cutoff for SVD truncation
num_sweep_TDVP = 8; #number of TDVP sweeps

for i in 1:num_sweep_DMRG
    M,psi,Env = TenMB.right_sweep_TDVP_twosite(dt,M,psi,Env,Ham,N,krydim_TDVP,d,chi_max,close_cutoff,ctf_val);
    M,psi,Env = TenMB.left_sweep_TDVP_twosite(dt,M,psi,Env,Ham,N,krydim_TDVP,d,chi_max,close_cutoff,ctf_val);
end

"""
TDVP algorithm can be tested by calculating magnetization (see observables.jl file) after every sweep and see if the magnetization
shows expected evolution
"""