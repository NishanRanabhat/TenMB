using LinearAlgebra
using PyCall

# Import NumPy via PyCall
np = pyimport("numpy")

"""
This is a file that tests our codes. We will show how to run a TDVP code with pure states for transverse field Ising Hamiltonian.

Setup: In this simplest example we will initialize our state as an fully polarized (in longitudinal direction) MPS and evolve it with
       two site TDVP algorithm. 
"""
#include the path to the module
include("/home/nishan/CodeDev/TenMB/Module/TenMB.jl")
#CALL module
using .TenMB


#building Ising hamiltonian

#parameters
J1 = 1; #transverse field
J2 = 1; #system size
N = 9
spins = "Pauli"; #Pauli spins

#Ham = TenMB.Hamiltonian_Ising(0.35,N,spins,"X")
Ham = TenMB.Hamiltonian_NNN_XXX(J1,J2,N,spins); #Ham is a MPO

#initialize neel spin state
psi = TenMB.neel_X(N); #psi is a MPS

#initialize before DMRG sweeps
M,psi,Env = TenMB.Initialize(N,psi,Ham,"MPS");

# M is the leftmost unnormalized tensor in MPS
# psi is the rest of MPS and is right canonical
# Env is the set of right environment tensors   

#DMRG sweeps
d = 2;
dt = 0.02; #trotterized time steps
krydim_TDVP = 14; #krylov dimension for TDVP algorithm
chi_max = 256; #maximum bond dimension
close_cutoff = 0.000000001 #cutoff value in Lanczos exponential solver
ctf_val = 0.000000001 #cutoff for SVD truncation
num_sweep_TDVP = 2000; #number of TDVP sweeps

function run_tdvp_sweeps(M, psi, Env, Ham, N, num_sweep_TDVP, dt, krydim_TDVP, d, chi_max, close_cutoff, ctf_val)
    data = Array{Any,1}(undef,2000)
    for i in 1:num_sweep_TDVP

        state = Array{Any,1}(undef,N)

        M, psi, Env = TenMB.right_sweep_TDVP_twosite(dt, M, psi, Env, Ham, N, krydim_TDVP, d, chi_max, close_cutoff, ctf_val)
        M, psi, Env = TenMB.left_sweep_TDVP_twosite(dt, M, psi, Env, Ham, N, krydim_TDVP, d, chi_max, close_cutoff, ctf_val)

        state[1] = M

        for j in 2:N
            state[j] = psi[j]
        end

        avg = TenMB.expect_single_site(5,state,[1 0; 0 -1],"MPS")
        data[i] = avg
        println(i)
        state = nothing
    end
    return data
end

sX = [0 1; 1 0];  sY = [0 -im; im 0];
sZ = [1 0; 0 -1]; sI = [1 0; 0 1];

# Call the function
avg_list = run_tdvp_sweeps(M, psi, Env, Ham, N, num_sweep_TDVP, dt, krydim_TDVP, d, chi_max, close_cutoff, ctf_val)
np.save("my_array_tdvp_X.npy", avg_list)


"""
TDVP algorithm can be tested by calculating magnetization (see observables.jl file) after every sweep and see if the magnetization
shows expected evolution
"""