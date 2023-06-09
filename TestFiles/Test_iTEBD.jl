using LinearAlgebra
using Kronecker


"""
This is a file that tests our codes. We will show how to run an 
iTEBD code with pure states for transverse field Ising Hamiltonian.
"""

#include the path to the module
include("/path/to/module/folder/TenMB.jl")

#CALL module
using .TenMB


#Build two site hamiltonian operator: here the transverse field Ising model with transverse direction along X
sX = [0 1; 1 0]; sY = [0 -im; im 0];
sZ = [1 0; 0 -1]; sI = [1 0; 0 1];

h = 0.3; #transverse field
operator = -kron(sZ,sZ)+ h*kron(sX,sI);

#Build list of trotter operators: here we are using first order Trotter approximation and real unitary evolution
dt = 0.01; #Trotter time steps
order = "first"; #Trotter order
U = TenMB.operator_list(operator=operator,order="first",evol_type="real",dt=dt);


#Initialize the state of type iMPS: here we are initializing fully polarized along Z direction
inf_MPS = TenMB.iMPS(Matrix{Float64}(I,1,1),reshape([1 0],(1,2,1)),reshape([1 0],(1,2,1)));

#Simulation parameters
chi_max = 100; #maximum bond dimension
ctf_val = 10^(-8); #cutoff value for singular values
num_iTEBD = 500; 



#Run iTEBD simulation
for i in 1:num_iTEBD
    
    inf_MPS = TenMB.itebd(inf_MPS,U,chi_max,ctf_val)
end

