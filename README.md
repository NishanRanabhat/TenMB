# TenMB

This repository consists of collection of files for tensor network algorithms; DMRG and TDVP for finite MPS and iTEBD fo infinite MPS. 
It consists of all the supporting files like contraction routines, Lanczos routines, several 
spin states, spin Hamiltonians as MPO etc. The codes are free to be used and customized according 
to your necessity. Any suggestion on optimization on the existing code is highly welcome, please write to
nranabha@sissa.it .

Note : 

1) The bottleneck of every tensor network algorithms is tensor contraction. There are several options available in Julia, one including writing your own contraction code. In the codes presented here I employ TensorOperations.jl (see https://jutho.github.io/TensorOperations.jl/stable/) a highly efficient contraction package using a convenient Einstein index notation. 

2) The bottleneck of DMRG and TDVP algorithms is local eigensolver and exponential solver. In the codes presented here I employ a lanczos (for Hermitian Hamiltonians) based subroutine that performs eigen decompositions and exponentions with tensors as inputs.  