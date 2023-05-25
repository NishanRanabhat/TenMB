module TenMB

    """
    In this module we collect all the subsidiary functions that make the bulk of 
    this package. The current available features are:
    
    1) Variational ground state search of short and long range spin Hamiltonians through one and two site DMRG.

    2) Real and imaginary time evolution of a spin state through one and two site TDVP

    3) Finite temperature evolution of a mixed density matrix through TDVP based LPTN method.
    """

include("states.jl")
include("Hamiltonians.jl")
include("ContractionRoutines.jl")
include("LanczosSolver.jl")
include("SvdTruncate.jl")
include("Indexfix.jl")
include("DMRG.jl")
include("TDVP.jl")
include("initialize.jl")
include("MPSobservables.jl")
include("iMPSobservables.jl")


end

"""
Note: The list of libraries to download to run this module are:

1) LinearAlgebra : pretty much does all the heavy lifting along with TensorOperations.
2) TensorOperations : by Jutho Haegman is the library for all the tensor contractions.
3) Kronecker : used to construct some MPO/MPS .
"""
