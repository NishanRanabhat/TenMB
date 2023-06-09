"""
This file consists function for Singular Value Decomposition, that has two
cutoff on the singular values:

1) ctf_val puts a cutoff on the minimum value a singular value can take. Singular
values smaller than ctf_val are discarded.

2) chi_max puts a cutoff on the maximum value of bond dimension i.e. the maximum number
of the Singular values.

The error created by these cutoffs is known as cutoff error.
"""

using LinearAlgebra

function truncate(S::Array{},ctf_val::Float64)
    
    c = 0
    
    for i in 1:length(S)
        
        if S[i] > ctf_val
            
            c += 1 
        else
            
            break
        end
    end
    
    return c
end

function svd_truncate(T::Array{},chi_max::Int64,ctf_val::Float64)

    """
    This function performs singular value decomposition on the vector "T" followed
     by a cutoff either based on "cut_val" or "chi_max"

    The svd decomposition is performed with QRiteration algorithm instead of
    Divide and Conquer algorithm. This is more expensive but robust.
    """

    F = svd(T,alg=LinearAlgebra.QRIteration())

    S = F.S/norm(F.S)

    c = truncate(S,ctf_val)
    
    chi_trunc = min(c,chi_max)
    
    S = S[1:chi_trunc]
    U = F.U[:,1:chi_trunc]
    V = F.Vt[1:chi_trunc,:]
    
    return U,S,V
end
