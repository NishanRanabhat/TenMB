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

function cutoff_sin_val(S,ctf_val::Number)

        """
        Cuts off a vector (with elements in descending order) "S" based on a
        cutoff value "ctf_val". Returns the index at which the vector is cut off.
        """

    i = 1
    while S[i] > ctf_val

        if i < length(S)
            i += 1
        else
            break
        end
    end
    return i
end

function svd_truncate(T,chi_max::Integer,ctf_val::Number)

        """
        This function performs singular value decomposition on the vector "T" followed
        by a cutoff either based on "cut_val" or "chi_max"

        The svd decomposition is performed with QRiteration algorithm instead of
        Divide and Conquer algorithm. This is more expensive but solid.
        """

    F = svd(T,alg=LinearAlgebra.QRIteration())

    S = F.S

    k = cutoff_sin_val(S,ctf_val)

    if k < chi_max

        S = S[1:k]/norm(S[1:k])
        U = F.U[:,1:k]
        V = F.Vt[1:k,:]

    else

        S = S[1:chi_max]/norm(S[1:chi_max])
        U = F.U[:,1:chi_max]
        V = F.Vt[1:chi_max,:]
    end

    return U,S,V
end
