"""
This file takes the initial state and initializes the system for DMRG and TDVP
sweeps. The outputs are the canonical MPS, and the Environment tensors.
"""
using LinearAlgebra
using TensorOperations


function right_normalize(psi,N)

"""
This function puts the input MPS into right canonical form starting from site N to
site 2. The site 1 is non- canonical as putting it in right canonical will result to
loss of information.
"""

    @inbounds for i in reverse(2:N)

        left_index = size(psi[i])[1]
        center_index = size(psi[i])[2]
        right_index = size(psi[i])[3]

        Mat = reshape(psi[i],(left_index,center_index*right_index))

        F = svd(Mat)

        psi[i] = reshape(F.Vt,(:,center_index,right_index))

        US = F.U*diagm(F.S)
        @tensoropt psi[i-1][-1,-2,-3] := psi[i-1][-1,-2,4]*US[4,-3]/norm(F.S)
    end

    return psi
end


function left_normalize(psi,N)

"""
This function accordingly puts the MPS in right canonical form.
"""
    @inbounds for i in 1:N-1

        left_index = size(psi[i])[1]
        center_index = size(psi[i])[2]
        right_index = size(psi[i])[3]

        Mat = reshape(psi[i],(left_index*center_index,right_index))

        F = svd(Mat)

        psi[i] = reshape(F.U,(left_index,center_index,:))

        SV = diagm(F.S)*F.Vt

        @tensor psi[i+1][-1,-2,-3] := SV[-1,4]*psi[i+1][4,-2,-3]/norm(F.S)
    end

    return psi
end


function Initialize(N,psi,Ham,object::String)
    
"""
This function initializes the state at time zero in right canonical form and the environment tensors.
Since a string of MPS cannot be put into complete left/right canonical form (i.e. all the individual
MPS are canonical) without losing some information, we have chosen to represent our state as an MPS with all
but the edge tensors ( "M" ) in the canonical form. 
"""
    
    if object == "MPS"

        psi = right_normalize(psi,N)

        M = psi[1]

        Env = Array{Any,1}(undef,N+1)
        Env[N+1] = ones(1,1,1)

        @inbounds for i in reverse(2:N)
            Env[i] = contract_right(psi[i],Env[i+1],Ham[i],"MPS")
        end
    
    elseif object == "MPDO"
        
        M = psi[1]

        Env = Array{Any,1}(undef,N+1)
        Env[N+1] = ones(1,1,1)

        @inbounds for i in reverse(2:N)
            Env[i] = contract_right(psi[i],Env[i+1],Ham[i],"MPDO")
        end
        
    end
    return M,psi,Env
end
