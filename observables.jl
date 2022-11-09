"""
one point (expectations) and two points (correlators) observables calculated on a MPS/MPDO
"""

using LinearAlgebra

"""
calculates single site expectation <psi|O_i|psi>
"""

function expect_single_site(i,psi,O,object::String)

    """
    i : site where expectation is calculated
    psi : input state in right canonical form
    O : operator of dimension (d,d)
    object : nature of psi, MPS or MPDO
    """
                
    if object == "MPS"

        siz = size(psi[i])[3]
    
        R = Matrix{Float64}(I,siz,siz)
    
        R = contract_right_nompo(psi[i],R,O,"MPS")
    
        @inbounds for i in reverse(1:i-1)
        
            R = contract_right_noop(psi[i],R,"MPS")
        end
        
        return real(R[1])
    
    elseif object == "MPDO"

        siz = size(psi[i])[4]
    
        R = Matrix{Float64}(I,siz,siz)
    
        R = contract_right_nompo(psi[i],R,O,"MPDO")
    
        @inbounds for i in reverse(1:i-1)
        
            R = contract_right_noop(psi[i],R,"MPDO")
        end
        
        return real(R[1])
    end
end

#Note : the inner product is calculated with this function by setting O as a (d,d) unit matrix, i.e. <psi|1|psi> = <psi||psi> 

"""
Calculates subsystem expectation, <psi|SUM_{i=1:l} O_i |psi>
"""

function expect_subsystem(N,l,psi,O,object::String)

    """
    N : system size
    l : subsystem size 
    psi : input state in right canonical form
    O : operator of dimension (d,d)
    object : nature of psi, MPS or MPDO
    """
    
    k = trunc(Int,N/2) - trunc(Int,l/2)

    expect_val = 0.0
    
    if object == "MPS"

        for i in k+1:k+l

            val = expect_single_site(i,psi,O,"MPS")

            expect_val += val
        end

        return expect_val/l

    elseif object == "MPDO"

            for i in k+1:k+l
    
                val = expect_single_site(i,psi,O,"MPDO")
    
                expect_val += val
            end
    
        return expect_val/l
    end
end

"""
Calculates equal time correlated function, <psi|O_i O_j|psi>
"""

function corr_func(j,k,psi,O,object::String)

    """
    j,k : sites with operators O
    psi : input state in right canonical form
    O : operator of dimension (d,d)
    object : nature of psi, MPS or MPDO
    """

    if object == "MPS"
    
        siz = size(psi[k])[3]
    
        R = Matrix{Float64}(I,siz,siz)
    
        R = contract_right_nompo(psi[k],R,O,"MPS")
    
        @inbounds for i in reverse(1:k-1)
        
            if i == j
                R = contract_right_nompo(psi[j],R,O,"MPS")
            else
                R = contract_right_noop(psi[i],R,"MPS")
            end
        end

        return R[1]

    elseif object == "MPDO"

        siz = size(psi[k])[4]
    
        R = Matrix{Float64}(I,siz,siz)
    
        R = contract_right_nompo(psi[k],R,O,"MPDO")
    
        @inbounds for i in reverse(1:k-1)
        
            if i == j
                R = contract_right_nompo(psi[j],R,O,"MPDO")
            else
                R = contract_right_noop(psi[i],R,"MPDO")
            end
        end

        return R[1]
    end         
end

"""
Calculates energy density of the given state (MPS or MPDO)
"""
function Energy_density(state,Ham,N,object::String)

    """
    state : input MPS or MPDO
    Ham : Hamiltonian as MPO
    N : system size
    object : type of state
    """

    X = ones(1,1,1)

    if object == "MPS"
        for i in 1:N
            X = contract_left(state[i],X,Ham[i],"MPS")
        end

        return reshape(X,1)[1]/N

    elseif object == "MPDO"
        for i in 1:N
            X = contract_left(state[i],X,Ham[i],"MPDO")
        end

        return reshape(X,1)[1]/N
    end        
end
