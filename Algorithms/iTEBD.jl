using LinearAlgebra
using TensorOperations
using ExponentialUtilities
using Kronecker
import Base.@kwdef
using KeywordDispatch
using Parameters

function operator_list(d = 2;operator::Matrix{},order::String,evol_type::String,dt::Float64)
        
    """
    d : local dimension, 2 for spin 1/2 system
    operator : two site operator 
    order : order of trotterization
    evol_type : real or imaginary
    """

    if evol_type == "real"

        coeff = im

    elseif evol_type = "imaginary"

        coeff = 1.0
    end

    if order == "first"
        
        U = Array{Any,1}(undef,2)
        
        U[1] = reshape(exp(-coeff*dt*operator),(d,d,d,d))
        U[2] = reshape(exp(-coeff*dt*operator),(d,d,d,d))
        
    elseif order == "second"
        
        U = Array{Any,1}(undef,3)
        
        U[1] = reshape(exp(-coeff*dt/2*operator),(d,d,d,d))
        U[2] = reshape(exp(-coeff*dt*operator),(d,d,d,d))
        U[3] = reshape(exp(-coeff*dt/2*operator),(d,d,d,d))
        
    elseif order == "forth"
        
        U = Array{Any,1}(undef,11)
        
        dt1 = dt/(4-4^(1/3)); dt2 = (1-4*dt1)*dt
        
        c = 0
        
        for tau in [dt1/2,dt1,dt1,dt1,(dt1+dt2)/2,dt2,(dt1+dt2)/2,dt1,dt1,dt1,dt1/2]
            
            c += 1
            
            U[c] = reshape(exp(-coeff*tau*operator),(d,d,d,d))
        end
    end
    
    return U         
end    


@with_kw mutable struct iMPS

    """
    object of type iMPS has three tensors in vidal notation
    """

    L::Matrix{}  #lambda tensor
    Bo::Array{} #right canonical tensor at odd site
    Be::Array{} #right canonical tensor at even site 
end

function itebd_single_timestep(inf_MPS::iMPS,H::Array{},chi_max::Int64,ctf_val::Float64)
    
    """
    inf_MPS : infinite MPS
    H : Hamiltonian as two site operator
    chi_max : maximum bond dimension
    ctf_val : cutoff value for singular values
    """

    @tensoropt theta[-1,-2,-3,-4] := inf_MPS.L[-1,5]*inf_MPS.Bo[5,6,7]*inf_MPS.Be[7,8,-4]*H[6,8,-2,-3]
    sz = size(theta)

    U,S,V = svd_truncate(reshape(theta,(sz[1]*sz[2],sz[3]*sz[4])),chi_max,ctf_val)
        
    inf_MPS.Bo = reshape(V,(:,sz[3],sz[4]))
            
    @tensoropt inf_MPS.Be[-1,-2,-3] := ((inf_MPS.L)^(-1))[-1,4]*reshape(U,(sz[1],sz[2],:))[4,-2,5]*diagm(S)[5,-3]
            
    inf_MPS.L = diagm(S)
    
    return inf_MPS
end 

function itebd(inf_MPS::iMPS,U::Array{},chi_max::Int64,ctf_val::Float64)
    
    for H in U
        
        inf_MPS = itebd_single_timestep1(inf_MPS,H,chi_max,ctf_val)        
    end
    
    return inf_MPS
end