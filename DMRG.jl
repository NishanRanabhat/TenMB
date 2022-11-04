using LinearAlgebra
using TensorOperations

"""
This file consists of functions for one and two site DMRG algorithms.

The DMRG algorithm has been discussed from scratch and greater detail in
https://doi.org/10.1016/j.aop.2010.09.012
"""

function right_sweep_DMRG_one_site(M,psi,Env,Ham,N::Integer,krydim_DMRG::Integer,maxit::Integer,d::Integer)

    """
    M : non-canonical MPS at the edge
    psi : input state as right canonical MPS
    Env : Environment tensors, consists both left environment and right environnent tensors.
    Ham : MPO
    N : lattice size
    krydim_DMRG : number of krylov vectors
    maxit : maximum iteration
    d : local dimension
    """

    @inbounds for i in 1:N-1

        shp = size(M)

        eig_vec, eig_val = EigenLancz(reshape(M,(shp[1]*shp[2]*shp[3])),MpoToMpsOneSite,(Env[indx(i-1,N)],Ham[i],Env[i+1],d),krydim_DMRG,maxit,"MPS")
        
        println(eig_val/N)

        QR_dec = qr(reshape(eig_vec,(shp[1]*shp[2],shp[3])))

        psi[i] = reshape(Matrix(QR_dec.Q),(shp[1],shp[2],:))

        Env[i] = contract_left(psi[i],Env[indx(i-1,N)],Ham[i],"MPS")

        @tensor M[-1,-2,-3] := QR_dec.R[-1,4]*psi[i+1][4,-2,-3]

        Env[i+1] = nothing
    end

    return M,psi,Env
end

function left_sweep_DMRG_one_site(M,psi,Env,Ham,N::Integer,krydim_DMRG::Integer,maxit::Integer,d::Integer)

    """
    M : non-canonical MPS at the edge
    psi : input state as left canonical MPS
    Env : Environment tensors, consists both left environment and right environnent tensors.
    Ham : MPO
    N : lattice size
    krydim_DMRG : number of krylov vectors
    maxit : maximum iteration
    d : local dimension
    """

    @inbounds for i in reverse(2:N)

        shp = size(M)

        eig_vec, eig_val = EigenLancz(reshape(M,(shp[1]*shp[2]*shp[3])),MpoToMpsOneSite,(Env[indx(i-1,N)],Ham[i],Env[i+1],d),krydim_DMRG,maxit,"MPS")

        println(eig_val/N)

        LQ_dec = lq(reshape(eig_vec,(shp[1],shp[2]*shp[3])))

        psi[i] = reshape(Matrix(LQ_dec.Q),(:,shp[2],shp[3]))

        Env[i] = contract_right(psi[i],Env[i+1],Ham[i],"MPS")

        @tensor M[-1,-2,-3] := psi[i-1][-1,-2,4]*LQ_dec.L[4,-3]

        Env[indx(i-1,N)]  = nothing

    end
    return M,psi,Env
end

function right_sweep_DMRG_two_site(M,psi,Env,Ham,N::Integer,krydim_DMRG::Integer,maxit::Integer,d::Integer,chi_max::Integer,ctf_val::Number)

    """
    M : non-canonicalized MPS at the edge
    psi : input state as right canonical MPS
    R : environment tensors, both left and right
    Ham : MPO
    N : lattice size
    krydim_DMRG : number of krylov vectors
    maxit : maximum iteration
    d : local dimension
    chi_max : upper cutoff for bond dimension
    ctf_val : cutoff value for truncation error
    """

    @inbounds for i in 1:N-1

        @tensor T[-1,-2,-3,-4] := M[-1,-2,5]*psi[i+1][5,-3,-4]

        shp = size(T)

        eig_vec, eig_val = EigenLancz(reshape(T,(shp[1]*shp[2]*shp[3]*shp[4])),MpoToMpsTwoSite,(Env[indx(i-1,N)],Ham[i],Ham[i+1],Env[i+2],d),krydim_DMRG,maxit,"MPS")
        
        println(eig_val/N)

        U,S,V = svd_truncate(reshape(eig_vec,(shp[1]*shp[2],shp[3]*shp[4])),chi_max,ctf_val)

        psi[i] = reshape(U,(shp[1],shp[2],:))

        @tensor M[-1,-2,-3] := diagm(S)[-1,4]*reshape(V,(:,shp[3],shp[4]))[4,-2,-3]

        Env[i] = contract_left(psi[i],Env[indx(i-1,N)],Ham[i],"MPS")

        if i != N-1
            Env[i+2] = nothing
            psi[i+1] = nothing
        end
    end
    return M,psi,Env
end

function left_sweep_DMRG_two_site(M,psi,Env,Ham,N::Integer,krydim_DMRG::Integer,maxit::Integer,d::Integer,chi_max::Integer,ctf_val::Number)

    """
    M : non-canonicalized MPS at the edge
    psi : input state as left canonical MPS
    Env : Environment tensors, both left and right
    Ham : MPO
    N : lattice size
    krydim_DMRG : number of krylov vectors
    maxit : maximum iteration
    d : local dimension
    chi_max : upper cutoff for bond dimension
    ctf_val : cutoff value for truncation error
    """

    @inbounds for j in reverse(2:N)

        @tensor T[-1,-2,-3,-4] := psi[j-1][-1,-2,5]*M[5,-3,-4]

        shp = size(T)

        eig_vec, eig_val = EigenLancz(reshape(T,(shp[1]*shp[2]*shp[3]*shp[4])),MpoToMpsTwoSite,(Env[indx(j-2,N)],Ham[j-1],Ham[j],Env[j+1],d),krydim_DMRG,maxit,"MPS")

        println(eig_val/N)

        U,S,V = svd_truncate( reshape(eig_vec,(shp[1]*shp[2],shp[3]*shp[4])),chi_max,ctf_val)

        psi[j] = reshape(V,(:,shp[3],shp[4]))

        @tensor M[-1,-2,-3] := reshape(U,(shp[1],shp[2],:))[-1,-2,4]*diagm(S)[4,-3]

        Env[j] = contract_right(psi[j],Env[j+1],Ham[j],"MPS")

        if j != 2
            Env[j-2] = nothing
            psi[j-1] = nothing
        end
    end
    return M,psi,Env
end
