using LinearAlgebra
using TensorOperations

"""
This file consists of functions for one and two site TDVP (for both MPS and MPDO using LPTN) algorithms.
The two seminal papers are:
1) https://doi.org/10.1103/PhysRevLett.107.070601 : for more mathematical background behind TDVP
2) https://doi.org/10.1103/PhysRevB.94.165116 : for more visual approach to TDVP

But the best explanation and a ready to use recipe for TDVP and several other timedependent MPS methods
are found in https://doi.org/10.1016/j.aop.2019.167998
"""

function right_sweep_TDVP_onesite(dt,M,psi,Env,Ham,N::Integer,krydim_TDVP::Integer,d::Integer,close_cutoff::Number)

    """
    dt : Trotter time step
    M : non-canonical MPS at the edge
    psi : input state as right canonical MPS
    Env : Environment tensors, consists both left environment and right environnent tensors.
    Ham : MPO
    N : lattice size
    krydim_TDVP : number of krylov vectors
    d : local dimension
    close_cutoff : cutoff to stop the Lanczos exponential.
    """

    for i in 1:N

        shp_M = size(M)

        M = reshape(M,(shp_M[1]*shp_M[2]*shp_M[3]))

        M = ExpLancz(M,MpoToMpsOneSite,(Env[indx(i-1,N)],Ham[i],Env[i+1],d),krydim_TDVP,dt/2,close_cutoff,"MPS","real")

        M = reshape(M,(shp_M[1],shp_M[2],shp_M[3]))

        QR_dec = qr(reshape(M,(shp_M[1]*shp_M[2],shp_M[3])))

        psi[i] = reshape(Matrix(QR_dec.Q),(shp_M[1],shp_M[2],:))

        Env[i] = contract_left(psi[i],Env[indx(i-1,N)],Ham[i],"MPS")

        if i != N

            shp_C = size(QR_dec.R,)

            C = ExpLancz(reshape(QR_dec.R,(shp_C[1]*shp_C[2])),MpoToMpsOneSiteKeff,(Env[i],Env[i+1]),krydim_TDVP,-dt/2,close_cutoff,"MPS","real")

            C = reshape(C,(shp_C[1],shp_C[2]))

            @tensor M[-1,-2,-3] := C[-1,4]*psi[i+1][4,-2,-3]

            Env[i+1] = nothing
        end
    end
    return M,psi,Env
end

function left_sweep_TDVP_onesite(dt,M,psi,Env,Ham,N::Integer,krydim_TDVP::Integer,d::Integer,close_cutoff::Number)

    """
    dt : Trotter time step
    M : non-canonical MPS at the edge
    psi : input state as left canonical MPS
    Env : Environment tensors, consists both left environment and right environnent tensors.
    Ham : MPO
    N : lattice size
    krydim_TDVP : number of krylov vectors
    d : local dimension
    close_cutoff : cutoff to stop the Lanczos exponential.
    """

    @inbounds for i in reverse(1:N)

        shp_M = size(M)

        M = reshape(M,(shp_M[1]*shp_M[2]*shp_M[3]))

        M = ExpLancz(M,MpoToMpsOneSite,(Env[indx(i-1,N)],Ham[i],Env[i+1],d),krydim_TDVP,dt/2,close_cutoff,"MPS","real")

        M = reshape(M,(shp_M[1],shp_M[2],shp_M[3]))

        LQ_dec = lq(reshape(M,(shp_M[1],shp_M[2]*shp_M[3])))

        psi[i] = reshape(Matrix(LQ_dec.Q),(:,shp_M[2],shp_M[3]))

        Env[i] = contract_right(psi[i],Env[i+1],Ham[i],"MPS")

        if i != 1

            shp_C = size(LQ_dec.L)

            C = ExpLancz(reshape(LQ_dec.L,(shp_C[1]*shp_C[2])),MpoToMpsOneSiteKeff,(Env[i-1],Env[i]),krydim_TDVP,-dt/2,close_cutoff,"MPS","real")

            C = reshape(C,(shp_C[1],shp_C[2]))

            @tensor M[-1,-2,-3] := psi[i-1][-1,-2,4]*C[4,-3]

            Env[i-1] = nothing
        end
    end
    return M,psi,Env
end

function right_sweep_TDVP_twosite(dt,M,psi,Env,Ham,N::Integer,krydim_TDVP::Integer,d::Integer,chi_max::Integer,close_cutoff::Number,ctf_val::Number)

    """
    dt : Trotter time step
    M : non-canonical MPS at the edge
    psi : input state as right canonical MPS
    Env : Environment tensors, consists both left environment and right environnent tensors.
    Ham : MPO
    N : lattice size
    krydim_TDVP : number of krylov vectors
    d : local dimension
    chi_max : upper bound for bond dimension
    close_cutoff : cutoff to stop the Lanczos exponential.
    ctf_val : cutoff for the singular values in SVD.
    """

    for i in 1:N-1

        @tensor T[-1,-2,-3,-4] := M[-1,-2,5]*psi[i+1][5,-3,-4]

        shp_T = size(T)

        T = reshape(T,(shp_T[1]*shp_T[2]*shp_T[3]*shp_T[4]))

        T = ExpLancz(T,MpoToMpsTwoSite,(Env[indx(i-1,N)],Ham[i],Ham[i+1],Env[i+2],d),krydim_TDVP,dt/2,close_cutoff,"MPS","real")

        T = reshape(T,(shp_T[1]*shp_T[2],shp_T[3]*shp_T[4]))

        U,S,V = svd_truncate(T,chi_max,ctf_val)

        psi[i] = reshape(U,(shp_T[1],shp_T[2],:))

        @tensor M[-1,-2,-3] := diagm(S)[-1,4]*reshape(V,(:,shp_T[3],shp_T[4]))[4,-2,-3]

        if i != N-1

            Env[i] = contract_left(psi[i],Env[indx(i-1,N)],Ham[i],"MPS")

            shp_M = size(M)

            M = reshape(M,(shp_M[1]*shp_M[2]*shp_M[3]))

            M = ExpLancz(M,MpoToMpsOneSite,(Env[i],Ham[i+1],Env[i+2],d),krydim_TDVP,-dt/2,close_cutoff,"MPS","real")

            M = reshape(M,(shp_M[1],shp_M[2],shp_M[3]))

            Env[i+2] = nothing
            psi[i+1] = nothing
        end
    end
    return M,psi,Env
end

function left_sweep_TDVP_twosite(dt,M,psi,Env,Ham,N::Integer,krydim_TDVP::Integer,d::Integer,chi_max::Integer,close_cutoff::Number,ctf_val::Number)

    """
    dt : Trotter time step
    M : non-canonical MPS at the edge
    psi : input state as left canonical MPS
    Env : Environment tensors, consists both left environment and right environnent tensors.
    Ham : MPO
    N : lattice size
    krydim_TDVP : number of krylov vectors
    d : local dimension
    chi_max : upper bound for bond dimension
    close_cutoff : cutoff to stop the Lanczos exponential.
    ctf_val : cutoff for the singular values in SVD.
    """

    for j in reverse(2:N)

        @tensor T[-1,-2,-3,-4] := psi[j-1][-1,-2,5]*M[5,-3,-4]

        shp_T = size(T)

        T = reshape(T,(shp_T[1]*shp_T[2]*shp_T[3]*shp_T[4]))

        T = ExpLancz(T,MpoToMpsTwoSite,(Env[indx(j-2,N)],Ham[j-1],Ham[j],Env[j+1],d),krydim_TDVP,dt/2,close_cutoff,"MPS","real")

        T = reshape(T,(shp_T[1]*shp_T[2],shp_T[3]*shp_T[4]))

        U,S,V = svd_truncate(T,chi_max,ctf_val)

        psi[j] = reshape(V,(:,shp_T[3],shp_T[4]))

        @tensor M[-1,-2,-3] := reshape(U,(shp_T[1],shp_T[2],:))[-1,-2,4]*diagm(S)[4,-3]

        if j != 2

            Env[j] = contract_right(psi[j],Env[j+1],Ham[j],"MPS")

            shp_M = size(M)

            M = reshape(M,(shp_M[1]*shp_M[2]*shp_M[3]))

            M = ExpLancz(M,MpoToMpsOneSite,(Env[j-2],Ham[j-1],Env[j],d),krydim_TDVP,-dt/2,close_cutoff,"MPS","real")

            M = reshape(M,(shp_M[1],shp_M[2],shp_M[3]))

            Env[j-2] = nothing
            psi[j-1] = nothing
        end
    end

    return M,psi,Env
end

"""
The TDVP sweeps below are for the imaginary time evolution of the density matrix (usually employed to cool down the system
from infinite temperature, maximally mixed state, to a given temperature) in a process called Locally Purified Tensor Network.
The object of interest here is a MPS like object with an additional index to physical index called the Krauss index, which in
the case of finite temperature evolution remains vestigial (in an open system however this index is also renormalized).
Refer to https://doi.org/10.1103/PhysRevLett.116.237201 for further reading.
"""

function right_sweep_TDVP_twosite_MPDO(dt,M,psi,Env,Ham,N::Integer,krydim_TDVP::Integer,d::Integer,chi_max::Integer,close_cutoff::Number,ctf_val::Number)

    """
    dt : Trotter time step
    M : non-canonical MPS at the edge
    psi : input state as right canonical MPS
    Env : Environment tensors, consists both left environment and right environnent tensors.
    Ham : MPO
    N : lattice size
    krydim_TDVP : number of krylov vectors
    d : local dimension
    chi_max : upper bound for bond dimension
    close_cutoff : cutoff to stop the Lanczos exponential.
    ctf_val : cutoff for the singular values in SVD.
    """

    for i in 1:N-1

        @tensor T[-1,-2,-3,-4,-5,-6] := M[-1,-2,-3,7]*psi[i+1][7,-4,-5,-6]

        shp_T = size(T)

        T = reshape(T,(shp_T[1]*shp_T[2]*shp_T[3]*shp_T[4]*shp_T[5]*shp_T[6]))

        T = ExpLancz(T,MpoToMpsTwoSite,(Env[indx(i-1,N)],Ham[i],Ham[i+1],Env[i+2],d),krydim_TDVP,dt/2,close_cutoff,"MPDO","imaginary")

        T = reshape(T,(shp_T[1]*shp_T[2]*shp_T[3],shp_T[4]*shp_T[5]*shp_T[6]))

        U,S,V = svd_truncate(T,chi_max,ctf_val)

        psi[i] = reshape(U,(shp_T[1],shp_T[2],shp_T[3],:))

        @tensor M[-1,-2,-3,-4] := diagm(S)[-1,5]*reshape(V,(:,shp_T[4],shp_T[5],shp_T[6]))[5,-2,-3,-4]

        if i != N-1

            Env[i] = contract_left(psi[i],Env[indx(i-1,N)],Ham[i],"MPDO")

            shp_M = size(M)

            M = reshape(M,(shp_M[1]*shp_M[2]*shp_M[3]*shp_M[4]))

            M = ExpLancz(M,MpoToMpsOneSite,(Env[i],Ham[i+1],Env[i+2],d),krydim_TDVP,-dt/2,close_cutoff,"MPDO","imaginary")

            M = reshape(M,(shp_M[1],shp_M[2],shp_M[3],shp_M[4]))

            Env[i+2] = nothing
            psi[i+1] = nothing
        end
    end
    return M,psi,Env
end

function left_sweep_TDVP_twosite_MPDO(dt,M,psi,Env,Ham,N::Integer,krydim_TDVP::Integer,d::Integer,chi_max::Integer,close_cutoff::Number,ctf_val::Number)

    """
    dt : Trotter time step
    M : non-canonical MPS at the edge
    psi : input state as left canonical MPS
    Env : Environment tensors, consists both left environment and right environnent tensors.
    Ham : MPO
    N : lattice size
    krydim_TDVP : number of krylov vectors
    d : local dimension
    chi_max : upper bound for bond dimension
    close_cutoff : cutoff to stop the Lanczos exponential.
    ctf_val : cutoff for the singular values in SVD.
    """

    for i in reverse(2:N)
                
        @tensor T[-1,-2,-3,-4,-5,-6] := psi[i-1][-1,-2,-3,7]*M[7,-4,-5,-6]

        shp_T = size(T)

        T = reshape(T,(shp_T[1]*shp_T[2]*shp_T[3]*shp_T[4]*shp_T[5]*shp_T[6]))
        
        T = ExpLancz(T,MpoToMpsTwoSite,(Env[indx(i-2,N)],Ham[i-1],Ham[i],Env[i+1],d),krydim_TDVP,dt/2,close_cutoff,"MPDO","imaginary")

        T = reshape(T,(shp_T[1]*shp_T[2]*shp_T[3],shp_T[4]*shp_T[5]*shp_T[6]))

        U,S,V = svd_truncate(T,chi_max,ctf_val)

        psi[i] = reshape(V,(:,shp_T[4],shp_T[5],shp_T[6]))

        @tensor M[-1,-2,-3,-4] := reshape(U,(shp_T[1],shp_T[2],shp_T[3],:))[-1,-2,-3,5]*diagm(S)[5,-4]

        if i != 2

            Env[i] = contract_right(psi[i],Env[i+1],Ham[i],"MPDO")

            shp_M = size(M)

            M = reshape(M,(shp_M[1]*shp_M[2]*shp_M[3]*shp_M[4]))

            M = ExpLancz(M,MpoToMpsOneSite,(Env[i-2],Ham[i-1],Env[i],d),krydim_TDVP,-dt/2,close_cutoff,"MPDO","imaginary")

            M = reshape(M,(shp_M[1],shp_M[2],shp_M[3],shp_M[4]))

            Env[i-2] = nothing
            psi[i-1] = nothing
        end
    end
    return M,psi,Env
end
