"""
This file consists of the functions for different spin 1/2 hamiltonians: 
1) Transverse field Ising
2) Long range (power-law decaying) Ising
3) Long range (power-law decaying) Ising in Liouville space (a Superoperator)

Other spin Hamiltonians can be similarly constructed.
"""

using LinearAlgebra
using Kronecker

function Hamiltonian_Ising(h,N,spins::String,longdir::String)

        """
        h : Transverse field
        N : system size
        spins : "Pauli" if Pauli matrices are used
        longdir : longitudinal direction, "X" or "Z"

        The bulk MPO is of size (3,3,d,d). Refer to Annals of Physics 326, 96 (2011)
        section 6.1 for details on construction of MPO for spin Hamiltonians.
        """

    if spins == "Pauli"

        sX = [0 1; 1 0];  sY = [0 -im; im 0];
        sZ = [1 0; 0 -1]; sI = [1 0; 0 1];

    elseif spins == "Onehalf"

        sX = 0.5*[0 1; 1 0]; sY = 0.5*[0 -im; im 0];
        sZ = 0.5*[1 0; 0 -1]; sI = [1 0; 0 1];
    end

    if longdir == "X"

        sLong  = sX
        sTrans = sZ

    elseif longdir == "Z"

        sLong  = sZ
        sTrans = sX
    end

    Ham = Array{Array{Float64,4},1}(undef,N)

    H = zeros(3,3,2,2)

    H[1,1,:,:] = sI; H[3,3,:,:] = sI; H[3,1,:,:] = -h*sTrans
    H[2,1,:,:] = sLong; H[3,2,:,:] = -sLong

    #building the boundary  MPOs
    HL = zeros(1,3,2,2)
    HL[1,:,:,:] = H[3,:,:,:]
    HR = zeros(3,1,2,2)
    HR[:,1,:,:] = H[:,1,:,:]

    #put the hamiltonian in a list so that it can be iteteratively recuperated
    Ham[1] = HL
    Ham[N] = HR

    @inbounds for i in 2:N-1
        Ham[i] = H
    end

    return Ham
end

function Hamiltonian_XXZ(h,N,spins::String)

        """
        h : Transverse field
        N : system size
        spins : "Pauli" if Pauli matrices are used
        """

    if spins == "Pauli"

        sX = [0 1; 1 0];  sY = [0 -im; im 0];
        sZ = [1 0; 0 -1]; sI = [1 0; 0 1];

    elseif spins == "Onehalf"

        sX = 0.5*[0 1; 1 0]; sY = 0.5*[0 -im; im 0];
        sZ = 0.5*[1 0; 0 -1]; sI = [1 0; 0 1];
    end


    Ham = Array{Array{ComplexF64,4},1}(undef,N)

    H = zeros(ComplexF64,5,5,2,2)

    H[1,1,:,:] = sI; H[5,5,:,:] = sI;
    H[2,1,:,:] = sX; H[5,2,:,:] = sX;
    H[3,1,:,:] = sY; H[5,3,:,:] = sY;
    H[4,1,:,:] = sZ; H[5,4,:,:] = h*sZ;



    HL = zeros(ComplexF64,1,5,2,2)
    HL[1,:,:,:] = H[5,:,:,:]
    HR = zeros(ComplexF64,5,1,2,2)
    HR[:,1,:,:] = H[:,1,:,:]


    Ham[1] = HL
    Ham[N] = HR

    @inbounds for i in 2:N-1
        Ham[i] = H
    end

    return Ham
end

function Hamiltonian_NNN_XXX(J1,J2,N,spins::String)

    """
    J : NNN interaction strength
    N : system size
    spins : "Pauli" if Pauli matrices are used
    """

    if spins == "Pauli"

        sX = [0 1; 1 0];  sY = [0 -im; im 0];
        sZ = [1 0; 0 -1]; sI = [1 0; 0 1];

    elseif spins == "Onehalf"

        sX = 0.5*[0 1; 1 0]; sY = 0.5*[0 -im; im 0];
        sZ = 0.5*[1 0; 0 -1]; sI = [1 0; 0 1];
    end

    Ham = Array{Array{ComplexF64,4},1}(undef,N)

    H = zeros(ComplexF64,8,8,2,2)

    H[1,1,:,:] = sI; H[8,8,:,:] = sI; H[3,2,:,:] = sI; H[5,4,:,:] = sI; H[7,6,:,:] = sI;
    H[2,1,:,:] = sX; H[8,2,:,:] = J1*sX; H[8,3,:,:] = J2*sX;
    H[4,1,:,:] = sY; H[8,4,:,:] = J1*sY; H[8,5,:,:] = J2*sY;
    H[6,1,:,:] = sZ; H[8,6,:,:] = J1*sZ; H[8,7,:,:] = J2*sZ;

    HL = zeros(ComplexF64,1,8,2,2)
    HL[1,:,:,:] = H[8,:,:,:]
    HR = zeros(ComplexF64,8,1,2,2)
    HR[:,1,:,:] = H[:,1,:,:]

    Ham[1] = HL
    Ham[N] = HR

    @inbounds for i in 2:N-1
        Ham[i] = H
    end

    return Ham
end


function power_law_to_exp(a::Float64,n::Integer,N::Integer)

        """

        function gives (x_i,lambda_i) such that

        1/r^a = Sum_{i=1-->n} x_i * (lambda_i)^r + errors

        a : interaction strength, a -> infinity is nearest neighbor Ising
            a -> 0 is fully connected Ising.

        n : number of exponential sums. Refer to SciPostPhys.12.4.126 appendix C
            for further details.

        N : lattice size.
        """

    F = Array{Float64,1}(undef,N)

    @inbounds for k in 1:N
        F[k] = 1/k^a
    end

    M = zeros(N-n+1,n)

    @inbounds for j in 1:n
        @inbounds for i in 1:N-n+1
            M[i,j] = F[i+j-1]
        end
    end

    F1 = qr(M)

    Q1 = F1.Q[1:N-n,1:n]
    Q1_inv = pinv(Q1)
    Q2 = F1.Q[2:N-n+1,1:n]

    V = Q1_inv*Q2

    lambda = real(eigvals(V))

    lam_mat = zeros(N,n)

    @inbounds for i in 1:length(lambda)
        @inbounds for k in 1:N
            lam_mat[k,i] = lambda[i]^k
        end
    end

    x = lam_mat\F

    return x, lambda
end

function Kac_norm(a::Float64,N::Int64)

        """
        Kac normalization
        """

    Kac = 0.0

    for i in 1:N
        Kac += (N-i)/i^a
    end

    Kac = Kac/(N-1)

    return Kac
end

function Hamiltonian_LR_Ising(a::Float64,h::Float64,N::Integer,n::Integer,spins::String,kac::String,longdir::String)

        """
        a : interaction strength
        h : Transverse field
        N : system size
        n : number of exponential sums. Also called the MPO bond dimension.
        spins : "Pauli" if Pauli matrices are used
        kac : "true" if kac normalization is used, "false" otherwise
        longdir : longitudinal direction, "X" or "Z"

        For given "n" the bulk MPO is of size (n+2,n+2,d,d). Refer to Phys. Rev. B 78, 035116 (2008)
        for further details on the construction of long range Hamiltonians.
        """

    #basic matrices
    if spins == "Pauli"

        sX = [0 1; 1 0];  sY = [0 -im; im 0];
        sZ = [1 0; 0 -1]; sI = [1 0; 0 1];

    elseif spins == "Onehalf"

        sX = 0.5*[0 1; 1 0]; sY = 0.5*[0 -im; im 0];
        sZ = 0.5*[1 0; 0 -1]; sI = [1 0; 0 1];
    end

    if kac == "true"

        Kac = Kac_norm(a,N)

    elseif kac == "false"

        Kac = 1.0
    end

    if longdir == "X"

        sLong  = sX
        sTrans = sZ

    elseif longdir == "Z"

        sLong  = sZ
        sTrans = sX
    end

    Ham = Array{Array{Float64,4},1}(undef,N)

    x, lambda = power_law_to_exp(a,n,N)

    #building the local bulk MPO
    H = zeros(n+2,n+2,2,2)

    H[1,1,:,:] = sI; H[n+2,n+2,:,:] = sI; H[n+2,1,:,:] = -h*sTrans


    @inbounds for i in 2:n+1
        H[i,1,:,:] = (x[i-1]/Kac)*sLong
        H[i,i,:,:] = lambda[i-1]*sI
    end

    @inbounds for j in 2:n+1
        H[n+2,j,:,:] = -lambda[j-1]*sLong
    end

    #building the boundary  MPOs
    HL = zeros(1,n+2,2,2)
    HL[1,:,:,:] = H[n+2,:,:,:]
    HR = zeros(n+2,1,2,2)
    HR[:,1,:,:] = H[:,1,:,:]

    #put the hamiltonian in a list so that it can be iteteratively recuperated

    Ham[1] = HL
    Ham[N] = HR

    @inbounds for i in 2:N-1
        Ham[i] = H
    end

    return Ham
end

function Hamiltonian_LR_Ising_Liouville(a::Float64,h::Float64,N::Integer,n::Integer,spins::String,kac::String,longdir::String)

        """
        If H is the Hamiltonian in Hilbert space the corresponding Hamiltonian in Liouville
        space is H (*) 1 - 1 (*) H. (*) is tensor product and 1 is a unit matrix of size d*d.

        a : interaction strength
        h : Transverse field
        N : system size
        n : number of exponential sums. Also called the MPO bond dimension.
        spins : 'Pauli' if Pauli matrices are used
        kac : "true" if kac normalization is used, "false" otherwise
        longdir : longitudinal direction, 'X' or 'Z'

        For given 'n' the bulk MPO is of size (2n+2,2n+2,d^2,d^2). Refer to
        'Daniel Jaschke et al 2019 Quantum Sci. Technol. 4 013001' section 3.1.3
        for further details on the construction of MPO in Liouville space.
        """

    #basic matrices
    if spins == "Pauli"

        sX = [0 1; 1 0];  sY = [0 -im; im 0];
        sZ = [1 0; 0 -1]; sI = [1 0; 0 1];

    elseif spins == "Onehalf"

        sX = 0.5*[0 1; 1 0]; sY = 0.5*[0 -im; im 0];
        sZ = 0.5*[1 0; 0 -1]; sI = [1 0; 0 1];
    end

    if kac == "true"

        Kac = kac(a,N)

    elseif kac == "false"

        Kac = 1.0
    end

    I = Matrix(kronecker(sI,sI));
    Zsys = Matrix(kronecker(sZ,sI)); Zanc = Matrix(kronecker(sI,sZ));
    Xsys = Matrix(kronecker(sX,sI)); Xanc = Matrix(kronecker(sI,sX));

    if longdir == "X"

        s_sysLong = Xsys; s_ancLong = Xanc;
        s_sysTrans = Zsys; s_ancTrans = Zanc;

    elseif longdir == "Z"

        s_sysLong = Zsys; s_ancLong = Zanc;
        s_sysTrans = Xsys; s_ancTrans = Xanc;
    end

    Ham = Array{Array{Float64,4},1}(undef,N)

    x, lambda = power_law_to_exp(a,n,N)

    H = zeros(2*n+2,2*n+2,4,4)

    H[1,1,:,:] = I; H[2*n+2,2*n+2,:,:] = I; H[2*n+2,1,:,:] = -h*s_sysTrans + h*s_ancTrans

    @inbounds for i in 1:n

        H[2*i,1,:,:] = (x[i]/Kac)*s_sysLong
        H[2*i+1,1,:,:] = (x[i]/Kac)*s_ancLong

        H[2i,2i,:,:] = lambda[i]*I
        H[2i+1,2i+1,:,:] = lambda[i]*I
    end

    @inbounds for j in 1:n
        H[2n+2,2*j,:,:] = -lambda[j]*s_sysLong
        H[2n+2,2*j+1,:,:] = lambda[j]*s_ancLong
    end

    #building the boundary  MPOs
    HL = zeros(1,2*n+2,4,4)
    HL[1,:,:,:] = H[2*n+2,:,:,:]
    HR = zeros(2*n+2,1,4,4)
    HR[:,1,:,:] = H[:,1,:,:]

    #put the hamiltonian in a list so that it can be iteteratively recuperated

    Ham[1] = HL
    Ham[N] = HR

    @inbounds for i in 2:N-1
        Ham[i] = H
    end

    return Ham
end