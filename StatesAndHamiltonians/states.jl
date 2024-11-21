"""
    File containing several initial spin 1/2 states that can be written as
    tensor products. There are three categories of it.

    1) MPS states in Hilbert space.

    2)Maximally entangled state (at infinite temperature, beta = 0) as MPDO. This
    is the initial state for local purification of maximally entangled state.

    3) MPS states in Liouville space. Every site in system is paired with an ancilla
    that acts as the bath. Useful for open system simulation. The local dimension becomes
    d^2 from d.
 """
 
using LinearAlgebra
using Kronecker

function random_psi(N::Integer,chi::Integer,d::Integer)

        """
        Random MPS of size (chi,d,chi), used as initial state for one-site DMRG
        """

    psi = Array{Any,1}(undef,N);

    psi[1] = rand(1,d,chi)
    psi[N] = rand(chi,d,1)

    @inbounds for i in 2:N-1

        psi[i] = rand(chi,d,chi)
    end

    return psi
end

function fully_pol_Z(N::Integer,spin_state::String)

        """
        Fully polarized state with N site in Z direction.

        spin_state : "up" or "down"
        """

    psi = Array{Any,1}(undef,N);

    if spin_state == "up"

        A = [1 0]

    elseif spin_state == "down"

        A = [0 1]
    end

    A = reshape(A,(1,2,1))

    for i in 1:N
        psi[i] = A
    end

    return psi
end

function neel_Z(N::Integer)

    """
    neel state with N site in Z direction.
    """

    psi = Array{Any,1}(undef,N);

    A_up = [1 0];
    A_down = [0 1];

    A_up = reshape(A_up,(1,2,1))
    A_down = reshape(A_down,(1,2,1))

    for i in 1:N
        if mod(i, 2) == 0
            psi[i] = A_up
        else
            psi[i] = A_down
        end
    end

    return psi
end

function fully_pol_X(N,spin_state::String)

        """
        Fully polarized state with N site in X direction.

        spin_state : "up" or "down"
        """

    psi = Array{Any,1}(undef,N);

    if spin_state == "up"

        A = [1/sqrt(2) 1/sqrt(2)]

    elseif spin_state == "down"

        A = [1/sqrt(2) -1/sqrt(2)]
    end

    A = reshape(A,(1,2,1))

    for i in 1:N
        psi[i] = A
    end

    return psi
end

function neel_X(N::Integer)

    """
    neel state with N site in X direction.
    """

    psi = Array{Any,1}(undef,N);

    A_up = [1/sqrt(2) 1/sqrt(2)];
    A_down = [1/sqrt(2) -1/sqrt(2)];

    A_up = reshape(A_up,(1,2,1))
    A_down = reshape(A_down,(1,2,1))

    for i in 1:N
        if mod(i, 2) == 0
            psi[i] = A_up
        else
            psi[i] = A_down
        end
    end

    return psi
end

function polarized_Z_to_X(N::Integer,theta::Any,spin_state::String)

    """
    Fully polarized state with N site in direction theta from Z towards X

    spin_state : "up" or "down"
    """

    sX = [0 1; 1 0];  sY = [0 -im; im 0];
    sZ = [1 0; 0 -1]; sI = [1 0; 0 1];

    psi = Array{Any,1}(undef,N);

    #single spin rotation operator 
    Ry = cos(theta/2)*sI - im*sin(theta/2)*sY
    
    if spin_state == "up"

        A = Ry*transpose([1 0])

    elseif spin_state == "down"

        A = Ry*transpose([0 1])
    end

    A = reshape(A,(1,2,1))

    for i in 1:N
        psi[i] = A
    end

    return psi
end

function GHZ_Z(N)

        """
        GHZ state with N site in Z direction.
        """

    psi = Array{Any,1}(undef,N);

    AL = zeros(1,2,2)
    AL[:,1,:] = (1/sqrt(2))*[1 0]
    AL[:,2,:] = (1/sqrt(2))*[0 1]

    AR = zeros(2,2,1)
    AR[:,1,:] = transpose([1 0])
    AR[:,2,:] = transpose([0 1])

    A = zeros(2,2,2)
    A[:,1,:] = [1 0;0 0]
    A[:,2,:] = [0 0;0 1]

    psi[1] = AL
    psi[N] = AR

    for i in 2:N-1
        psi[i] = A
    end

    return psi
end

function GHZ_X(N)

        """
        GHZ state with N site in X direction.
        """

    psi = Array{Any,1}(undef,N);

    AL = zeros(1,2,2)
    AL[:,1,:] = (1/sqrt(2))*[1/sqrt(2) 1/sqrt(2)]
    AL[:,2,:] = (1/sqrt(2))*[1/sqrt(2) -1/sqrt(2)]

    AR = zeros(2,2,1)
    AR[:,1,:] = transpose([1/sqrt(2) 1/sqrt(2)])
    AR[:,2,:] = transpose([1/sqrt(2) -1/sqrt(2)])

    A = zeros(2,2,2)
    A[:,1,:] = [1/sqrt(2) 0;0 1/sqrt(2)]
    A[:,2,:] = [1/sqrt(2) 0;0 -1/sqrt(2)]

    psi[1] = AL
    psi[N] = AR

    for i in 2:N-1
        psi[i] = A
    end

    return psi
end


function maximally_entangled(N)

        """
        maximally entangled state (at infinite temperature) with N sites.
        """

    psi = Array{Any,1}(undef,N);

    A = rand(1,2,2,1)
    A[1,:,:,1] = 1/sqrt(2) *[1 0; 0 1]

    @inbounds for i in 1:N
        psi[i] = A
    end

    return psi
end


function fully_pol_liouville_Z(N,spin_state::String)

            """
            Fully polarized state in Liouville space with N site in Z direction.

            spin_state : "up" or "down"
            """

    psi = Array{Any,1}(undef,N)

    if spin_state == "up"

        c = Matrix(kronecker([1 0],[1 0]))

    elseif spin_state == "down"

        c = Matrix(kronecker([0 1],[0 1]))
    end

    c = reshape(c,(1,4,1))

    for i in 1:N
        psi[i] = c
    end

    return psi
end


function fully_pol_liouville_X(N,spin_state::String)

            """
            Fully polarized state in Liouville space with N site in X direction.

            spin_state : "up" or "down"
            """

    psi = Array{Any,1}(undef,N)

    if spin_state == "up"

        c = Matrix(kronecker([1/sqrt(2) 1/sqrt(2)],[1/sqrt(2) 1/sqrt(2)]))

    elseif spin_state == "down"

        c = Matrix(kronecker([1/sqrt(2) -1/sqrt(2)],[1/sqrt(2) -1/sqrt(2)]))
    end

    c = reshape(c,(1,4,1))

    for i in 1:N
        psi[i] = c
    end

    return psi
end


function GHZ_liouville_Z(N)

        """
        GHZ state in Liouville space with N site in Z direction.
        """

    psi = Array{Any,1}(undef,N)

    AL = zeros(1,4,2)
    AL[:,1,:] = 1/sqrt(2)*[1 0];AL[:,2,:] = 1/sqrt(2)*[0 0]
    AL[:,3,:] = 1/sqrt(2)*[0 0];AL[:,4,:] = 1/sqrt(2)*[0 1]

    AR = zeros(2,4,1)
    AR[:,1,:] = transpose([1 0]);AR[:,2,:] = transpose([0 0])
    AR[:,3,:] = transpose([0 0]);AR[:,4,:] = transpose([0 1])

    A = zeros(2,4,2)
    A[:,1,:] = [1 0;0 0];A[:,2,:] = [0 0;0 0]
    A[:,3,:] = [0 0;0 0];A[:,4,:] = [0 0;0 1]

    psi[1] = AL
    psi[N] = AR

    for i in 2:N-1
        psi[i] = A
    end

    return psi
end

function GHZ_liouville_X(N)

        """
        GHZ state in Liouville space with N site in X direction.
        """

    psi = Array{Any,1}(undef,N)

    AL = zeros(1,4,2)
    AL[:,1,:] = 1/sqrt(2)*[0.5 0.5];AL[:,2,:] = 1/sqrt(2)*[0.5 -0.5]
    AL[:,3,:] = 1/sqrt(2)*[0.5 -0.5];AL[:,4,:] = 1/sqrt(2)*[0.5 0.5]

    AR = zeros(2,4,1)
    AR[:,1,:] = transpose([0.5 0.5]);AR[:,2,:] = transpose([0.5 -0.5])
    AR[:,3,:] = transpose([0.5 -0.5]);AR[:,4,:] = transpose([0.5 0.5])

    A = zeros(2,4,2)
    A[:,1,:] = [0.5 0;0 0.5];A[:,2,:] = [0.5 0;0 -0.5]
    A[:,3,:] = [0.5 0;0 -0.5];A[:,4,:] = [0.5 0;0 0.5]

    psi[1] = AL
    psi[N] = AR

    for i in 2:N-1
        psi[i] = A
    end

    return psi
end
