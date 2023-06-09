"""
This file consists of Lanczos solvers, both for eigenvalue problem for
DMRG and exponential problem for TDVP.

1) The Lanczos solver functions perform local eigen or exponential procedure,
however because the effective hamiltonian can be written as the contraction of
several tensors we don't build effective hamiltonian explicitly.

2) The matiix to vector multiplication is done by the "linfunct" without having
to build the matrix (effective hamiltonian in our case) explicitly.

The Lanczos implementation used here is a modification to the one presented in a 
very useful website by Glen Evenbly (c) for www.tensors.net, (v1.1)
"""

using LinearAlgebra
using TensorOperations

function EigenLancz(psivec,linfunct,functArgs,krydim::Integer,maxit::Integer,object::String)

        """
        1) psivec : input vector
        2) linfunct : linear function applying operator to the vector
        3) functArgs : tensors that make the effective hamitonian
        4) krydim : krylov dimension
        5) maxit : maximum number of iteration
        6) object : "MPS" or "MPDO"
        """

    if norm(psivec) == 0
        psivec = rand(length(psivec),1);
    end

    psi = zeros(length(psivec),krydim+1);
    A = zeros(krydim,krydim);
    dval = 0;

    for k = 1:maxit

        psi[:,1] = psivec/max(norm(psivec),1e-16);

        for p = 2:krydim+1

            psi[:,p] = linfunct(psi[:,p-1],functArgs...,object)

            for g = p-2:1:p-1

                if g >= 1
                    A[p-1,g] = dot(psi[:,p],psi[:,g]);
                    A[g,p-1] = conj(A[p-1,g]);
                end

            end

            for g = 1:1:p-1
                psi[:,p] = psi[:,p] - dot(psi[:,g],psi[:,p])*psi[:,g];
                psi[:,p] = psi[:,p]/max(norm(psi[:,p]),1e-16);
            end

        end

        G = eigen(0.5*(A+A'));
        dval, xloc = findmin(G.values);
        psivec = psi[:,1:krydim]*G.vectors[:,xloc[1]];
    end

    psivec = psivec/norm(psivec);

    return psivec, dval
end

function ExpLancz(psivec,linfunct,functArgs,krydim_TDVP::Integer,dt::Float64,close_cutoff::Number,object::String,evol_type::String)

        """
        1) psivec : input vector
        2) linfunct : linear function applying operator to the vector
        3) functArgs : tensors that make the effective hamitonian
        4) krydim_TDVP : krylov dimension
        5) dt = time step
        6) close_cutoff = cutoff for exponential
        7) object : "MPS" or "MPDO"
        8) evol_type : "real" or "imaginary"

        Note: Real time evolution involves complex numbers, imaginary time evolution
        involves real number.
        """

    if norm(psivec) == 0
        psivec = rand(length(psivec),1);
    end

    if evol_type == "real"

        psi = zeros(ComplexF64,length(psivec),krydim_TDVP+1);
        A_mat = zeros(ComplexF64,krydim_TDVP,krydim_TDVP);
        output_vec = zeros(ComplexF64,length(psivec));
        transit_vec = zeros(ComplexF64,length(psivec));

    elseif evol_type == "imaginary"

        psi = zeros(length(psivec),krydim_TDVP+1);
        A_mat = zeros(krydim_TDVP,krydim_TDVP);
        output_vec = zeros(length(psivec));
        transit_vec = zeros(length(psivec));
    end

    nom = norm(psivec)

    psi[:,1] = psivec/max(norm(psivec),1e-16);

    for p = 2:krydim_TDVP+1

        output_vec = 0*output_vec

        psi[:,p] = linfunct(psi[:,p-1],functArgs...,object)

        for g = p-2:1:p-1
            if g >= 1
                A_mat[p-1,g] = dot(psi[:,p],psi[:,g]);
                A_mat[g,p-1] = conj(A_mat[p-1,g]);
            end
        end

        for g = 1:1:p-1
            psi[:,p] = (psi[:,p] - dot(psi[:,g],psi[:,p])*psi[:,g])
        end
        psi[:,p] = psi[:,p]/max(norm(psi[:,p]),1e-16);

        if p > 3

            output_vec = Lancz_final(A_mat[1:p-1,1:p-1],psi,dt,output_vec,evol_type)
            c = closeness(transit_vec,output_vec,close_cutoff)

            if c == length(output_vec)
                break
            else
                transit_vec = output_vec
            end
        end
    end

    return nom*output_vec
end


function Lancz_final(A_mat,psi,dt,output_vec,evol_type::String)

        """
        Exponenting the tri-diagonal matrix "A_mat" and applying in applying in the
        vector "psi".

        dt : time step

        output_vec = vector with same size as "psi"

        evol_type : "real" for real time evolution, "imaginary" for imaginary time evolution
        """

    if evol_type == "real"

        c = exp(-im*(dt)*A_mat)*I(length(A_mat[:,1]))[:,1]

    elseif evol_type == "imaginary"

        c = exp(-dt*A_mat)*I(length(A_mat[:,1]))[:,1]
    end

    for i in 1:length(c)
        output_vec += c[i]*psi[:,i]
    end

    return output_vec
end

function closeness(list1,list2,cutoff)

        """
        Calculates closeness of two vectors (lists) according to a cutoff.
        Hard condition : true only if every corresponding elements are smaller than cutoff.
        """

    c = 0
    for i in 1:length(list1)
        if abs(list1[i]-list2[i]) <= cutoff
            c += 1
        end
    end

    return c
end
