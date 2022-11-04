"""
This file consists of all the necessary contraction routines for DMRG and TDVP
algorithms.

1) The routines for the contraction of MPDO and MPS are written in the
same function. The argument "object" distinguishes between them.

2) Every function is provided with the tensor network diagram representing
the contraction performed for MPS case, the MPDO case follows similarly.

3) Finally, every tensor network contraction should be performed in an optimal
manner, here the @tensoropt macro automatically finds the optimal contraction order.

4) The TensorOperations library is said to benefit from multithreading so asking for 
threads --n where n is the number of threads will acclerate the contractions.
"""

"""
Some modifications
"""

using LinearAlgebra
using TensorOperations

function contract_right(B,R,W,object::String)

        """
         _ _ _ _                                    _ _ _ _
     ~~~|       |                    ~~~B_i---  ---|       |
        |       |                       |          |       |
        |       |                                  |       |
        |       |                       |          |       |
     ~~~|  R_i  |         <---       ~~~W_i---  ---|R_{i+1}|
        |       |                       |          |       |
        |       |                                  |       |
        |       |                       |          |       |
     ~~~|_ _ _ _|                   ~~~B'_i---  ---|_ _ _ _|

        """

    if object == "MPDO"

        @tensoropt fin[-1,-2,-3] := conj(B)[-1,4,5,6]*R[6,7,8]*W[-2,7,4,9]*B[-3,9,5,8]
        return fin

    elseif object == "MPS"

        @tensoropt fin[-1,-2,-3] := conj(B)[-1,4,5]*R[5,6,7]*W[-2,6,4,8]*B[-3,8,7]
        return fin

    end
end

function contract_left(A,L,W,object::String)

        """
         _ _ _ _                           _ _ _ _
        |       |---  ---A_i~~~           |       |~~~
        |       |        |                |       |
        |       |                         |       |
        |       |        |                |       |
        |L_{i-1}|---  ---W_i~~~    --->   | L_{i} |~~~
        |       |        |                |       |
        |       |                         |       |
        |       |        |                |       |
        |_ _ _ _|---  ---A'_i~~~          |_ _ _ _|~~~

        """

    if object == "MPDO"

        @tensoropt fin[-1,-2,-3] := conj(A)[6,5,4,-1]*L[6,7,8]*W[7,-2,5,9]*A[8,9,4,-3]
        return fin

    elseif object == "MPS"

        @tensoropt fin[-1,-2,-3] := conj(A)[5,4,-1]*L[5,6,7]*W[6,-2,4,8]*A[7,8,-3]
        return fin

    end
end

function contract_left_noop(A,L,object::String)

        """
         _ _ _ _                           _ _ _ _
        |       |---  ---A_i~~~           |       |~~~
        |       |        |                |       |
        |       |                         |       |
        |       |                         |       |
        |L_{i-1}|                 --->    | L_{i} |
        |       |                         |       |
        |       |                         |       |
        |       |        |                |       |
        |_ _ _ _|---  ---A'_i~~~          |_ _ _ _|~~~

        """

    if object == "MPDO"

        @tensoropt fin[-1,-2] :=  conj(A)[5,4,3,-1]*L[5,6]*A[6,4,3,-2]
        return fin

    elseif object == "MPS"

        @tensoropt fin[-1,-2] :=  conj(A)[4,3,-1]*L[4,5]*A[5,3,-2]
        return fin

    end
end

function contract_left_nompo(A,L,W,object::String)

        """
         _ _ _ _                           _ _ _ _
        |       |---  ---A_i~~~           |       |~~~
        |       |        |                |       |
        |       |                         |       |
        |       |        |                |       |
        |L_{i-1}|---    W_i       --->    | L_{i} |
        |       |        |                |       |
        |       |                         |       |
        |       |        |                |       |
        |_ _ _ _|---  ---A'_i~~~          |_ _ _ _|~~~

        """

    if object == "MPDO"

        @tensoropt fin[-1,-2] := conj(A)[5,4,3,-1]*L[5,6]*W[4,7]*A[6,7,3,-2]
        return fin

    elseif object == "MPS"

        @tensoropt fin[-1,-2] := conj(A)[4,3,-1]*L[4,5]*W[3,6]*A[5,6,-2]
        return fin

    end
end

function MpoToMpsOneSite(M,L,W,R,d::Integer,object::String)

        """
         _ _ _ _                    _ _ _ _
        |       |---  ---M_i--- ---|       |
        |       |        |         |       |
        |       |                  |       |
        |       |        |         |       |
        |L_{i-1}|---  ---W_i--  ---|R_{i+1}|   --->  ~~~M'_i~~~
        |       |        |         |       |             |
        |       |                  |       |
        |       |                  |       |
        |_ _ _ _|~~~            ~~~|_ _ _ _|

        """

    if object == "MPDO"

        M =  reshape(M,(size(L,3),d,d,size(R,3)))

        @tensoropt fin[-1,-2,-3,-4] := L[-1,7,8]*M[8,9,-3,11]*W[7,10,-2,9]*R[-4,10,11]

        fin = reshape(fin,(size(L,1)*d*d*size(R,1)))

        return fin

    elseif object == "MPS"

        M =  reshape(M,(size(L,3),d,size(R,3)))

        @tensoropt fin[-1,-2,-3] := L[-1,4,5]*M[5,6,8]*W[4,7,-2,6]*R[-3,7,8]

        fin = reshape(fin,(size(L,1)*d*size(R,1)))

        return fin

    end
end

function MpoToMpsOneSiteKeff(M,L,R,object::String)

        """
         _ _ _ _                    _ _ _ _
        |       |---   ---M---  ---|       |
        |       |                  |       |
        |       |                  |       |
        |       |                  |       |
        |L_{i}  |---            ---|R_{i+1}|   --->  ~~~M'~~~
        |       |                  |       |
        |       |                  |       |
        |       |                  |       |
        |_ _ _ _|~~~            ~~~|_ _ _ _|

        """

    M = reshape(M,(size(L,3),size(R,3)))

    @tensoropt fin[-1,-2] := L[-1,3,4]*M[4,5]*R[-2,3,5]

    fin = reshape(fin,(size(L,1)*size(R,1)))

    return fin
end

function MpoToMpsTwoSite(M,L,W1,W2,R,d::Integer,object::String)

        """
         _ _ _ _                                     _ _ _ _
        |       |---  ---M_i---  ---M_{i+1}---   ---|       |
        |       |        |          |               |       |
        |       |                                   |       |
        |       |        |          |               |       |
        |L_{i-1}|---  ---W_i--   ---W_{i+1}--    ---|R_{i+2}|   --->  ~~~M'_i---M'{i+1}~~~
        |       |        |          |               |       |            |      |
        |       |                                   |       |
        |       |                                   |       |
        |_ _ _ _|~~~                             ~~~|_ _ _ _|

        """

    if object == "MPDO"

        M = reshape(M,(size(L,3),d,d,d,d,size(R,3)))

        @tensoropt fin[-1,-2,-3,-4,-5,-6] := L[-1,7,8]*M[8,9,-3,11,-5,13]*W1[7,10,-2,9]*W2[10,12,-4,11]*R[-6,12,13]

        fin = reshape(fin,(size(L,1)*d*d*d*d*size(R,1)))

        return fin

    elseif object == "MPS"

        M = reshape(M,(size(L,3),d,d,size(R,3)))

        @tensoropt fin[-1,-2,-3,-4] := L[-1,5,6]*M[6,7,9,11]*W1[5,8,-2,7]*W2[8,10,-3,9]*R[-4,10,11]

        fin = reshape(fin,(size(L,1)*d*d*size(R,1)))

        return fin
    end
end
