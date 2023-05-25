using LinearAlgebra
using TensorOperations

"""
Functions to calculate one and two site observables in iMPS
"""

function inner_prod(inf_MPS::iMPS)

    """
    calculates inner product of iMPS
    """
    
    indx_right = size(inf_MPS.Be)[3]
    indx_left  = size(inf_MPS.L)[1]
    
    R = Matrix{Float64}(I,indx_right,indx_right)
    
    @tensoropt theta[-1,-2,-3,-4] := inf_MPS.L[-1,5]*inf_MPS.Bo[5,-2,6]*inf_MPS.Be[6,-3,-4]
    @tensoropt R[-1,-2] := conj(theta)[-1,3,4,5]*theta[-2,3,4,6]*R[5,6]
    @tensoropt val[] := Matrix{Float64}(I,indx_left,indx_left)[1,2]*R[1,2]
    
    return val
end

function one_site_expect(inf_MPS::iMPS,O::Array{})

    """
    calculates expectation of single site operator over iMPS
    """
    
    indx_right = size(inf_MPS.Be)[3]
    indx_left  = size(inf_MPS.L)[1]
    
    R = Matrix{Float64}(I,indx_right,indx_right)
    
    @tensoropt theta[-1,-2,-3,-4] := inf_MPS.L[-1,5]*inf_MPS.Bo[5,-2,6]*inf_MPS.Be[6,-3,-4]
    @tensoropt R[-1,-2] := conj(theta)[-1,3,4,5]*O[3,6]*theta[-2,6,4,8]*R[5,8]
    @tensoropt val[] := Matrix{Float64}(I,indx_left,indx_left)[1,2]*R[1,2]
    
    return val
end   

function two_site_expect(inf_MPS::iMPS,O::Array{})

    """
    calculates expectation of two site operator over iMPS
    """
        
    indx_right = size(inf_MPS.Be)[3]
    indx_left  = size(inf_MPS.L)[1]
    
    R = Matrix{Float64}(I,indx_right,indx_right)
    
    @tensoropt theta[-1,-2,-3,-4] := inf_MPS.L[-1,5]*inf_MPS.Bo[5,-2,6]*inf_MPS.Be[6,-3,-4]
    @tensoropt R[-1,-2] := conj(theta)[-1,3,4,7]*O[3,4,5,6]*theta[-2,5,6,8]*R[7,8]
    @tensoropt val[] := Matrix{Float64}(I,indx_left,indx_left)[1,2]*R[1,2]
    
    return val
end 