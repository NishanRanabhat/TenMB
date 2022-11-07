"""
This small function converts julia indexing system to python indexing system
by allowing us to access site 0. This could have been avoided by modifying the 
index system of our DMRG/TDVP routines by taking the first physical site at index 2,
however this is a more logical approach.
"""
function indx(i,N)
    if i == 0
        return N+1
    else
        return i
    end
end
