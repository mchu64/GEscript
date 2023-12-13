import numpy as np
import warnings

def swapRows(A, i, j):
    """
    interchange two rows of A
    operates on A in place
    """
    tmp = A[i].copy()
    A[i] = A[j]
    A[j] = tmp

def relError(a, b):
    """
    compute the relative error of a and b
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            return np.abs(a-b)/np.max(np.abs(np.array([a, b])))
        except:
            return 0.0

def rowReduce(A, i, j, pivot):
    """
    reduce row j using row i with pivot, in matrix A
    operates on A in place
    """
    factor = A[j][pivot] / A[i][pivot]
    for k in range(len(A[j])):
        if np.isclose(A[j][k], factor * A[i][k]):
            A[j][k] = 0.0
        else:
            A[j][k] = A[j][k] - factor * A[i][k]


# stage 1 (forward elimination)
def forwardElimination(B):
    """
    Return the row echelon form of B
    """
    A = B.copy().astype(float)
    m, n = np.shape(A)
    for i in range(m-1):
        # Let lefmostNonZeroCol be the position of the leftmost nonzero value 
        # in row i or any row below it 
        leftmostNonZeroRow = m
        leftmostNonZeroCol = n
        ## for each row below row i (including row i)
        for h in range(i,m):
            ## search, starting from the left, for the first nonzero
            for k in range(i,n):
                if (A[h][k] != 0.0) and (k < leftmostNonZeroCol):
                    leftmostNonZeroRow = h
                    leftmostNonZeroCol = k
                    break
        # if there is no such position, stop
        if leftmostNonZeroRow == m:
            break
        # If the leftmostNonZeroCol in row i is zero, swap this row 
        # with a row below it
        # to make that position nonzero. This creates a pivot in that position.
        if (leftmostNonZeroRow > i):
            swapRows(A, leftmostNonZeroRow, i)
        # Use row reduction operations to create zeros in all positions 
        # below the pivot.
        for h in range(i+1,m):
            rowReduce(A, i, h, leftmostNonZeroCol)
    return A

#################### 

# If any operation creates a row that is all zeros except the last element,
# the system is inconsistent; stop.
def inconsistentSystem(A):
    """
    B is assumed to be in echelon form; return True if it represents
    an inconsistent system, and False otherwise
    """
    total = 0
    nonzero = np.nonzero(A[-1])[0]
    nonzero = list(nonzero)
    for r in nonzero[:-1]:
        total += A[-1][r]
        
        
    if total == 0 and A[-1][-1] != 0:
        return True
    else:
        return False
    
def backsubstitution(B):
    """
    return the reduced row echelon form matrix of B
    """
    A = B.copy().astype(float)
    row, col = np.shape(A)
    for i in range(row):
        
        nonzero = np.nonzero(A[i])
        
        if len(nonzero[0]) > 0:
            pivot = nonzero[0][0]
            value = A[i][pivot]
            
            A[i] = A[i] / value
            for h in range(row):
                if h != pivot:
                    rowReduce(A,i,h,pivot)
    return A
        

#####################

def test():
    A = np.array([[0.90,0.01,0.09],[0.01,0.90,0.01],[0.09,0.09,0.90]])
    I = np.identity(3)
    B = A - I
    
    B = forwardElimination(B)
    if not inconsistentSystem(B):
        B = backsubstitution(B)
    else:
        return "inconsistent"
    x = np.array([-B[0][2], -B[1][2], 1])
    y = x[0] + x[1] + x[2]
    x = 1/y * x
    x = 2500 * x
    print(x)
    
