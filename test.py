import sympy as sym
from sympy import *

def symbolize_matrix(matrix_prefix="k"):
    """Symbolize a matrix with the given prefix.
    
    Parameters
    ----------
    matrix_prefix : str
        The prefix to use for the matrix.
        
    Returns
    -------
    A : sympy.Matrix
        The symbolized matrix.
    """
    A = sym.Matrix([
        [sym.symbols(f"{matrix_prefix}_00"), sym.symbols(f"{matrix_prefix}_01"), sym.symbols(f"{matrix_prefix}_02")],
        [sym.symbols(f"{matrix_prefix}_10"), sym.symbols(f"{matrix_prefix}_11"), sym.symbols(f"{matrix_prefix}_12")],
        [sym.symbols(f"{matrix_prefix}_20"), sym.symbols(f"{matrix_prefix}_21"), sym.symbols(f"{matrix_prefix}_22")]
    ])
    return A

A = symbolize_matrix("a")
D = symbolize_matrix("P") 

B=A*(A.T)

DA = []
for r in range(3):
    for c in range(3):
        DA.append(0)
        for i in range(3):
            for j in range(3):
                DA[-1] += diff(B[i, j], A[r,c]) * D[i, j]
print(DA)

