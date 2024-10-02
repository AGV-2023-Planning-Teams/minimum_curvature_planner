"""_summary_
This file contains functions for generating the various Matrices described in the paper
"""

from perception_data import Centreline
from numba import cuda

from import_cp import cp

# def matAInv(N: np.int32):
#     A = np.zeros((4*N, 4*N))
#     for i in range(0, 4*N, 4):
#         # Equation: x_n = a_n
#         # i is address of a_n
#         A[i][i] = 1 # coeff of a_n
#     for i in range(1, 4*N, 4):
#         # Equation: x_(n+1) = a_n + b_n + c_n + d_n
#         # i is address of b_n
#         A[i][i-1] = 1 # coeff of a_n
#         A[i][i] = 1 # coeff of b_n
#         A[i][i+1] = 1 # coeff of c_n
#         A[i][i+2] = 1 # coeff of d_n
#     for i in range(2, 4*N, 4):
#         # Equation: 0 = x_1' - x_1' = b_(n-1) + 2c_(n-1) + 3d_(n-1) - b_n
#         # i is address of c_n
#         A[i][i-1] = -1 # coeff of b_n
#         addr_A_N_1 = (i+4*N-6)%(4*N) # address of a_(n-1)
#         A[i][addr_A_N_1 + 1] = 1 # coeff of b_(n-1)
#         A[i][addr_A_N_1 + 2] = 2 # coeff of c_(n-1)
#         A[i][addr_A_N_1 + 3] = 3 # coeff of d_(n-1)
#     for i in range(3, 4*N, 4):
#         # Equation: 0 = x_1'' - x_1'' = 2c_(n-1) + 6d_(n-1) - 2c_n
#         # i is address of d_n
#         A[i][i-1] = -2 # coeff of c_n
#         addr_A_N_1 = (i+4*N-7)%(4*N) # address of a_(n-1)
#         A[i][addr_A_N_1 + 2] = 2 # coeff of c_(n-1)
#         A[i][addr_A_N_1 + 3] = 6 # coeff of d_(n-1)
#     A_inv = np.linalg.inv(A)
#     A_inv[np.isclose(A_inv, 0, atol=1e-15)] = 0
#     return A_inv

@cuda.jit
def fill_A(A: cp.ndarray, N: cp.uint32) -> None:
    i, j = cuda.grid(2)
    if i < 4*N and j < 4*N:
        if (i % 4 == 0):
            if j == i: val = 1
            else: val = 0
        elif (i % 4 == 1):
            if j == i-1: val = 1
            elif j == i: val = 1
            elif j == i+1: val = 1
            elif j == i+2: val = 1
            else: val = 0
        elif (i % 4 == 2):
            addr_A_N_1 = (i + 4*N - 6) % (4*N)
            if j == i-1: val = -1
            elif j == addr_A_N_1 + 1: val = 1
            elif j == addr_A_N_1 + 2: val = 2
            elif j == addr_A_N_1 + 3: val = 3
            else: val = 0
        elif (i % 4 == 3):
            addr_A_N_1 = (i + 4*N - 7) % (4*N)
            if j == i-1: val = -2
            elif j == addr_A_N_1 + 2: val = 2
            elif j == addr_A_N_1 + 3: val = 6
            else: val = 0
        A[i, j] = val

def matAInv(N: cp.uint32) -> cp.ndarray:
    A = cp.zeros((4*N, 4*N), dtype=cp.float32)
    fill_A[((4*N+31)//32, (4*N+31)//32), (32, 32)](A, N)
    A_inv = cp.linalg.inv(A)
    A_inv[cp.isclose(A_inv, 0, atol=1e-15)] = 0
    return A_inv

@cuda.jit
def fill_A_ex_comp(A_ex_comp: cp.ndarray, N: cp.uint32, component: cp.uint32) -> None:
    i, j = cuda.grid(2)
    if i < N and j < 4*N: A_ex_comp[i, j] = (j == 4*i + component)

def A_ex_comp(N: cp.uint32, component: cp.uint32) -> cp.ndarray:
    A_ex = cp.ndarray((N, 4*N), dtype=cp.float32)
    fill_A_ex_comp[((N+31)//32, (4*N+31)//32), (32, 32)](A_ex, N, component)
    return A_ex

@cuda.jit
def fill_q_comp(q: cp.ndarray, p: cp.ndarray, component: cp.uint32) -> None:
    N = p.shape[0]
    i = cuda.grid(1)
    if i < 4*N:
        if (i % 4 == 0): q[i] = p[i//4, component]
        elif (i % 4 == 1): q[i] = p[(i//4 + 1) % N, component]
        elif (i % 4 == 2) or (i % 4 == 3): q[i] = 0

def q_comp(centreline: Centreline, component: cp.int32) -> cp.ndarray:
    N = centreline.N
    q = cp.ndarray(4 * N, dtype=cp.float32)
    fill_q_comp[(4*N+31)//32, 32](q, centreline.p, component)
    return q

@cuda.jit
def fill_M_comp(M: cp.ndarray, n: cp.ndarray, component: cp.int32) -> None:
    N = n.shape[0]
    i, j = cuda.grid(2)
    if i < 4*N and j < N:
        if (i % 4 == 0):
            M[i, j] = (n[j, component] if (j == i//4) else 0)
        if (i % 4 == 1):
            M[i, j] = (n[j, component] if (j == (i//4 + 1)%N) else 0)
        if (i % 4 == 2) or (i % 4 == 3):
            M[i, j] = 0

def M_comp(centreline: Centreline, component: cp.int32):
    N = centreline.N
    M = cp.ndarray((4 * N, N), dtype=cp.float32)
    fill_M_comp[((4*N+31)//32, (N+31)//32), (32, 32)](M, centreline.n, component)
    
    return M

def first_derivatives(centreline: Centreline, Ainv: cp.ndarray, q: cp.ndarray):
    A_ex_b = A_ex_comp(centreline.N, 1)
    return A_ex_b @ Ainv @ q

def matPxx(x_dashed: cp.ndarray, y_dashed: cp.ndarray):
    values = y_dashed**2 / (x_dashed**2 + y_dashed**2)**3
    return cp.diag(values)

def matPxy(x_dashed: cp.ndarray, y_dashed: cp.ndarray):
    values = -2 * x_dashed * y_dashed / (x_dashed**2 + y_dashed**2)**3
    return cp.diag(values)

def matPyy(x_dashed: cp.ndarray, y_dashed: cp.ndarray):
    values = x_dashed**2 / (x_dashed**2 + y_dashed**2)**3
    return cp.diag(values)

def matrices_H_f(centreline: Centreline):
    # returns a tuple of the matrices H and f that define the QP
    Ainv = matAInv(centreline.N)
    A_ex_c = A_ex_comp(centreline.N, 2)
    q_x = q_comp(centreline, 0)
    q_y = q_comp(centreline, 1)
    x_d = first_derivatives(centreline, Ainv, q_x) # vector containing x_i'
    y_d = first_derivatives(centreline, Ainv, q_y) # vector containing y_i'

    centreline.calc_n(x_d, y_d)

    M_x = M_comp(centreline, 0)
    M_y = M_comp(centreline, 1)

    T_c = 2 * A_ex_c @ Ainv
    T_n_x = T_c @ M_x
    T_n_y = T_c @ M_y
    
    P_xx = matPxx(x_d, y_d)
    P_xy = matPxy(x_d, y_d)
    P_yy = matPyy(x_d, y_d)

    H_x = T_n_x.T @ P_xx @ T_n_x
    H_xy = T_n_y.T @ P_xy @ T_n_x
    H_y = T_n_y.T @ P_yy @ T_n_y
    H = 2*(H_x + H_xy + H_y)

    f_x = 2 * T_n_x.T @ P_xx.T @ T_c @ q_x
    f_xy = T_n_y.T @ P_xy.T @ T_c @ q_x + T_n_x.T @ P_xy.T @ T_c @ q_y
    f_y = 2 * T_n_y.T @ P_yy.T @ T_c @ q_y
    f = f_x + f_xy + f_y
    
    return H, f
