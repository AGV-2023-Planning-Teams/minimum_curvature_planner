"""_summary_
This file contains functions for generating the various Matrices described in the paper
"""

from perception_data import Centreline
import numpy as np

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

def matAInv(N: np.int32):
    A = np.zeros((4*N, 4*N))
    
    # Fill the matrix using vectorized operations
    indices = np.arange(0, 4*N, 4)
    A[indices, indices] = 1
    
    indices = np.arange(1, 4*N, 4)
    A[indices, indices-1] = 1
    A[indices, indices] = 1
    A[indices, indices+1] = 1
    A[indices, indices+2] = 1
    
    indices = np.arange(2, 4*N, 4)
    addr_A_N_1 = (indices + 4*N - 6) % (4*N)
    A[indices, indices-1] = -1
    A[indices, addr_A_N_1 + 1] = 1
    A[indices, addr_A_N_1 + 2] = 2
    A[indices, addr_A_N_1 + 3] = 3
    
    indices = np.arange(3, 4*N, 4)
    addr_A_N_1 = (indices + 4*N - 7) % (4*N)
    A[indices, indices-1] = -2
    A[indices, addr_A_N_1 + 2] = 2
    A[indices, addr_A_N_1 + 3] = 6
    
    A_inv = np.linalg.inv(A)
    A_inv[np.isclose(A_inv, 0, atol=1e-15)] = 0
    return A_inv

def A_ex_comp(N: np.int32, component: np.int32):
    A_ex = np.zeros((N, 4*N), dtype=np.float64)
    indices = np.arange(N)
    A_ex[indices, 4*indices + component] = 1
    return A_ex


def q_comp(centreline, component: np.int32):
    N = centreline.N
    q = np.zeros(4 * N, dtype=np.float64)
    indices = np.arange(N)
    q[4 * indices] = centreline.p[indices, component]
    q[4 * indices + 1] = centreline.p[(indices + 1) % N, component]
    
    return q


def M_comp(centreline, component: np.int32):
    N = centreline.N
    M = np.zeros((4 * N, N), dtype=np.float64)
    indices = np.arange(N)
    M[4 * indices, indices] = centreline.n[indices, component]
    M[4 * indices + 1, (indices + 1) % N] = centreline.n[(indices + 1) % N, component]
    
    return M

def first_derivatives(centreline: Centreline, Ainv: np.ndarray, q: np.ndarray):
    A_ex_b = A_ex_comp(centreline.N, 1)
    return A_ex_b @ Ainv @ q
import numpy as np

def matPxx(x_dashed: np.ndarray, y_dashed: np.ndarray):
    values = y_dashed**2 / (x_dashed**2 + y_dashed**2)**3
    return np.diag(values)

def matPxy(x_dashed: np.ndarray, y_dashed: np.ndarray):
    values = -2 * x_dashed * y_dashed / (x_dashed**2 + y_dashed**2)**3
    return np.diag(values)

def matPyy(x_dashed: np.ndarray, y_dashed: np.ndarray):
    values = x_dashed**2 / (x_dashed**2 + y_dashed**2)**3
    return np.diag(values)

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
