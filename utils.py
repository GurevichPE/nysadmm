import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log
from sklearn.metrics import mean_squared_error as mse 
from scipy.linalg import cholesky, svd, solve_triangular
from time import time



def Sthresh(x, gamma):
    return np.sign(x)*np.maximum(0, np.absolute(x)-gamma/2.0)

def ADMM(A, y, A_test=None, y_test=None, MAX_ITER = 100_000):
    lossses = []
    m, n = A.shape
    w, v = np.linalg.eig(A.T.dot(A))
    

    "Function to caluculate min 1/2(y - Ax) + l||x||"
    "via alternating direction methods"
    xhat = np.zeros([n, 1])
    zhat = np.zeros([n, 1])
    u = np.zeros([n, 1])

    "Calculate regression co-efficient and stepsize"
    l = sqrt(2*log(n, 10))
    rho = 1/(np.amax(np.absolute(w)))

    "Pre-compute to save some multiplications"
    AtA = A.T.dot(A)
    Aty = A.T.dot(y)
    Q = AtA + rho*np.identity(n)
    Q = np.linalg.inv(Q)

    i = 0
    times = []
    curr_time = 0

    while(i < MAX_ITER):
        t0 = time()
        "x minimisation step via posterier OLS"
        xhat = Q.dot(Aty + rho*(zhat - u))

        "z minimisation via soft-thresholding"
        zhat = Sthresh(xhat + u, l/rho)

        "mulitplier update"
        u += xhat - zhat

        i += 1
        if A_test is not None:
            lossses.append(mse(y_test, A_test@zhat))
        else:
            lossses.append(mse(y, A@zhat))
        t1 = time()
        times.append(curr_time + t1 - t0)
        curr_time = times[-1]
    return zhat, rho, l, lossses, times


epsilon = lambda x: 1 / (1 + x ** 7)
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def nystrom_approximation(H, s, eps=epsilon):
    print(is_pos_def(H))
    d = H.shape[0]
    Omega = np.random.normal(0, 1, (d, s)) # Gaussian test matrix
    Omega, _ = np.linalg.qr(Omega, mode='reduced')
    Y = H @ Omega
    nu = eps(np.linalg.norm(Y, 2))
    Y_nu = Y + nu * Omega
    print(is_pos_def(Omega.T @ Y_nu))
    C = cholesky(Omega.T @ Y_nu)
    B = solve_triangular(C, Y_nu.T, lower=False, trans='T').T 
    U, Sigma, _ = svd(B, full_matrices=False)
    Id = np.ones(Sigma.shape)
    lambda_prime = np.maximum(0, Sigma ** 2 - nu * Id)
    return U, np.diag(lambda_prime)

# from lecture
def accurate_randomized_svd(A, rank, p, q):
    m, n = A.shape
    G = np.random.randn(n, rank + p)
    Y = A @ G
    Q, _ = np.linalg.qr(Y)
    for i in range(q):
        W = A.T @ Q
        W, _ = np.linalg.qr(W)
        Q = A @ W
        Q, _ = np.linalg.qr(Q)
    B = Q.T @ A
    u, S, V = np.linalg.svd(B)
    U = Q @ u
    return U, S, V


eps_k = lambda k: 1 / (1 + k**2)

def inverse_diagonal_matrix(D):
    diagonal_elements = D.diagonal()
    D_inv = np.diag(1 / diagonal_elements)
    return D_inv

def get_preconditioner_inv(U, S, s, ro):
    diag_elements = S.diagonal()
    lambda_s = diag_elements[-1]
    I = np.eye(S.shape[0])
    D = S + ro * I
    D_inv = inverse_diagonal_matrix(D)
    P_inv = (lambda_s + ro) * U @ D_inv @ U.T + (np.eye(U.shape[0]) - U @ U.T)
    return P_inv

def nystrom_pcg(H, U, S, r, x0, ro=0.01, s=50, epsilon=0.05):
    #U, S = nystrom_approximation(H, s)
    
    I = np.eye(H.shape[0])
    w0 = r - (H + ro * I) @ x0

    P_inv = get_preconditioner_inv(U, S, s, ro)

    y0 = P_inv @ w0
    p0 = y0.copy()
    x = x0
    while np.linalg.norm(w0) > epsilon:
        v = (H + ro * I) @ p0
        alpha = (w0.T @ y0) / (p0.T @ v)
        x = x0 + alpha * p0
        w = w0 - alpha * v
        y = P_inv @ w
        beta = (w.T @ y) / (w0.T @ y0)
        x0, w0, p0, y0 = x, w, y + beta * p0, y

    return x  # Output approximate solution x*



def NysADMM(A, y, A_test=None, y_test=None, MAX_ITER = 100_000):
    lossses = []
    m, n = A.shape
    w, v = np.linalg.eig(A.T.dot(A))
    
    xhat = np.zeros([n, 1])
    zhat = np.zeros([n, 1])
    u = np.zeros([n, 1])

    l = sqrt(2*log(n, 10))
    rho = 1/(np.amax(np.absolute(w)))

    AtA = A.T.dot(A)
    Aty = A.T.dot(y)
    Q = AtA + rho*np.identity(n)
   

    U, S = nystrom_approximation(Q, s=50)
    times = []
    curr_time = 0

    for _ in range(MAX_ITER):

        t0 = time()
        r = Aty
        xhat = nystrom_pcg(Q.copy(), U, S, r, rho*(zhat - u), ro=rho, s=50, epsilon=0.1)

        zhat = Sthresh(xhat + u, l/rho)


        u += xhat - zhat

        if A_test is not None:
            lossses.append(mse(y_test, A_test@zhat))
        else:
            lossses.append(mse(y, A@zhat))
        t1 = time()
        times.append(curr_time + t1 - t0)
        curr_time = times[-1]

        
    return zhat, rho, l, lossses, times