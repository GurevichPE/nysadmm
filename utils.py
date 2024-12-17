import numpy as np
from sklearn.metrics import mean_squared_error as mse 
from scipy.linalg import cholesky, svd, solve_triangular
from time import time



def Sthresh(x, gamma):
    """Calculates soft thresholding"""
    return np.sign(x)*np.maximum(0, np.absolute(x)-gamma/2.0)


def ADMM(A, y, A_test=None, y_test=None, MAX_ITER = 100_000):
    """
    Calculates ADMM. 
    In:
    A: np.array -- feature matrix
    y: np.array -- response
    A_test, y_test -- feature matrix and response for testing
    MAX_ITER -- number of iterations

    Returns:
    zhat -- solution of minimization problem
    rho -- step size
    l -- regularization coefficient
    losses -- list of MSE values for each iteration. If A_test=None, losses are for MSE(A@x, y)
                else, losses are MSE(A_test@x, y_test)
    times -- list of time values for each iteration
    """
    lossses = []
    _, n = A.shape
    w, _ = np.linalg.eig(A.T.dot(A))
    
    xhat = np.zeros([n, 1])
    zhat = np.zeros([n, 1])
    u = np.zeros([n, 1])

    "Calculate regression coefficient and stepsize"
    l = np.sqrt(2*np.log10(n))
    rho = 1/(np.amax(np.absolute(w)))


    AtA = A.T.dot(A)
    Aty = A.T.dot(y)
    Q = AtA + rho*np.identity(n)
    Q = np.linalg.inv(Q)

    times = []
    curr_time = 0

    for _ in range(MAX_ITER):
        t0 = time()
        "x minimisation step via posterier OLS"
        xhat = Q.dot(Aty + rho*(zhat - u))

        "z minimisation via soft-thresholding"
        zhat = Sthresh(xhat + u, l/rho)

        "mulitplier update"
        u += xhat - zhat

        if A_test is not None:
            lossses.append(mse(y_test, A_test@zhat))
        else:
            lossses.append(mse(y, A@zhat))
        t1 = time()
        times.append(curr_time + t1 - t0)
        curr_time = times[-1]

    return zhat, rho, l, lossses, times


epsilon = lambda x: 1 / (1 + x ** 7)


def nystrom_approximation(H, s, eps=epsilon):
    """
    H: np.array -- SPD matrix
    s: int -- sketch size
    eps: shift

    returns:
    U, S, such that H \\approx U@S@U.T 
    """
    d = H.shape[0]
    Omega = np.random.normal(0, 1, (d, s)) # Gaussian test matrix
    Omega, _ = np.linalg.qr(Omega, mode='reduced')
    Y = H @ Omega
    nu = eps(np.linalg.norm(Y, 2))
    Y_nu = Y + nu * Omega
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
    """Performs diagonal matrix inversion"""
    diagonal_elements = D.diagonal()
    D_inv = np.diag(1 / diagonal_elements)
    return D_inv


def get_preconditioner_inv(U, S, s, ro):
    """Calculates preconditioner P^{-1}"""
    diag_elements = S.diagonal()
    lambda_s = diag_elements[-1]
    I = np.eye(S.shape[0])
    D = S + ro * I
    D_inv = inverse_diagonal_matrix(D)
    P_inv = (lambda_s + ro) * U @ D_inv @ U.T + (np.eye(U.shape[0]) - U @ U.T)
    return P_inv


def nystrom_pcg(H, U, S, r, x0, ro=0.01, s=50, epsilon=0.05):
    """
    Performs Nystrom Preconitioned Conjugate Gradients algorithm

    H -- SPD matrix
    U, S -- Nystrom approximation of H
    r -- righthandside residual (for our task it is A.T@y)
    x0 -- starting point for solution
    ro -- stepsize
    s -- sketch size
    epsilon -- shift

    returns:
    x -- solution
    """

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



def NysADMM(A, y, A_test=None, y_test=None, MAX_ITER=1_000):
    """
    Calculates Nystrom ADMM
    Parameters and outputs are the same as for ADMM function
    """
    lossses = []
    _, n = A.shape
    w, _ = np.linalg.eig(A.T.dot(A))
    
    xhat = np.zeros([n, 1])
    zhat = np.zeros([n, 1])
    u = np.zeros([n, 1])

    l = np.sqrt(2*np.log10(n))
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