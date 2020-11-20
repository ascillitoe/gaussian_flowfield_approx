import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import ConjugateGradient
from pymanopt import Problem
from funcs import standardise_minmax, unstandardise_minmax

class grf:
    def __init__(self,subdim=1,verbose=0,maxtime=10,maxiter=100,mingradnorm=1e-6,minstepsize=1e-10,maxcostevals=5000,nattempts=3):
        self.verbose = verbose
        self.subdim  = subdim
        self.maxtime = maxtime
        self.maxiter = maxiter
        self.mingradnorm  = mingradnorm
        self.minstepsize  = minstepsize
        self.maxcostevals = maxcostevals
        self.nattempts    = nattempts

    def fit(self,X_train, X_test, f_train, f_test, M0 = None, tol = 1e-5):
        """
        Fits a gaussian ridge function.
        This code is based upon the grf_fit() function from https://github.com/psesh/Gaussian-Ridges.
        See https://doi.org/10.1137/18M1168571.
        """

        # Standardise targets
        f_train, min_f, max_f = standardise_minmax(f_train.reshape(-1,1))
        f_train = np.squeeze(f_train)
        f_test, *_ = standardise_minmax(f_test.reshape(-1,1),min_value=min_f,max_value=max_f)
        f_test = np.squeeze(f_test)
        self.scaling = [min_f,max_f]

        attempts = 0
        success  = False
        while attempts < self.nattempts and not success:
            if self.verbose>0: print('Attempt %d' %attempts)

            # Initial guess
            if M0 is None:
                d = np.shape(X_train)[1]
                M0 = np.random.randn(d,self.subdim)
                Q = np.linalg.qr(M0)[0]
                M0 = Q.copy()
            last_r = 1.0
            err = 1.0
            M_guess = M0.copy()
            d, m = M0.shape
    
            # Fit gaussian ridge function
            out_iter = 0
            in_iter  = []
            fx       = []
            n_iter   = []
            gradnorm = []
            final_gradnorm = []
            while err > tol:
                if self.verbose>0: print('Outer iteration %d' %(out_iter+1))
                U_train = X_train @ M_guess
                ker = 1.0 * RBF(length_scale=[1 for _ in range(m)]) + WhiteKernel(noise_level=1.0)
                gpr = GaussianProcessRegressor(kernel=ker, n_restarts_optimizer=10,alpha=.1)
                gpr.fit(U_train, f_train)
    
                kernel = gpr.kernel_
    
                my_cost = lambda M: cost(M, X_train, X_test, f_train, f_test, kernel)
                my_dcost = lambda M: dcost(M, X_train, X_test, f_train, f_test, kernel)
    
                manifold = Stiefel(d, m)
                problem = Problem(manifold=manifold, cost=my_cost, grad=my_dcost, verbosity=self.verbose)
                solver = ConjugateGradient(logverbosity=2,maxtime=self.maxtime, maxiter=self.maxiter,
                        mingradnorm=self.mingradnorm,minstepsize=self.minstepsize,maxcostevals=self.maxcostevals)
                solver_results = solver.solve(problem, x=M_guess)
                M_new = solver_results[0]
                M_guess = M_new.copy()

                r = cost(M_guess, X_train, X_test, f_train, f_test, kernel)
    
                err = np.abs(last_r - r) / r
                last_r = r

                # Store convergence info
                log   = solver_results[1]
                n_iter.append(log['final_values']['iterations'])
                final_gradnorm.append(log['final_values']['gradnorm'])
                in_iter.append(log['iterations']['iteration'])
                fx.append(log['iterations']['f(x)'])
                gradnorm.append(log['iterations']['gradnorm'])
                out_iter += 1
            
                # If break conditions is mingradnorm one (on first iter), then redo from M_guess
                reason = log['stoppingreason'] 
                if 'min grad norm' in reason:
                    if self.verbose>0: print(reason + ', restarting with new M0')
                    success = False
                    M0 = None
                    break
                else:
                    success = True
            attempts += 1

        self.log = {'n_out_iter': out_iter,'in_iter': in_iter,'fx':fx,'grad_norm':gradnorm,
                    'n_in_iter': n_iter, 'final_grad_norm':final_gradnorm, 'err':err, 'tol':tol,
                'attempts':attempts,'stoppingreason':reason}

        # Store M, GP model 
        self.M     = M_guess
        self.model = gpr 

    def predict(self,X,return_std=False):
        
        # If X given instead of U, get U from X
        if X.shape[1]!=self.subdim:
            X = X @ self.M

        # Prediction
        result = self.model.predict(X,return_std=return_std)
            
        # Rescale
        
        if return_std:
            mean = unstandardise_minmax(result[0].reshape(-1,1),self.scaling[0],self.scaling[1])
            std = result[1]*np.sqrt(0.5)*(self.scaling[1]-self.scaling[0])
                        #unstandardise_minmax(result[1].reshape(-1,1),self.scaling[0],self.scaling[1]).squeeze()
            return mean.squeeze(), std.squeeze()
        else:
            mean = unstandardise_minmax(result.reshape(-1,1),self.scaling[0],self.scaling[1])
            return mean.squeeze()

def cost(M_guess, X_train, X_test, f_train, f_test, kernel):

    U_train = X_train @ M_guess
    U_test = X_test @ M_guess

    G = kernel(U_train)
    b = np.linalg.solve(G, f_train)

    K_test = kernel(U_test, U_train)
    g_test = K_test @ b

    r = 0.5 * np.linalg.norm(f_test - g_test)**2
    return r

def dcost(M_guess, X_train, X_test, f_train, f_test, kernel):

    ell = kernel.get_params()['k1__k2__length_scale']
    U_train = X_train @ M_guess
    U_test = X_test @ M_guess
    N_test = X_test.shape[0]

    G = kernel(U_train)
    b = np.linalg.solve(G, f_train)
    K_test = kernel(U_test, U_train)
    g_test = K_test @ b

    if ell.ndim==0: ell = np.array([ell])
    inv_P = np.diag(1.0/ell**2)
    dr = np.zeros(M_guess.shape)
    for i in range(N_test):
        U_tilde = U_test[i] - U_train
        dgdu = inv_P @ U_tilde.T @ (K_test[i,:] * b)
        dy = np.outer(dgdu, X_test[i,:]).T
        assert(dy.shape == M_guess.shape)
        dr += (f_test[i] - g_test[i]) * (dy - M_guess @ dy.T @ M_guess)

    return dr


