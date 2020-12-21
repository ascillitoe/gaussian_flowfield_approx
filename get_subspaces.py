import os
import sys
import json
import numpy as np
from tqdm import tqdm
from funcs import r2_score, mae_score, datapoint, standardise_minmax
from joblib import Parallel, delayed, cpu_count, parallel_backend
from grf import grf
import pickle
import warnings

global seed
seed = 42

basedir = os.getcwd()

# Read json file
inputfile = sys.argv[1]
with open(inputfile) as json_file:
    json_dat = json.load(json_file)

# Parse json options
casename        = json_dat['casename']
datadir         = json_dat['datadir']
if datadir == '.':
    datadir = basedir
vars = json_dat['vars'] #list of vars indexes to find subspaces for. (See ordering in proc_dat.py)
subdim = json_dat['subdim']
n_jobs = json_dat['njobs']
cutoff = json_dat['cutoff']

nattempts = 3
if (("nattempts" in json_dat)==True): nattempts = json_dat['nattempts']

maxiter = 20
if (("maxiter" in json_dat)==True): maxiter = json_dat['maxiter']

mingradnorm = 1e-6
if (("mingradnorm" in json_dat)==True): mingradnorm = json_dat['mingradnorm']

minstepsize = 1e-8
if (("minstepsize" in json_dat)==True): minstepsize = json_dat['minstepsize']

maxd = None
if (("maxd" in json_dat)==True): maxd = json_dat['maxd']

# Housekeeping
ncores = cpu_count()
if(n_jobs==-1): n_jobs = ncores
if (n_jobs>ncores): quit('STOPPING: n_jobs > available cores')

# Read in numpy input and output arrays
dataloc = os.path.join(datadir,'PROCESSED_DATA',casename)
os.chdir(dataloc)
X_train = np.load('X.npy')
X_train, min_X, max_X = standardise_minmax(X_train)

data_train = np.load('D.npy',allow_pickle=True)

num_pts     = data_train.shape[0]
num_designs = np.array([np.shape(point.D)[0] for point in data_train])
num_bumps   = X_train.shape[1]
num_vars    = data_train[0].D.shape[1]
to_run = np.argwhere( (np.array(num_designs)>=cutoff) )
np.random.shuffle(to_run) # to_run index is shuffled in an attempt to load balance (since some spatial regions are more challenging to find grf for than others)
print('Max number of designs = %d' % np.max(num_designs))
print('Min number of designs = %d' % np.min(num_designs))
print('Number of nodes = %d' %(num_pts))
print('Number of nodes with num_designs > cutoff (=%d) = %d' %(cutoff,len(to_run)))
print('Number of bump functions = %d' %(num_bumps))
if max(vars) > num_vars-1: quit('Stopping: A vars index is greater than the number of arrays stored in D')


# Package up calc of subspaces at jth node into function
def get_subspaces(X,data, var,subdim=1,nattempts=3, maxiter=100, mingradnorm=1e-6, minstepsize=1e-8, maxd=None):
    global seed
    np.random.seed(seed)

    indices = data.indices
    X = X[indices,:]
    y = data.D[:,var]

    if maxd is not None:
        index = np.random.randint(0,X.shape[0],maxd)
        X = X[index]
        y = y[index]

    # Further split training data into "train" and "validation" for grf fitting (with 70/30 split)
    N = X.shape[0]
    N_train = int(0.7*N)
    idx = np.arange(N)
    np.random.shuffle(idx)
    idx_train, idx_valid = idx[:N_train], idx[N_train:]
    X_train = X[idx_train]
    y_train = y[idx_train]
    X_valid = X[idx_valid]
    y_valid = y[idx_valid]

    # Find Gaussian ridges
    warnings.filterwarnings("error")
    attempt = 0
    finish = False
    while not finish:
        attempt += 1
        if attempt > 1: print('r2 score = %.2f, trying grf algo again, attempt %d' %(r2_valid,attempt))
        try:
            mygrf = grf(subdim=subdim,nattempts=nattempts,verbose=0,maxiter=maxiter,mingradnorm=mingradnorm,minstepsize=minstepsize,maxcostevals=5000)
            mygrf.fit(X_train, X_valid, y_train, y_valid, tol = 5e-4)

            M = mygrf.M
            u_train = X_train @ M
            u_valid = X_valid @ M

            y_pred_train = mygrf.predict(u_train.reshape(-1,1),return_std=False)
            y_pred_valid = mygrf.predict(u_valid.reshape(-1,1),return_std=False)

            # Get training r2 score and MAE at each point. 
            r2_train  = r2_score(y_train,y_pred_train)
            mae_train = mae_score(y_train,y_pred_train)
            r2_valid  = r2_score(y_valid,y_pred_valid)
            mae_valid = mae_score(y_valid,y_pred_valid)
            if r2_valid > 0.4: #If validation score good enough, finish.
                finish = True  

        except RuntimeWarning: 
            mygrf = None # Hopefully these will be overwritten with better results on next attempt!
            r2_train  = np.nan
            mae_train = np.nan
            r2_valid  = np.nan
            mae_valid = np.nan

        # If too many attempts, give up
        if attempt == 3:
            finish = True
            print('Giving up grf for this point')

    return (mygrf,r2_train,mae_train,r2_valid,mae_valid)

# Execute get_subspaces function in parallel with joblib
n_pre = 5
print('\nFinding subspaces at %d nodes, using %d threads...' % (len(to_run),n_jobs))
mygrf = np.empty([num_pts,num_vars],dtype='object')
for var in vars:
    print('\nVariable index %d...' % var)

    with parallel_backend("loky", inner_max_num_threads=2):
        results = Parallel(n_jobs=n_jobs,batch_size=1,pre_dispatch=int(n_pre*n_jobs),verbose=2)(delayed(get_subspaces)(X_train, data_train[j].item(), var, subdim=subdim,
            nattempts=nattempts, maxiter=maxiter, mingradnorm=mingradnorm, minstepsize=minstepsize, maxd=maxd) for j in tqdm(to_run))

    mygrf[to_run,var] = np.array([item[0] for item in results]).reshape(-1,1)
    r2_train          = [item[1] for item in results]
    mae_train         = [item[2] for item in results]
    r2_valid          = [item[3] for item in results]
    mae_valid         = [item[4] for item in results]
    
    # Print average r2 score
    print('Average training r2 score = %.3f' %(np.nanmean(r2_train)))
    print('Average training mae      = %.4f' %(np.nanmean(mae_train)))
    print('Average validation r2 score = %.3f' %(np.nanmean(r2_valid)))
    print('Average validation mae      = %.4f' %(np.nanmean(mae_valid)))

# Write array of subspaces to file
saveloc = os.path.join(datadir,'SUBSPACE_DATA',casename)
os.makedirs(saveloc,exist_ok=True)
os.chdir(saveloc)
print('\nWriting to file...')
#np.save('SS.npy',mysubspaces)
with open("mygrf.pickle", "wb") as pickle_file:
    pickle.dump(mygrf,pickle_file)
