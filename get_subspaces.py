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
X_test = np.load('X_test.npy')
X_test,_,_ = standardise_minmax(X_test,min_value=min_X,max_value=max_X)

data_train = np.load('D.npy',allow_pickle=True)
data_test  = np.load('D_test.npy',allow_pickle=True)

num_pts     = data_train.shape[0]
num_designs = np.array([np.shape(point.D)[0] for point in data_train])
num_designs_test = np.array([np.shape(point.D)[0] for point in data_test])
num_bumps   = X_train.shape[1]
num_vars    = data_train[0].D.shape[1]
to_run = np.argwhere( (np.array(num_designs)>=cutoff) & (np.array(num_designs_test)>=cutoff) ) #Still check cutoff here as didn't check test in preproc
np.random.shuffle(to_run) # to_run index is shuffled in an attempt to load balance (since some spatial regions are more challenging to find grf for than others)
print('Max number of designs = %d' % np.max(num_designs))
print('Min number of designs = %d' % np.min(num_designs))
print('Number of nodes = %d' %(num_pts))
print('Number of nodes with num_designs > cutoff (=%d) = %d' %(cutoff,len(to_run)))
print('Number of bump functions = %d' %(num_bumps))
if max(vars) > num_vars-1: quit('Stopping: A vars index is greater than the number of arrays stored in D')


# Package up calc of subspaces at jth node into function
def get_subspaces(X_train,data_train, X_test, data_test, var,subdim=1,nattempts=3, maxiter=100, mingradnorm=1e-6, minstepsize=1e-8, maxd=None):
    indices = data_train.indices
    X_train = X_train[indices,:]
    y_train = data_train.D[:,var]
    indices_test = data_test.indices
    X_test = X_test[indices_test,:]
    y_test = data_test.D[:,var]

    if maxd is not None:
        index = np.random.randint(0,X_train.shape[0],maxd)
        X_train = X_train[index]
        y_train = y_train[index]
        X_test  = X_test[index]
        y_test  = y_test[index]

    # Further split training data into "train" and "validation" for grf fitting
    


    # Find Gaussian ridges
    warnings.filterwarnings("error")
    attempt = 0
    finish = False
    while not finish:
        attempt += 1
        if attempt > 1: print('r2 score = %.2f, trying grf algo again, attempt %d' %(r2_test,attempt))
        try:
            mygrf = grf(subdim=subdim,nattempts=nattempts,verbose=0,maxiter=maxiter,mingradnorm=mingradnorm,minstepsize=minstepsize,maxcostevals=5000)
            mygrf.fit(X_train, X_test, y_train, y_test, tol = 5e-4)

            M = mygrf.M
            u_train = X_train @ M
            u_test  = X_test @ M

            y_pred_train = mygrf.predict(u_train.reshape(-1,1),return_std=False)
            y_pred_test  = mygrf.predict(u_test.reshape(-1,1),return_std=False)

            # Get training r2 score and MAE at each point. 
            r2_train  = r2_score(y_train,y_pred_train)
            mae_train = mae_score(y_train,y_pred_train)
            r2_test   = r2_score(y_test,y_pred_test)
            mae_test  = mae_score(y_test,y_pred_test)
            if r2_test > 0.4: #If test score good enough, finish. (in practice, test should be validation set not test) 
                finish = True  

        except RuntimeWarning: 
            mygrf = None # Hopefully these will be overwritten with better results on next attempt!
            r2_train  = np.nan
            mae_train = np.nan
            r2_test  = np.nan
            mae_test = np.nan

        # If too many attempts, give up
        if attempt == 3:
            finish = True
            print('Giving up grf for this point')

    return (mygrf,r2_train,mae_train,r2_test,mae_test)

# Execute get_subspaces function in parallel with joblib
n_pre = 5
print('\nFinding subspaces at %d nodes, using %d threads...' % (len(to_run),n_jobs))
mygrf = np.empty([num_pts,num_vars],dtype='object')
for var in vars:
    print('\nVariable index %d...' % var)

    with parallel_backend("loky", inner_max_num_threads=2):
        results = Parallel(n_jobs=n_jobs,batch_size=1,pre_dispatch=int(n_pre*n_jobs),verbose=2)(delayed(get_subspaces)(X_train, data_train[j].item(), X_test, data_test[j].item(), var, subdim=subdim,
            nattempts=nattempts, maxiter=maxiter, mingradnorm=mingradnorm, minstepsize=minstepsize, maxd=None) for j in tqdm(to_run))

    mygrf[to_run,var] = np.array([item[0] for item in results]).reshape(-1,1)
    r2_train          = [item[1] for item in results]
    mae_train         = [item[2] for item in results]
    r2_test           = [item[3] for item in results]
    mae_test          = [item[4] for item in results]
    
    # Print average r2 score
    print('Average training r2 score = %.3f' %(np.nanmean(r2_train)))
    print('Average training mae      = %.4f' %(np.nanmean(mae_train)))
    print('Average test r2 score = %.3f' %(np.nanmean(r2_test)))
    print('Average test mae      = %.4f' %(np.nanmean(mae_test)))

# Write array of subspaces to file
saveloc = os.path.join(datadir,'SUBSPACE_DATA',casename)
os.makedirs(saveloc,exist_ok=True)
os.chdir(saveloc)
print('\nWriting to file...')
#np.save('SS.npy',mysubspaces)
with open("mygrf.pickle", "wb") as pickle_file:
    pickle.dump(mygrf,pickle_file)
