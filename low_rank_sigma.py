import os
import sys
import json
import numpy as np
import pyvista as pv
from tqdm import tqdm

#################
# Settings
#################
LR = 20
vars = [0]

#####################
# Setup stuff 
#####################
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

dataloc = os.path.join(datadir,'PROCESSED_DATA',casename)

for var in vars:
    # Read in eigenvalues
    eig = np.load(os.path.join(dataloc,'eigvals_%d.npy' %var))
    
    # Read in eigenvectors
    V = np.load(os.path.join(dataloc,'eigvecs_%d.npy' %var))
    
    ##########################################################
    # Rebuild Sigma with LR no. of largest eigenvalues/vectors
    ##########################################################
    # Sort by descreasing eigenvalue
    idx_eig = eig.argsort()[::-1]
    
    # Extract LR largest eigenvalues and vectors
    Lambda = np.diag(eig[idx_eig[:LR]])
    Q = V[:,idx_eig[:LR]]
    
    # Rebuild Sigma
    Sigma_LR = Q @ Lambda @ Q.T
    
    ## Save low rank approx of Sigma
    savefile = os.path.join(dataloc,'covar_lr_%d.npy' %var)
    np.save(savefile,Sigma_LR)
