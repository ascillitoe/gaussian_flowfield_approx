import os
import sys
import json
import pyvista as pv
import numpy as np
from tqdm import tqdm
from funcs import proc_bump, datapoint, parse_designs, corr_from_cov
from joblib import Parallel, delayed, cpu_count

global basedir, seed
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
n_jobs          = json_dat['njobs']
solver          = json_dat['solver']
Ncoarse         = json_dat['samples']
cutoff          = json_dat['cutoff']

eig = False
if (("eig" in json_dat)==True): eig = json_dat['eig']

corr = False
if (("corr" in json_dat)==True): corr = json_dat['corr']

designs_train, designs_test = parse_designs(json_dat)

# Housekeeping
ncores = cpu_count()
if(n_jobs==-1): n_jobs = ncores
if (n_jobs>ncores): 
    quit('STOPPING: n_jobs > available cores')
else:
    print('n_jobs = %d' %n_jobs)

def proc_data(d,casename,resample=False,solver='incompressible_SA'):
    if solver == 'compressible_euler':
        Tref = 273.15
        Uref = 99.3964
        num_var = 4 # Cp, T, u,v for now...
    elif solver == 'incompressible_SA':
        Uref  = 1.0
        muref = 1.853e-05 
        num_var = 4 # Cp, T, u,v for now...

    # Read in the base mesh to resample on to
    if resample:
        cfd_data = os.path.join(datadir,'CFD_DATA',casename)
        os.chdir(cfd_data)
        samplegrid = pv.read('basegrid.vtk')

    # Read design vtk file and resample onto the base grid
    design = 'design_%04d' % d
    cfd_data = os.path.join(datadir,'CFD_DATA',casename,design)
    os.chdir(cfd_data)
    designgrid = pv.read('flow.vtk')

    # Resample
    if resample:
        designgrid = samplegrid.sample(designgrid)
        designgrid.save('flow_base.vtk')
      
    # Get index for nodes in fluid region  (vtkGhostType=2 denotes solid regions, vtkGhostType=0 fluid) 
    fluid = designgrid.point_arrays['vtkGhostType'] #TODO - don't know if vtkGhostType exists before resample. If not it needs to be created in resample=False case.
    fluid[fluid == 0] = 1 #Fluid 
    fluid[fluid == 2] = 0 #Solid
    index = np.argwhere(fluid==1)
    
    # Extract flow parameters (Array elements for nodes that lie in solid region of the design's mesh are left empty)
    num_pts = designgrid.n_points
    D = np.empty([num_pts,num_var])
    if solver == 'compressible_euler':
        D[index,0] = designgrid.point_arrays['Pressure_Coefficient'][index] #p
        D[index,1] = designgrid.point_arrays['Temperature'][index]/Tref #T
        ro = designgrid.point_arrays['Density']
        D[index,2] = (designgrid.point_arrays['Momentum'][index,0]/ro[index])/Uref #u
        D[index,3] = (designgrid.point_arrays['Momentum'][index,1]/ro[index])/Uref #v
    elif solver == 'incompressible_SA':
        D[index,0] = designgrid.point_arrays['Pressure_Coefficient'][index] #p
        D[index,1] = designgrid.point_arrays['Eddy_Viscosity'][index]/muref #mut
        D[index,2] = designgrid.point_arrays['Velocity'][index,0]/Uref #u
        D[index,3] = designgrid.point_arrays['Velocity'][index,1]/Uref #v

    # Read bump data
    deform_data, *_ = proc_bump('deform_hh.cfg')
    
    return (deform_data, D, fluid)

def proc_results(results,saveloc,idx_coarse=None,idx_fine=None,eig=False):
    global seed
    np.random.seed(seed)

    X     = np.array([item[0] for item in results])
    D     = np.array([item[1] for item in results])
    fluid = np.array([item[2] for item in results])
    num_designs = D.shape[0]
    num_pts     = D.shape[1]
    num_vars    = D.shape[2]
    num_bumps   = X.shape[1]
    print('Number of designs = %d' %(num_designs))
    print('Number of nodes = %d' %(num_pts))
    print('Number of bump functions = %d' %(num_bumps))

    train = False
    if idx_fine is None and idx_coarse is None: train = True

    if train:
        ################
        # Subsample data
        ################
        print('Subsampling data...')

        # Random sampling 
        print('Subsampled number of points: %d' %Ncoarse)
        # First pick out acceptable points (i.e. where sufficient number of designs have "fluid" at each point)
        n_valid_designs = np.count_nonzero(fluid == 1, axis=0)
        idx = np.argwhere(n_valid_designs>=cutoff).reshape(-1)
#        idx = np.arange(num_pts)
        # Now subsample remaining points
        idx_coarse = np.sort(np.random.choice(idx,Ncoarse,replace=False))

        flag = np.full(num_pts,False)
        flag[idx_coarse] = True
        idx = np.arange(num_pts)
        idx_fine = idx[~flag]

        # Save point cloud of subsampled points (for inspection)
        filename = os.path.join(datadir,'CFD_DATA',casename,'basegrid.vtk')
        samplegrid = pv.read(filename)
        points = samplegrid.points[idx_coarse]
        point_cloud = pv.PolyData(points)
        point_cloud.save(os.path.join(saveloc,'sample_points.vtk'))

        ############################################
        # Build covariance matrix (if training data)
        ############################################
        # Get mean of data (needed for Schur complement)
        print('Getting mean of data...')
        Dmean = np.mean(D,axis=0)

        print('Computing covariance matrix for variable...')
        # Reorder data so that fine and course subsets are colocated within the data (and Sigma matrix)
        idx_new = np.concatenate([idx_fine,idx_coarse])
        Dnew = D[:,idx_new,:].transpose(1,0,2)  # reshape to (num_pts,num_designs,num_var) ordering
        
        # Obtain covariance matrix 
        Nfine = len(idx_fine)
        for j in range(num_vars):
            print(j)
            Sigma = np.cov(Dnew[:,:,j],bias=True)
        
            # Compute eigenvalues and eigenvectors
            if eig:
                print('Computing eigen decomposition...')
                DD, V = np.linalg.eigh(Sigma)
                idx_eig = DD.argsort()[::-1]
                Vec  = np.empty([num_pts,5]) # Save the first 5 eigenvector fields only
                # Save to grid (after reordering)
                Vec[idx_fine,:]   = V[:Nfine,idx_eig[:5]]
                Vec[idx_coarse,:] = V[Nfine:,idx_eig[:5]]
                samplegrid['eigvec_%d' %j] = Vec
                # Save eigenvec ordered by eigenvalue size
                print('Saving eigenvalues and eigenvectors...')
                np.save(os.path.join(saveloc,'eigvals_%d.npy' %j),DD)
                np.save(os.path.join(saveloc,'eigvecs_%d.npy' %j),V)

            # Compute correlation matrix
            if corr:
                print('Computing correlation matrix...')
                # Rearrange Sigma to original ordering first
                Sigma_orig = np.cov(D[:,:,j].T,bias=True)
#                Sigma_orig = np.empty([num_pts,num_pts])
#                Sigma_orig[np.ix_(idx_fine,idx_fine)]     = Sigma[:Nfine,:Nfine]
#                Sigma_orig[np.ix_(idx_coarse,idx_coarse)] = Sigma[Nfine:,Nfine:]
                Corr = corr_from_cov(Sigma_orig)
                savefile = os.path.join(saveloc,'corr_%d.npy' %j)
                np.save(savefile,Corr)

            # Save to file
            print('Saving covariance data...')
            savefile = os.path.join(saveloc,'covar_%d.npy' %j)
            np.save(savefile,Sigma)

        # Save misc covar data
        print('Saving misc data...')
        savefile = os.path.join(saveloc,'covar.npz')
        np.savez(savefile,Dmean=Dmean,idx_fine=idx_fine,idx_coarse=idx_coarse)
        if eig: 
            print('Saving eigenvectors vtk file...')
            samplegrid.save(os.path.join(saveloc,'covar_fields.vtk'))

    #########################
    # Process subsampled data
    #########################
    # Coarse data
    Dcoarse = D[:,idx_coarse,:]
    fluid_coarse = fluid[:,idx_coarse]

    # Convert D[designs,pts,var] -> D[pts][designs,var+1] (as some nodes have less designs with valid/fluid data). Store in array if datapoint objects
    print('Rearranging data ordering for subsampled data...')
    data = np.empty(Ncoarse,dtype='object')
    for j in tqdm(range(Ncoarse)):
        indices = np.argwhere(fluid_coarse[:,j]==1).reshape(-1) # Store indices for valid designs at each node
        dat = Dcoarse[:,j,:][indices,:]
        data[j] = datapoint(D=dat,indices=indices)

    print('Saving subsampled data...')
    if train:
        np.save(os.path.join(saveloc,'X.npy'),X)
        np.save(os.path.join(saveloc,'D.npy'),data)
    else:
        np.save(os.path.join(saveloc,'X_test.npy'),X)
        np.save(os.path.join(saveloc,'D_test.npy'),data)

    #############################
    # NEW: Process original data (so we can calc. error metrics in postproc)
    #############################
    # Convert D[designs,pts,var] -> D[pts][designs,var+1] (as some nodes have less designs with valid/fluid data). Store in array if datapoint objects
    print('Rearranging data ordering for original data...')
    data = np.empty(num_pts,dtype='object')
    for j in tqdm(range(num_pts)):
        indices = np.argwhere(fluid[:,j]==1).reshape(-1) # Store indices for valid designs at each node
        dat = D[:,j,:][indices,:]
        data[j] = datapoint(D=dat,indices=indices)

    print('Saving original data...')
    if train:
        np.save(os.path.join(saveloc,'D_orig.npy'),data)
        return idx_coarse, idx_fine
    else:
        np.save(os.path.join(saveloc,'D_test_orig.npy'),data)

saveloc = os.path.join(datadir,'PROCESSED_DATA',casename)
os.makedirs(saveloc,exist_ok=True)

###############
# Training data
###############
print('\nTraining data...')
results = Parallel(n_jobs=n_jobs)(delayed(proc_data)(d,casename,resample=True,solver=solver) for d in tqdm(designs_train))
idx_coarse, idx_fine = proc_results(results,saveloc,eig=eig)

###########
# Test data
###########
print('\nTest data...')
results = Parallel(n_jobs=n_jobs)(delayed(proc_data)(d,casename,resample=True,solver=solver) for d in tqdm(designs_test))
proc_results(results,saveloc,idx_coarse=idx_coarse,idx_fine=idx_fine)

print('Finished...')
