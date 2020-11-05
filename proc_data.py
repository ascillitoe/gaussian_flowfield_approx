import os
import sys
import json
import pyvista
import numpy as np
from tqdm import tqdm
from funcs import proc_bump, datapoint, parse_designs
from joblib import Parallel, delayed, cpu_count

global basedir
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
        num_var = 1 # Cp, T, u,v for now...

    # Read in the base mesh to resample on to
    if resample:
        cfd_data = os.path.join(datadir,'CFD_DATA',casename)
        os.chdir(cfd_data)
        samplegrid = pyvista.read('basegrid.vtk')

    # Read design vtk file and resample onto samplegrid
    design = 'design_%04d' % d
    cfd_data = os.path.join(datadir,'CFD_DATA',casename,design)
    os.chdir(cfd_data)
    designgrid = pyvista.read('flow.vtk')

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
        Cp = designgrid.point_arrays['Pressure_Coefficient'][index] #p
        u  = designgrid.point_arrays['Velocity'][index,0] #u
        v  = designgrid.point_arrays['Velocity'][index,1] #u
        Yp = 1.0 - (Cp + (1/Uref**2)*(u**2+v**2))
        D[index,0] = Yp

    # Read bump data
    deform_data, *_ = proc_bump('deform_hh.cfg')
    
    return (deform_data, D, fluid)

def proc_results(results):
    X     = np.array([item[0] for item in results])
#    dtag  = np.array([d for i,d in enumerate(range(len(results)))])
    fluid = np.array([item[2] for item in results])
    num_designs = fluid.shape[0]
    num_pts     = fluid.shape[1]
    num_vars    = X.shape[1]
    print('Number of designs = %d' %(num_designs))
    print('Number of nodes = %d' %(num_pts))
    print('Number of bump functions = %d' %(num_vars))
    D     = np.array([item[1] for item in results])
    # Convert D[designs,pts,var] -> D[pts][designs,var+1] (as some nodes have less designs with valid/fluid data). Store in array if datapoint objects
    print('Rearranging data ordering...')
    data = np.empty(num_pts,dtype='object')
    for j in tqdm(range(num_pts)):
        indices = np.argwhere(fluid[:,j]==1).reshape(-1) # Store indices for valid designs at each node
        dat = D[:,j,:][indices,:]
        data[j] = datapoint(D=dat,indices=indices)
    return X,data#,dtag

saveloc = os.path.join(datadir,'PROCESSED_DATA',casename)
os.makedirs(saveloc,exist_ok=True)

# Training
if designs_train is not None:
    print('\nTraining data...')
    results = Parallel(n_jobs=n_jobs)(delayed(proc_data)(d,casename,resample=True,solver=solver) for d in tqdm(designs_train))
    X,D = proc_results(results)
    print('Saving to file...')
    np.save(os.path.join(saveloc,'X.npy'),X)
    np.save(os.path.join(saveloc,'D.npy'),D)
#    np.save(os.path.join(saveloc,'dtag.npy'),dtag)

# Test
if designs_test is not None:
    print('\nTest data...')
    results = Parallel(n_jobs=n_jobs)(delayed(proc_data)(d,casename,resample=True,solver=solver) for d in tqdm(designs_test))
    X,D = proc_results(results)
    print('Saving to file...')
    np.save(os.path.join(saveloc,'X_test.npy'),X)
    np.save(os.path.join(saveloc,'D_test.npy'),D)
#    np.save(os.path.join(saveloc,'dtag_test.npy'),dtag)
