import os
import sys
import json
import numpy as np
import pyvista as pv
from tqdm import tqdm

#################
# Settings
#################
points = [ [0.01,0.03], [1.1,0.08], [0.085,-0.055], [0.085, 0.055], [0.98,0.02] ] 
vars = [0,1,2]

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

# Read in base vtk file
dataloc = os.path.join(datadir,'CFD_DATA',casename)
grid = pv.read(os.path.join(dataloc,'basegrid.vtk'))
coords = grid.points[:,:2]

# Read in corr matrix
dataloc = os.path.join(datadir,'PROCESSED_DATA',casename)
print('Reading on correlation matrices...')
corr = []
for var in vars:
    print(var)
    corr.append(np.load(os.path.join(dataloc,'corr_%d.npy'%var)))

###########################################################
# Save correlations for given points (and vars) to vtk file
###########################################################
for p, pt in enumerate(points):
    # Find nearest point
    dist = coords - pt   
    dist = np.sqrt(dist[:,0]**2.0 + dist[:,1]**2.0)
    loc = np.argmin(dist)

    # Save correlation between all points and point above
    for j, var in enumerate(vars):
        corr_w_pt = corr[j][loc,:]
        grid['corr_pt%d_var%d' %(p,var)] = corr_w_pt

saveloc = os.path.join(datadir,'POSTPROC_DATA',casename)
grid.save(os.path.join(saveloc,'corr.vtk'))
