import pyvista as pv
import numpy as np
import os
import sys
import json

basedir = os.getcwd()

#################################################################################
# Inputs
#################################################################################
# Fine box limits
xbox  = [-0.15,1.15]
ybox  = [-0.15,0.15]
boxdx = [0.02,0.015] 

# growth rate outside of box
xgr = 1.5
ygr = 1.5
xdist = 0.8 
ydist = 0.4

inputfile = sys.argv[1]
with open(inputfile) as json_file:
    json_dat = json.load(json_file)
casename = json_dat['casename']
datadir  = json_dat['datadir']
testfile = None #os.path.join(datadir,'CFD_DATA',casename,'design_0000','flow.vtk')

#################################################################################



# Create box
x = np.arange(xbox[0],xbox[1],boxdx[0])
y = np.arange(ybox[0],ybox[1],boxdx[1])
z = 0

# Grow outside of box
def grow_mesh(startcell,gr,dist):
    dx    = (startcell[1]-startcell[0])*gr
    start = startcell[1]
    xnew = []
    temp = start
    if dx > 0:
        dist = start + dist
        while temp <= dist:
            temp += dx
            xnew.append(temp)
            dx *= xgr
        return np.asarray(xnew)
    elif dx < 0:
        dist = start - dist
        while temp >= dist:
            temp += dx
            xnew.append(temp)
            dx *= xgr
        xnew.reverse()
        return np.asarray(xnew)

xrhs = grow_mesh(x[-2:]    ,xgr,xdist)
xlhs = grow_mesh(x[1::-1] ,xgr,xdist)
yupp = grow_mesh(y[-2:]    ,ygr,ydist)
ydwn = grow_mesh(y[1::-1] ,ygr,ydist)
x = np.append(np.append(xlhs,x),xrhs)
y = np.append(np.append(ydwn,y),yupp)


# Create vtk structured grid
nx = len(x)
ny = len(y)
print('nx = ', nx)
print('ny = ', ny)
print('n = ', nx*ny)
xx,yy,zz = np.meshgrid(x,y,z)
grid = pv.StructuredGrid(xx,yy,zz)

# Resample flow onto base mesh to test
if testfile is not None:
    testgrid = pv.read(testfile)
    grid = grid.sample(testgrid,mark_blank=False) 

savefile = os.path.join(datadir,'CFD_DATA',casename,'basegrid.vtk')
grid.save(savefile)
