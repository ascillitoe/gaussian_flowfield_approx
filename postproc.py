import os
import sys
import json
import pyvista
import numpy as np
from tqdm import tqdm
from funcs import r2_score, mae_score, standardise_minmax, unstandardise_minmax, proc_bump, 
evaluate_subspace, predict_subspace, parse_designs
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed, cpu_count

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
n_jobs = json_dat['njobs']
subdim = json_dat['subdim']
if (("points" in json_dat)==True): # Points to plot summary plots at (put None to skip)
    plot_pts = json_dat['points'] 
else:
    plot_pts = None

if (("designs" in json_dat)==True): # Designs to write to vtk for (put None to skip)
    designs_proc = json_dat['designs'] 
else:
    designs_proc = None

designmesh = False
if (("sample to design" in json_dat)==True): designmesh = json_dat['sample to design']

test = False
if (("plot test" in json_dat)==True): test = json_dat['plot test']

test_score = False
if (("test score" in json_dat)==True): test_score = json_dat['test score']

label = False
if (("label" in json_dat)==True): label = json_dat['label']

field_score = False
if (("field score" in json_dat)==True): field_score = json_dat['field score']

designs_train, designs_test = parse_designs(json_dat)

# housekeeping
saveloc = os.path.join(datadir,'POSTPROC_DATA',casename)
os.makedirs(saveloc,exist_ok=True)
os.makedirs(os.path.join(saveloc,'Figures'),exist_ok=True)

ncores = cpu_count()
if(n_jobs==-1): n_jobs = ncores
if (n_jobs>ncores): quit('STOPPING: n_jobs > available cores')
zp = 1.96

# Read in subspace weights and grf
dataloc = os.path.join(datadir,'SUBSPACE_DATA',casename)
os.chdir(dataloc)
#mysubspaces = np.load('SS.npy',allow_pickle=True)
print('Loading mygrf.pickle from '+ dataloc)
with open("mygrf.pickle","rb") as pickle_file:
    mygrfs = pickle.load(pickle_file)

# Read in bump data (training)
dataloc = os.path.join(datadir,'PROCESSED_DATA',casename)
os.chdir(dataloc)
X = np.load('X.npy')
X, min_X, max_X = standardise_minmax(X)

# Read in bump data (test)
if test:
    X_test = np.load('X_test.npy')
    X_test,*_ = standardise_minmax(X_test,min_value=min_X,max_value=max_X)

# Read in dtag array and true field data
print('Loading D.npy from '+ dataloc)
data = np.load('D.npy',allow_pickle=True)
if test:
    print('Loading D_test.npy from '+ dataloc)
    data_test = np.load('D_test.npy',allow_pickle=True)

num_pts     = data.shape[0]
num_designs = np.array([np.shape(point.D)[0] for point in data])
num_bumps   = X.shape[1]
num_vars    = data[0].D.shape[1]
print('Max number of designs = %d' % np.max(num_designs))
print('Min number of designs = %d' % np.min(num_designs))
print('Number of nodes = %d' %(num_pts))
print('Number of bump functions = %d' %(num_bumps))
if max(vars) > num_vars-1: quit('STOPPING: A vars index is greater than the number of arrays stored in D')

# Base vtk file - Sufficient summary plots, accuracy scores & variance to vtk file
###########################################################################
file = os.path.join(datadir,'CFD_DATA',casename,'basegrid.vtk')
#file = os.path.join(datadir,'CFD_DATA',casename,'baseline.vtk')
basegrid = pyvista.read(file)
gridcopy = basegrid.copy(deep=True)
coords = basegrid.points[:,:2]
vtk_pts = basegrid.n_points
if (vtk_pts!=num_pts): quit('STOPPING: Number of points in np arrays != vtk points')

if field_score:
    print('Evaluating subspace grfs at %d nodes...' % num_pts)
    for var in vars:
        print('Variable index %d...' % var)
        results = Parallel(n_jobs=n_jobs)(delayed(evaluate_subspace)(X, mygrfs[j,var], data[j],var) for j in tqdm(range(num_pts)))
        r2  = np.array([item[0] for item in results])
        mae = np.array([item[1] for item in results])

        # Save to basemesh vtk
        basegrid['r2_var'+str(var)] = np.array(r2)
        basegrid['mae_var'+str(var)] = np.array(mae)

        # Print average scores
        print('Average training r2 score = %.3f' %(np.mean(r2[r2>=-1])))
        print('Average training mae      = %.4f' %(np.mean(mae[mae>=-1])))

        if (test_score):
            results = Parallel(n_jobs=n_jobs)(delayed(evaluate_subspace)(X_test, mygrfs[j,var], data_test[j],var) for j in tqdm(range(num_pts)))
            r2  = np.array([item[0] for item in results])
            mae = np.array([item[1] for item in results])
            print('Average test r2 score = %.3f' %(np.mean(r2[r2>=-1])))
            print('Average test mae      = %.4f' %(np.mean(mae[mae>=-1])))

    os.chdir(saveloc)
    basegrid.save('basemesh_post.vtk')

# Sufficient summary plots at selected coords 
if plot_pts is not None:
    # Get bump x locations from 0th design
    design = 'design_0000'
    cfd_data = os.path.join(datadir,'CFD_DATA',casename,design)
    os.chdir(cfd_data)
    bump,lower,upper = proc_bump('deform_hh.cfg')

    for p, pt in enumerate(plot_pts):
        dist = coords - pt   
        dist = np.sqrt(dist[:,0]**2.0 + dist[:,1]**2.0)
        loc = np.argmin(dist)
        point = data[loc]
        indices = point.indices
        x = X[indices,:]
        if test:
            point_test = data_test[loc]
            indices_test = point_test.indices
            x_test = X_test[indices_test,:]

        for var in vars:
            mygrf = mygrfs[loc,var]
            M = mygrf.M
            u = x @ M
            y = point.D[:,var]

##            # TODO - TEMP DUMP for testing etc
#            dumpdir = os.path.join(basedir,'DUMP')
#            os.makedirs(dumpdir,exist_ok=True)
#            os.chdir(dumpdir)
#            if var==0:
#                np.save('pt%d_u.npy' %p,u)
#                np.save('pt%d_x.npy' %p,x)
#            np.save('pt%d_subpoly_%d.npy' %(p,var),subpoly)
#            np.save('pt%d_y_%d.npy' %(p,var),point.D[:,var])
##            # TODO 

            # Summary plot
            fig, ax = plt.subplots()
            fig.suptitle('Var %d, Point = (%.4f,%.4f)' %(var,coords[loc,0],coords[loc,1]))
            ax.set_xlabel('$\mathbf{w}^T \mathbf{x}$')
            ax.set_ylabel('Var %d' %var)
            ax.plot(u,y,'C0o',ms=8,mec='k',mew=1,alpha=0.6,label='Training') # Plot real field values

            if test:
                u_test = x_test @ M
                y_test = point_test.D[:,var]
                ax.plot(u_test,y_test,'C2o',ms=8,mec='k',mew=1,alpha=0.6,label='Test') # Plot real field values

            if label: 
                for i,d in enumerate(point.indices):
                    design = designs_train[d]
                    ax.annotate('%d' % design,[u[i],y[i]],color='C0')
                if test: 
                    for i,d in enumerate(point_test.indices):
                        design = designs_test[d]
                        ax.annotate('%d' % design,[u_test[i],y_test[i]],color='C2')

            ugrf = np.linspace(np.min(u),np.max(u),50)
            y_mean, y_std = mygrf.predict(ugrf.reshape(-1,1),return_std=True)
            ax.plot(ugrf,y_mean,'C3-',lw=3,label='Mean')
            ax.fill_between(u,y_mean-y_std,y_mean+y_std,color='C2',label='$\sigma$',alpha=0.3)

            # Annotate with mae and r2 score
            ypred = mygrf.predict(u,return_std=False)
            mae = mae_score(y,ypred)
            r2  = r2_score(y,ypred)
            if test:
                ypred_test = mygrf.predict(u_test,return_std=False)
                mae_test = mae_score(y_test,ypred_test)
                r2_test  = r2_score(y_test,ypred_test)
                ax.set_title('Train/Test MAE = %.2g/%.2g, Train/Test $R^2$ = %.3f/%.3f' %(mae,mae_test,r2,r2_test))
            else:
                ax.set_title('Train MAE = %.2g, Train $R^2$ = %.3f' %(mae,r2))
            ax.legend()

            filename = 'summary_pt%d_var%d.pickle' %(p,var)
            filename = os.path.join(saveloc,'Figures',filename)
            with open(filename,'wb') as f:
                pickle.dump(fig,f)

# TODO
#            # Plot active modes (M)
#            for d in range(subdim):
#                fig, ax = plt.subplots()
#                fig.suptitle('Var = %d, Mode = %d, Point = (%.4f,%.4f)' %(var, d, coords[loc,0],coords[loc,1]))
#                ax.set_xlabel('$x/C_x$')
#                ax.set_ylabel('$\mathbf{w}$')
#                # Lower surface
#                ax.plot(lower[:,0],W[loc,var,lower[:,1].astype(int),d],'-oC1',label='Lower')
#                ax.fill_between(lower[:,0],0,W[loc,var,lower[:,1].astype(int),d],color='C1',alpha=0.3)
#                ax.plot(upper[:,0],W[loc,var,upper[:,1].astype(int),d],'-oC2',label='Upper')
#                ax.fill_between(upper[:,0],0,W[loc,var,upper[:,1].astype(int),d],color='C2',alpha=0.3)
#                ax.legend()
#
#                filename = 'weights_pt%d_var%d_subdim%d.pickle' %(p,var,d)
#                filename = os.path.join(saveloc,'Figures',filename)
#                with open(filename,'wb') as f:
#                    pickle.dump(fig,f)

# Save predicted fields and errors to vtk for selected designs
##############################################################
if designs_proc is not None:
    print('Predicting fields for %d designs...' % len(designs_proc))
    for i in designs_proc:
        design = 'design_%04d' % i
        print(design)
        cfd_data = os.path.join(datadir,'CFD_DATA',casename,design)
        os.chdir(cfd_data)
        if designmesh: 
            designgrid = pyvista.read('flow.vtk')
        else:
            designgrid = pyvista.read('flow_base.vtk')
    
        # Field predictions (could use D_pred and D here, but get new pred and truth instead so works for designs outside of training/test set)
        Xi,*_ = proc_bump('deform_hh.cfg')
        Xi,*_ = standardise_minmax(Xi.reshape(1,-1), min_value=min_X, max_value=max_X) #Standardise with same min/max as original training x set
        for var in vars:
            # Use grf for prediction
            field_pred = Parallel(n_jobs=n_jobs)(delayed(predict_subspace)(Xi, mygrfs[j,var]) for j in tqdm(range(num_pts)))
            gridcopy['pred_var'+str(var)] = np.array(field_pred)

        # Resample back on to design grid
        gridcopy = gridcopy.sample(designgrid)

        # Save vtk
        savefile = os.path.join(saveloc,'flow_post_design_%04d.vtk' % i)
        gridcopy.save(savefile)

plt.show()
