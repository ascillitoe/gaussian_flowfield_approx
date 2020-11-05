import numpy as np

class datapoint:
    def __init__(self,D,indices):
        self.D       = D
        self.indices = indices

def standardise_minmax(X,min_value=None,max_value=None):
    M,d=X.shape
    X_stnd=np.zeros((M,d))
    if min_value is None:
        min_value = np.empty(d)
        max_value = np.empty(d)
        for j in range(0,d):
            max_value[j] = np.max(X[:,j])
            min_value[j] = np.min(X[:,j])
    for j in range(0,d):
        for i in range(0,M):
            X_stnd[i,j]=2.0 * ( (X[i,j]-min_value[j])/(max_value[j] - min_value[j]) ) -1
    return X_stnd, min_value, max_value

def unstandardise_minmax(X,min_value,max_value):
    d=X.shape[1]
    if len(min_value)==1:
        min_value = [min_value]*d
        max_value = [max_value]*d
    X_unstnd=np.zeros_like(X)
    for j in range(0,d):
        X_unstnd[:,j] = 0.5*(X[:,j] +1)*(max_value[j] - min_value[j]) + min_value
    return X_unstnd

def standardise_mustd(X,mean_value=None,std_value=None):
    M,d=X.shape
    X_stnd=np.zeros((M,d))
    if mean_value is None:
        mean_value = np.empty(d)
        std_value  = np.empty(d)
        for j in range(0,d):
            mean_value[j] = np.mean(X[:,j])
            std_value[j]  = np.sqrt(np.var(X[:,j]))
    for j in range(0,d):
        for i in range(0,M):
            X_stnd[i,j]= (X[i,j]-mean_value[j])/std_value[j]     
    return X_stnd, mean_value, std_value

def r2_score(ytrue, ypred):
    from scipy.stats import linregress
    """ Return R^2 where ytrue and ypreed are array-like."""
    slope, intercept, r_value, p_value, std_err = linregress(ytrue, ypred)
    return r_value**2

def rmse_score(ytrue, ypred):
    return np.sqrt(np.mean((ytrue-ypred)**2))

def mae_score(ytrue, ypred,norm=False):
    MAE = np.mean(np.abs(ytrue-ypred))
    if norm:
        MAE /= np.std(ytrue)
    return MAE 

def proc_bump(filename): 
    #TODO - search for DV_VALUE and DV_PARAM instead of hardcoded lines/cols
    from ast import literal_eval

    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

    # Find bump data
    param_line = None
    value_line = None
    for l, line in enumerate(lines):
        if 'DV_PARAM' in line: param_line = l
        if 'DV_VALUE' in line: value_line = l
    if param_line is None: quit('STOPPING: DV_PARAM not found')
    if value_line is None: quit('STOPPING: DV_VALUE not found')

    # Read bump location data
    string  = lines[param_line][10:]
    string = string.replace(" ", "").strip(";").split(";")
#    N = len(string)
    lower = []
    upper = []
    for i,s in enumerate(string):
        temp = literal_eval(s)
        surf = temp[0] #0 for lower, 1 for upper
        x    = temp[1]   
        if surf==0:
            lower.append([x,i])
        elif surf==1:
            upper.append([x,i])

    # Read bump amplitude data
    string = lines[value_line][10:]
    bump   = np.fromstring(string, sep=',')

    return bump, np.array(lower), np.array(upper)
def parse_designs(json_dat):
    from os.path import join
    casename        = json_dat['casename']
    datadir         = json_dat['datadir']
    designs_train   = json_dat['train']
    if isinstance(designs_train,list): # Range is specified... 
        designs_train = range(designs_train[0],designs_train[1]+1)
        if (("test" in json_dat)==True):
            designs_test = json_dat['test']
            designs_test = range(designs_test[0],designs_test[1]+1)
        else:
            designs_test = None
    elif isinstance(designs_train,int):
        all_designs = np.load(join(datadir,'CFD_DATA',casename,'design_data.npy'))
        print('Total number of designs = %d' %len(all_designs))
        designs_train = all_designs[:designs_train]
        if (("test" in json_dat)==True):
            designs_test = json_dat['test']
            designs_test = all_designs[len(designs_train):len(designs_train)+designs_test]
        else:
            designs_test = None
    return designs_train, designs_test

def evaluate_subspace(X,mygrf,point,var):
    if mygrf==None:
        ypred = -99 
        r2    = -99
        mae   = -99
    else:
        indices = point.indices
        x = X[indices,:]
        M = mygrf.M
        u = x @ M
        ypred = mygrf.predict(u,return_std=False)
        r2  = r2_score(point.D[:,var],ypred)
        mae = mae_score(point.D[:,var],ypred,norm=True)
    return (r2,mae)

def predict_subspace(X,mygrf):
    if mygrf==None:
        ypred = -99 
    else:
        M = mygrf.M
        u = X @ M
        ypred = mygrf.predict(u,return_std=False)
    return ypred
