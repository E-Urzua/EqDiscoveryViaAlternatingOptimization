"""
Text here last updated - 15/02/2024

Python file to test all things related to data loading for the Learn PDE classes as well as 
older data loader classes that I used when originally learning things in Python. Once I feel 
that anyone of the classes I have tests out enough and I feel are up to a good working state
I will more them over into the DataLoaders.py file since that file is meant to contain only
stuff that I would feel comfortable other people using for whatever reason. Also because 
this file also contains all of my tests it also can be seen as a sort of way to document the
paths that were an were not taken to get where the code is. 
"""
import os
import warnings
from typing import Union
from scipy.stats import qmc, uniform
import scipy.io as sio
import numpy as np
from numpy.random import default_rng, SeedSequence
# from primesieve.numpy import n_primes
import torch
from torch.utils.data import Dataset
import torch.utils.data as Data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NOTE: Seriously take note that each of the PDE learning method classes here for each method assume that the data is
# stored in a certain ordering. More specifically, each method assumes that the solution to a PDE is given at specific 
# spatial-temporal data points (x,y,z,t) is contained within an arrray/matrix that has a dictionary key name as 'usol'
# in the .mat file while the the spatial and temporal points having a dictionary keys as 'x', 'y', 'z', 't'     
# and are vectors (either one 1D arrays or 2D arrays with one dimension havine a size of 1). Furthermore the solution
# to the PDE found in the 'usol' matrix/array is expected to be stored such that the evalutation of the PDE solution
# at the sptaial temporal point (x_{i}, y_{j}, z_{k}, t_{l}) is usol[i,j,k,l] (in 1+1 dims (x_{i}, t_{l}) is at 
# usol[i,l]). If one is curious as to why this is, the reason is that in each of the classes below we generate/create
# a grid using the [X, Y, Z, T] = np.meshgrid(x, y, z, t, indexing='xy') and then flatten/reshape the usol matrix via 
# a Fortran (F) column-major like ordreing so that all the evaluations of the PDE solution over the spatial grid 
# apprear for a given temporal point usol(x, y, z, t_{l}) appear before usol(x,y,z, t_{l'}) for 0<=l<l'<n_{t}-1 where
#  n_{t} is the number of discrete time points. Regarding how the spatial points (x_{i}, y_{j}, z_{k}, t_{l}) vary
# when t_{l} is held constant when flattened into a 1D array, the z_{k} vary first, then the y_{j} and then followed
# by the x_{i},  .

# Note further that we have not come up with a way to check that the data in usol is stored this way when it is read 
# in from the .mat file. We do a simple check that the the shape of the usol matrix matches that of the X, Y, Z, T 
# matrices above amd then do some tranposes to make usol have the same shape as X, Y, Z, and T  but this does not 
# always work because this check fails when n_{x}=n_{y}=n_{z}=n_{t} or some combination of equality (how to know which 
# axis to tranpose when multiple are the same size). I can come up with a way to do something when some of the number
# of x, y, z or t points equals any any one of the others but I honeslty do not  think that there is a way to always 
# check if the data in usol is stored as we expect it. Like at this point it is up to the user to make sure that that 
# data in the .mat file is saved as excepted and we will leave the little check we have as is becuase in the case of 
# of here being multiple dimensions with the sam enumber of points then as least the thrown error will signal some-
# thing is wrong with the data,

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TODO: double check the statements in the above note and clarify the ordering of the usol points in relation to the 
#      spatial-temporal points, are such 
def PDELearningMatData(fname:str, Sptldims:int, split:float=0.80, smpleprcnt:float=0.20, noisePrcntg:float=0.15, 
                 seed:int=None, to_float:bool=True, **kwargs):
    """
        Revised method of how to get the data from a .mat file that will be used to learn a PDE EQ 
        using the PDE learning Classes seen in the Model.py File. This method/class is a second 
        version of sorts to the LearningMatData classes seen above in that it is built off of those
        classes but simplifies many things. Most important is that this one class is meant to work
        with data that has 1,2, or 3 spatial dimensions and with one of the three Monte Carlo method
        to create the random collocation points. And so unlike with the classes this one is built
        off of, there will not be one version for each Monte Carlo method for data with 1, 2, and 3
        spatial dimensions. As well this class determines the data points for training and in a more
        streamlined way that takes up less code lines and is more along the line how I have created
        similar classes elsewhere in my code library for similar things.
        
        Initialization method for the class. The Initializaton parameters/arguments are defined 
        as follows:
            * fname -  file name for the .mat file that contains the training data. The data is expectd to be stored 
                or contained within the .mat file in a specific way; the spatial variable coordinates saved as a 
                variable named 'x', 'y' 'z' and the temporal values stored as a matrix/vector named 't' and the 
                data values at the (x,y,z,t) points saved to a matrix variable named 'usol'. 
            * Sptldims - The number of spatial dimension of the data. This argument is used to determine whether
                or not to read the fname data file for variables y,z if it's value is greater than one. The 
                values this parameter can take 1, 2 or 3 currently. 
            * split - The percentage to split the data apart into training and validation data. The value given is 
                the percentage that constitutes the training data set. The rest is for the validation set
            * subsample_prcntg - The percentage of all the data points that are used found in the .mat file that 
                are used in training the model, either as part of the training or testing data sets). If the given 
                value is 0.10, then only 10 percent of the usol data in the .mat file will randomly selected to be 
                split into training and testing set according to the given value of the split parameter.
            * noisePrcntg - The amount of noise to add to the data set. The noise is random sample from normal 
                distibution with 0 mean (mu = 0) and standard deviation equal to the standard deviation of 
                the entire usol data (may be changed later to just the traing and testing data).
            * Ncp -  the number of collocation points used in eveluating the models candidate libray terms
                and thus for determining/infering a (reduced order or surogate)  modeling PDE equation for the data.
            to_float -  boolean argument indicating to have torch tensors be of torch.float32 or (True) or 
                torch.float64 data types when transfering from numpy to torch. 
            * seeds - list argument that contains the seeds values for numpy and scipy. The seeds are used to create 
                the rng method that control which points are selected to be in the training, testing and validation
                data sets. This argument is for reproducibility of results though usage of this does not garuentee 
                that the results are reproducable as it has been seen with usage of a computer cluster that even 
                when giving the same seed value, the randomly determined values from the seeded rng machine can be 
                different between one run to another. Specifically on said cluster for a job that was run over 4 
                GPUs, when the speed argument was passed from here was passed to the SeedSequence() function 
                with in the numpy.random module, the resulting entropies was not the same on each of the GPUs,
            * to_float - boolean parameter that indicates whether the data when it is eventually passed to torch needs
                to have a dtype of toch.float32 - True (normal 32 bit float data type) or if it needs to be 
                torch.float64 - False (normal 64 bit data type i.e double)

        TODO: 08/02/2024 - 
            (1). Finish input argument checking
            (2). Better method for the fname checking or just maybe looking to path library and os library
                 stuff to see if the file exists and things like that. 
            (3) Change up the rng seed stuff. It may be best not to have a seed value passed by a user and 
                instead get the seeds using the SeedSequence().entropy thing, save it to a class variables
                and then have as a **kwargs arguments for restarting that involve seed values.  ¯ \\ _ (ツ) _ // ¯
        """
    
    # TODO: Finish input argument checking
    if not isinstance(fname, str):
        raise TypeError(f"The fname argument/paramter is expected to be a str not a {type(fname).__name__}")
    elif '.mat' not in fname:
        raise ValueError(f"The given fname str parameter is not a .mat file which is the expected file type.")
    if not isinstance(Sptldims, int):
        raise TypeError(f"The Sptldims argument/paramter is expected to be a int type object not a {type(Sptldims).__name__}")
    elif Sptldims<=0 or Sptldims>=4:
        raise ValueError(f"The given Sptldims parameter needs be a value of 1,2, or 3. What was passed = {Sptldims}.")
    if not isinstance(split, float):
        msg = (f"The split input argument that splits the learning data set into training and testing needs to\n"
               f"be a float argument between 0 and 1.0. What you gave is a {type(split).__name__} type argument")
        raise TypeError(msg)
    if not 0.0<split<1.0:
        msg = (f"The split input argument needs to be a float be in the inteval (0.0, 1.0) and what you gave {split}\n"
               "is not within  that interval.")
        raise ValueError(msg)
    if not isinstance(smpleprcnt, float):
        msg = (f"The smpleprcnt input argument that splits the data set into learning and validation sets needs to\n"
               f"be a float argument between 0 and 1.0. What you gave is a {type(smpleprcnt).__name__} type argument")
        raise TypeError(msg)
    if not 0.0<smpleprcnt<1.0:
        msg = (f"The smpleprcnt input argument needs to be a float be in the inteval (0.0, 1.0) and what you gave {smpleprcnt}\n"
               "is not within that interval.")
        raise ValueError(msg)
    if seed and (not isinstance(seed, int)):
        raise TypeError(f"seed arguement needs to be a positive integer greater than 0")
    if seed==None:
        seed = SeedSequence().entropy
    if seed<1:
        raise ValueError(f"seed arguement needs to be a positive integer greater than 0")
    ss1 = SeedSequence(seed)
    
    print(f"The entropy or np seed used for seeding the default rng routine is {ss1.entropy}")
    # self.NPentropy = ss1.entropy
    rng = default_rng(seed=ss1)
    if not isinstance(to_float, bool):
        raise TypeError("The to_float argument needs to be a boolean valued argument!")
    # self._to_float = to_float
    data = sio.loadmat(os.getcwd()+'/'+fname)
    # get the x and t points at column vectors/arrays. Then create a mesh grid of the points
    x = np.real(data['x'].flatten()[:, None])
    t = np.real(data['t'].flatten()[:, None])
    sol = np.real(data['usol'])
    if isinstance(noisePrcntg, (int, float)):
        if noisePrcntg >=0 and noisePrcntg <=1.0:
            sol = sol + rng.normal(loc=0, scale=noisePrcntg*np.std(sol) , size=sol.shape)
            # self.noisePrcntg = noisePrcntg
        elif noisePrcntg>1.0:
            msg = ('WARNING; Given float noisePrcntg value is greater than 1.0 and so to convert it to\n'
                   'a decimal value it will be divided by 100 as we are interpretting any value greater than 1.0\n'
                   'to be a percentage and thus to convert out of a percentage (i.e p%) divided by 100 and use \n'
                   'that value (i.e p/100).')
            warnings.warn(msg, stacklevel=2)
            sol = sol + rng.normal(loc=0, scale=(noisePrcntg/100)*np.std(sol) , size=sol.shape)
            noisePrcntg = (noisePrcntg/100)
        else:
            raise ValueError(f"The noisePrcntg input value that was given was a negative number and it needs to be positive")
    else:
        msg = ('The user given noisePrcntg argument/parameter is not a floating number - Will not use any noise')
        warnings.warn(msg, stacklevel=2)
    # All the data fines regardless of spatial dims has x, t, and sol in the data file now read if there is more. 
    if Sptldims==2:
        y = np.real(data['y'].flatten()[:, None])
        t = np.real(data['t'].flatten()[:, None])
        n_x, n_y, n_t = x.shape[0], y.shape[0], t.shape[0]
        Nsp =n_x*n_y
        X, Y, T = np.meshgrid(x, y, t)  # with indexing parameter. value shape is (n_y, n_x, n_t)
        if X.shape!=sol.shape:
            lst = []
            trgts = np.array(X.shape, dtype=int)
            inpts = np.array(sol.shape, dtype=int)
            for i in range(trgts.shape[0]):
                lst.append(np.nonzero(inpts[i]==trgts)[0].item())
            sol= np.transpose(sol, axes=tuple(lst))
            del lst
        # pts = np.hstack((np.copy(X.reshape(-1,1,order='F')), np.copy(Y.reshape(-1,1,order='F')), np.copy(T.reshape(-1,1,order='F'))))
        pts = np.hstack((np.copy(X.reshape(-1,1,order='C')), np.copy(Y.reshape(-1,1,order='C')), np.copy(T.reshape(-1,1,order='C'))))
        # U = np.copy(sol.reshape(-1,1, order='F'))
        U = np.copy(sol.reshape(-1,1, order='C'))
    elif Sptldims==3:
        y = np.real(data['y'].flatten()[:, None])
        z = np.real(data['z'].flatten()[:, None])
        t = np.real(data['t'].flatten()[:, None])
        n_x, n_y, n_z, n_t = x.shape[0], y.shape[0], z.shape[0], t.shape[0]
        Nsp =n_x*n_y*n_z
        Y, Z, X, T = np.meshgrid(y, z, x, t)  # with indexing parameter. value shape is (n_z, n_y, n_x, n_t)
        if X.shape!=sol.shape:
            lst = []
            trgts = np.array(X.shape, dtype=int)
            inpts = np.array(sol.shape, dtype=int)
            for i in range(trgts.shape[0]):
                lst.append(np.nonzero(inpts[i]==trgts)[0].item())
            sol= np.transpose(sol, axes=tuple(lst))
            del lst
        # pts = np.hstack((np.copy(X.reshape(-1,1,order='F')), np.copy(Y.reshape(-1,1,order='F')), np.copy(Z.reshape(-1,1,order='F')), np.copy(T.reshape(-1,1,order='F'))))
        pts = np.hstack((np.copy(X.reshape(-1,1,order='C')), np.copy(Y.reshape(-1,1,order='C')), np.copy(Z.reshape(-1,1,order='C')), np.copy(T.reshape(-1,1,order='C'))))
        # U = np.copy(sol.reshape(-1,1, order='F'))
        U = np.copy(sol.reshape(-1,1, order='C'))
    else: # just revert back down to one spatial dim
        n_x, n_t = x.shape[0], t.shape[0]
        Nsp =n_x
        X, T = np.meshgrid(x, t) # the shape of each of these is n_t by n_x
        if X.shape!=sol.shape:
            sol = sol.T
        pts = np.hstack((np.copy(X.reshape(-1,1, order='C')), np.copy(T.reshape(-1,1, order='C'))))
        U = np.copy(sol.reshape(-1,1,order='C'))

    if len(kwargs)!=0:
        # check if the number of training points is given. 
        if "N_trn_pnts" not in kwargs.keys():
            raise KeyError("Only allowed additional keyword argument is N_trn_pnts")
        N_trn = kwargs["N_trn_pnts"]
        smpleprcnt = np.around(np.around(N_trn/split)/n_t)/Nsp
        # smpleprcnt = 0.64
    
    num_smpls = int(np.around(Nsp*smpleprcnt))
    val_num = int(Nsp - num_smpls)
    N_trn = int(np.around(num_smpls*split))
    N_tst = int(num_smpls-N_trn)
    innerIDs = rng.choice(Nsp, num_smpls, replace=False).astype(np.int32)
    innerIDs.sort() # NOTE - may not need this line
    valIDS = np.setdiff1d(np.arange(Nsp), innerIDs, assume_unique=True)
    X_meas = np.empty((num_smpls*n_t,Sptldims+1), dtype=pts.dtype)
    X_val = np.empty((val_num*n_t,Sptldims+1), dtype=pts.dtype)
    U_meas = np.empty((num_smpls*n_t,1), dtype=pts.dtype)
    U_val = np.empty((val_num*n_t,1), dtype=pts.dtype)
    for i in range(0, n_t):
        X_meas[i*num_smpls:num_smpls*(i+1), :] = pts[innerIDs+Nsp*i,:]
        X_val[i*val_num:val_num*(i+1), :] = pts[valIDS+Nsp*i,:]
        U_meas[i*num_smpls:num_smpls*(i+1), :] = U[innerIDs+Nsp*i,:]
        U_val[i*val_num:val_num*(i+1), :] = U[valIDS+Nsp*i,:]
    # Now split the data that will be use to train the model into a train and test sets.
    X_trn = np.empty((N_trn*n_t, Sptldims+1), dtype=X_meas.dtype,)
    U_trn = np.empty((N_trn*n_t, 1), dtype=U_meas.dtype,)
    X_tst = np.empty(((num_smpls-N_trn)*n_t, Sptldims+1), dtype=X_meas.dtype,)
    U_tst = np.empty(((num_smpls-N_trn)*n_t, 1), dtype=U_meas.dtype,)
    for i in range(0,n_t): 
        trnIDs = rng.choice(num_smpls, N_trn, replace=False).astype(np.int32)
        trnIDs.sort()
        tstIDs = np.setdiff1d(np.arange(num_smpls), trnIDs, assume_unique=True)
        X_trn[i*N_trn:N_trn*(i+1), :] = X_meas[trnIDs+num_smpls*i,:]
        U_trn[i*N_trn:N_trn*(i+1), :] = U_meas[trnIDs+num_smpls*i,:]
        X_tst[i*N_tst:N_tst*(i+1), :] = X_meas[tstIDs+num_smpls*i,:]
        U_tst[i*N_tst:N_tst*(i+1), :] = U_meas[tstIDs+num_smpls*i,:]
    # get both the time and spatial upper and lower bounds and use them to get the collocation points
    ub = pts.max(0) # or X_meas.max(0)
    lb = pts.min(0) # or X_meas.min(0)
    num_sensors = Nsp
    num_sensors = num_smpls
    Ns = (N_trn, N_tst)
    subsample_prcntg = smpleprcnt
    length = N_trn*n_t

    return (ss1.entropy, Nsp, Ns, smpleprcnt,
            X_trn, U_trn,
            X_tst, U_tst,
            np.vstack((lb, ub),).T)
        
def PDELearningMatDataVerB(fname:str, Sptldims:int, Ntrn:int, Ntst:int, noisePrcntg:float=0.60, 
                 seed:int=None, to_float:bool=True, **kwargs):
    """
        Revised method of how to get the data from a .mat file that will be used to learn a PDE EQ 
        using the PDE learning Classes seen in the Model.py File. This method/class is a second 
        version of sorts to the LearningMatData classes seen above in that it is built off of those
        classes but simplifies many things. Most important is that this one class is meant to work
        with data that has 1,2, or 3 spatial dimensions and with one of the three Monte Carlo method
        to create the random collocation points. And so unlike with the classes this one is built
        off of, there will not be one version for each Monte Carlo method for data with 1, 2, and 3
        spatial dimensions. As well this class determines the data points for training and in a more
        streamlined way that takes up less code lines and is more along the line how I have created
        similar classes elsewhere in my code library for similar things.
        
        Initialization method for the class. The Initializaton parameters/arguments are defined 
        as follows:
            fname -  file name for the .mat file that contains the training data. The data is expectd to be stored 
                or contained within the .mat file in a specific way; the spatial variable coordinates saved as a 
                variable named 'x', 'y' 'z' and the temporal values stored as a matrix/vector named 't' and the 
                data values at the (x,y,z,t) points saved to a matrix variable named 'usol'. 
            Sptldims - The number of spatial dimension of the data. This argument is used to determine whether
                or not to read the fname data file for variables y,z if it's value is greater than one. The 
                values this parameter can take 1, 2 or 3 currently. 
            split - The percentage to split the data apart into training and validation data. The value given is 
                the percentage that constitutes the training data set. The rest is for the validation set
            subsample_prcntg - The percentage of all the data points that are used found in the .mat file that 
                are used in training the model, either as part of the training or testing data sets). If the given 
                value is 0.10, then only 10 percent of the usol data in the .mat file will randomly selected to be 
                split into training and testing set according to the given value of the split parameter.
            noisePrcntg - The amount of noise to add to the data set. The noise is random sample from normal 
                distibution with 0 mean (mu = 0) and standard deviation equal to the standard deviation of 
                the entire usol data (may be changed later to just the traing and testing data).
            Ncp -  the number of collocation points used in eveluating the models candidate libray terms
                and thus for determining/infering a (reduced order or surogate)  modeling PDE equation for the data.
            to_float -  boolean argument indicating to have torch tensors be of torch.float32 or (True) or 
                torch.float64 data types when transfering from numpy to torch. 
            seed - list argument that contains the seeds values for numpy and scipy. The seeds are used to create 
                the rng method that control which points are selected to be in the training, testing and validation
                data sets. This argument is for reproducibility of results though usage of this does not garuentee 
                that the results are reproducable as it has been seen with usage of a computer cluster that even 
                when giving the same seed value, the randomly determined values from the seeded rng machine can be 
                different between one run to another. Specifically on said cluster for a job that was run over 4 
                GPUs, when the speed argument was passed from here was passed to the SeedSequence() function 
                with in the numpy.random module, the resulting entropies was not the same on each of the GPUs,
            to_float - boolean parameter that indicates whether the data when it is eventually passed to torch needs
                to have a dtype of toch.float32 - True (normal 32 bit float data type) or if it needs to be 
                torch.float64 - False (normal 64 bit data type i.e double)

        TODO: 08/02/2024 - 
            (1). Finish input argument checking
            (2). Better method for the fname checking or just maybe looking to path library and os library
                 stuff to see if the file exists and things like that. 
            (3) Change up the rng seed stuff. It may be best not to have a seed value passed by a user and 
                instead get the seeds using the SeedSequence().entropy thing, save it to a class variables
                and then have as a **kwargs arguments for restarting that involve seed values.  ¯ \\ _ (ツ) _ // ¯
        """
    
    # TODO: Finish input argument checking
    if not isinstance(fname, str):
        raise TypeError(f"The fname argument/paramter is expected to be a str not a {type(fname).__name__}")
    elif '.mat' not in fname:
        raise ValueError(f"The given fname str parameter is not a .mat file which is the expected file type.")
    if not isinstance(Sptldims, int):
        raise TypeError(f"The Sptldims argument/paramter is expected to be a int type object not a {type(Sptldims).__name__}")
    elif Sptldims<=0 or Sptldims>=4:
        raise ValueError(f"The given Sptldims parameter needs be a value of 1,2, or 3. What was passed = {Sptldims}.")
    elif Ntrn <=0:
        raise ValueError(f"The Ntrn argument needs to be a POSITVE integer. What was given was {Ntrn}")
    if not isinstance(Ntst, int):
        raise TypeError(f"The Ntst argument NEEDS to be a positive integer value not a {type(Ntst).__name__}")
    elif Ntst <=0:
        raise ValueError(f"The Ntst argument needs to be a POSITVE integer. What was given was {Ntst}")
        
        
    # only reach this point seeds is either a list of just None
    if seed and (not isinstance(seed, int)):
        raise TypeError(f"seed arguement needs to be a positive integer greater than 0")
    if seed==None:
        seed = SeedSequence().entropy
    if seed<1:
        raise ValueError(f"seed arguement needs to be a positive integer greater than 0")
    ss1 = SeedSequence(seed)
    print(f"The entropy or np seed used for seeding the default rng routine is {ss1.entropy}")
    # self.NPentropy = ss1.entropy
    rng = default_rng(seed=ss1)
    if not isinstance(to_float, bool):
        raise TypeError("The to_float argument needs to be a boolean valued argument!")
    # self._to_float = to_float
    # data = sio.loadmat(os.getcwd()+'/'+fname)
    data = sio.loadmat(fname)
    # get the x and t points at column vectors/arrays. Then create a mesh grid of the points
    x = np.real(data['x'].flatten()[:, None])
    t = np.real(data['t'].flatten()[:, None])
    sol = np.real(data['usol'])
    if isinstance(noisePrcntg, (int, float)):
        if noisePrcntg >=0 and noisePrcntg <=1.0:
            sol = sol + rng.normal(loc=0, scale=noisePrcntg*np.std(sol) , size=sol.shape)
            # self.noisePrcntg = noisePrcntg
        elif noisePrcntg>1.0:
            msg = ('WARNING; Given float noisePrcntg value is either is greater than 1.0 and so to convert it to\n'
                   'a decimal value it will be divided by 100 as we are interpretting any value greater than 1.0\n'
                   'to be a percentage and thus to convert out of a percentage (i.e p%) divided by 100 and use \n'
                   'that value (i.e p/100).')
            warnings.warn(msg, stacklevel=2)
            sol = sol + rng.normal(loc=0, scale=(noisePrcntg/100)*np.std(sol) , size=sol.shape)
            noisePrcntg = (noisePrcntg/100)
        else:
            raise ValueError(f"The noisePrcntg input value that was given was a negative number and it needs to be positive")
    else:
        msg = ('The user given noisePrcntg function argument/parameter is not a floating number - Will not use any noise')
        warnings.warn(msg, stacklevel=2)
    # All the data fines regardless of spatial dims has x, t, and sol in the data file now read if there is more. 
    if Sptldims==2:
        y = np.real(data['y'].flatten()[:, None])
        t = np.real(data['t'].flatten()[:, None])
        n_x, n_y, n_t = x.shape[0], y.shape[0], t.shape[0]
        Nsp =n_x*n_y
        # X, Y, T = np.meshgrid(x, y, t)  # with indexing parameter 'xy' value shape is (n_y, n_x, n_t)
        T, X, Y = np.meshgrid(t, x, y, indexing='ij') # with indexing parameter 'ij' shape is (n_t, n_x, n_y)
        if X.shape!=sol.shape:
            lst = []
            trgts = np.array(X.shape, dtype=int)
            inpts = np.array(sol.shape, dtype=int)
            for i in range(trgts.shape[0]):
                lst.append(np.nonzero(inpts[i]==trgts)[0].item())
            sol= np.transpose(sol, axes=tuple(lst))
            del lst
            sol = np.transpose(sol,  axes=(1,0,2))
        # pts = np.hstack((np.copy(X.reshape(-1,1,order='F')), np.copy(Y.reshape(-1,1,order='F')), np.copy(T.reshape(-1,1,order='F'))))
        pts = np.hstack((np.copy(X.reshape(-1,1,order='C')), np.copy(Y.reshape(-1,1,order='C')), np.copy(T.reshape(-1,1,order='C'))))
        # U = np.copy(sol.reshape(-1,1, order='F'))
        U = np.copy(sol.reshape(-1,1, order='C'))
    elif Sptldims==3:
        y = np.real(data['y'].flatten()[:, None])
        z = np.real(data['z'].flatten()[:, None])
        t = np.real(data['t'].flatten()[:, None])
        n_x, n_y, n_z, n_t = x.shape[0], y.shape[0], z.shape[0], t.shape[0]
        Nsp =n_x*n_y*n_z
        Y, Z, X, T = np.meshgrid(y, z, x, t)  # with indexing parameter. value shape is (n_z, n_y, n_x, n_t)
        if X.shape!=sol.shape:
            lst = []
            trgts = np.array(X.shape, dtype=int)
            inpts = np.array(sol.shape, dtype=int)
            for i in range(trgts.shape[0]):
                lst.append(np.nonzero(inpts[i]==trgts)[0].item())
            sol= np.transpose(sol, axes=tuple(lst))
            del lst
            sol = np.transpose(sol,  axes=(1,0,2,3))
        # pts = np.hstack((np.copy(X.reshape(-1,1,order='F')), np.copy(Y.reshape(-1,1,order='F')), np.copy(Z.reshape(-1,1,order='F')), np.copy(T.reshape(-1,1,order='F'))))
        pts = np.hstack((np.copy(X.reshape(-1,1,order='C')), np.copy(Y.reshape(-1,1,order='C')), np.copy(Z.reshape(-1,1,order='C')), np.copy(T.reshape(-1,1,order='C'))))
        # U = np.copy(sol.reshape(-1,1, order='F'))
        U = np.copy(sol.reshape(-1,1, order='C'))
    else: # just revert back down to one spatial dim
        n_x, n_t = x.shape[0], t.shape[0]
        Nsp =n_x
        X, T = np.meshgrid(x, t) # the shape of each of these is n_t by n_x
        if X.shape!=sol.shape:
            sol = sol.T
        pts = np.hstack((np.copy(X.reshape(-1,1, order='C')), np.copy(T.reshape(-1,1, order='C'))))
        U = np.copy(sol.reshape(-1,1,order='C'))
    N = pts.shape[0]
    if Ntrn+Ntst>N:
        msg = ("WARNING; The given values for the number of data points to training on (Ntrn) and that the\n"
               "number of data points to test on (Ntst) combined is greater than the number of (spatio-temporal)\n"
                "data points in the given .mat data file. To handle this training will be done on using 80% of\n"
                "all the spatio-temporal data points leaving 20% to test on and not the amount that was\n"
               "wanted via the user Ntrn and Ntst argument.")
        Ntrn = int(0.80*N)
        Ntst = int(0.20*N)
        warnings.warn(msg, stacklevel=2)
    # Now split the data that will be use to train the model and extra points since paper say nothing 
    # about test/validation set/s.
    if 'PdeReadMatch' in kwargs.keys():
        mtch = kwargs['PdeReadMatch']
    else: mtch=False
    if mtch:
        trnIds = rng.choice(N, Ntrn, replace=False).astype(np.int32)
        Js = trnIds//n_t
        Is = trnIds - Js*n_t
        correctedTrnIds = Is*n_x + Js
        probs = np.zeros((N,),)
        probs[np.setdiff1d(np.arange(N), trnIds, assume_unique=True)] = 1
        probs = probs/probs.sum()
        tstIds = rng.choice(N, Ntst, replace=False,p=probs).astype(np.int32)
        Js = tstIds//n_t
        Is = tstIds - Js*n_t
        correctedTstIds = Is*n_x + Js
        valIds = np.setdiff1d(np.arange(N), np.concatenate((trnIds, tstIds), axis=0), assume_unique=True)
        correctedTrnIds.sort()
        correctedTstIds.sort()
        valIds.sort()
        X_trn = pts[correctedTrnIds,:]
        X_tst = pts[correctedTstIds,:]
        X_val = pts[valIds,:]
        U_trn = U[correctedTrnIds,:]
        U_tst = U[correctedTstIds,:]
        U_val = U[valIds,:]
    else: 
        trnIds = rng.choice(N, Ntrn, replace=False).astype(np.int32)
        probs = np.zeros((N,),)
        probs[np.setdiff1d(np.arange(N), trnIds, assume_unique=True)] = 1
        probs = probs/probs.sum()
        tstIds = rng.choice(N, Ntst, replace=False,p=probs).astype(np.int32)
        valIds = np.setdiff1d(np.arange(N), np.concatenate((trnIds, tstIds),axis=0), assume_unique=True)
        trnIds.sort()
        tstIds.sort()
        valIds.sort()
        X_trn = pts[trnIds,:]
        X_tst = pts[tstIds,:]
        X_val = pts[valIds,:]
        U_trn = U[trnIds,:]
        U_tst = U[tstIds,:]
        U_val = U[valIds,:]

    # get both the time and spatial upper and lower bounds and use them to get the collocation points
    ub = pts.max(0) # or X_meas.max(0)
    lb = pts.min(0) # or X_meas.min(0)
    num_sensors = Nsp
    Ns = (Ntrn, Ntst)
    subsample_prcntg = np.around((Ntrn+Ntst)/N, 3,)
    length = Ntrn

    return (ss1.entropy, Nsp, Ns, subsample_prcntg,
            X_trn, U_trn,
            X_tst, U_tst,
            np.vstack((lb, ub),).T)

# Data loaders and Stuff for distributed/parallel models.
def  ParallelPDELearningMatDataMaker(tmpName:str, fname:str, Sptldims:int, split:float=0.80, smpleprcnt:float=0.20, Ncp:int=10000, 
                                noisePrcntg:float=0.15, qrng:str='halton', seeds:Union[list, np.ndarray]=None, 
                                to_float:bool=True, **kwargs) -> None:
        """
        Initialization method for the class. The Initializaton parameters/arguments are defined 
        as follows:
            fname -  file name for the .mat file that contains the training data. The data is expectd to be stored 
                or contained within the .mat file in a specific way; the spatial variable coordinates saved as a 
                variable named 'x', 'y' 'z' and the temporal values stored as a matrix/vector named 't' and the 
                data values at the (x,y,z,t) points saved to a matrix variable named 'usol'. 
            Sptldims - The number of spatial dimension of the data. This argument is used to determine whether
                or not to read the fname data file for variables y,z if it's value is greater than one. The 
                values this parameter can take 1, 2 or 3 currently. 
            split - The percentage to split the data apart into training and validation data. The value given is 
                the percentage that constitutes the training data set. The rest is for the validation set
            subsample_prcntg - The percentage of all the data points that are used found in the .mat file that 
                are used in training the model, either as part of the training or testing data sets). If the given 
                value is 0.10, then only 10 percent of the usol data in the .mat file will randomly selected to be 
                split into training and testing set according to the given value of the split parameter.
            noisePrcntg - The amount of noise to add to the data set. The noise is random sample from normal 
                distibution with 0 mean (mu = 0) and standard deviation equal to the standard deviation of 
                the entire usol data (may be changed later to just the traing and testing data).
            Ncp -  the number of collocation points used in eveluating the models candidate libray terms
                and thus for determining/infering a (reduced order or surogate)  modeling PDE equation for the data.
            qrng- String argument indicating which Quasi-Monte Carlo (QMC) to use to generate the collocation points
                Value can be sobol for Sobol squence, halton for Halton squence and lhc for Latin Hybercupe sampling.
                Regarding Latin hypercube sampling, an additional argument needs to be passed indicating the strength
                of the sampleing with a value of either 1 or 2. If no value is given default value is strength=1
            to_float -  boolean argument indicating to have torch tensors be of torch.float32 or (True) or 
                torch.float64 data types when transfering from numpy to torch. 
            seeds - list argument that contains the seeds values for numpy and scipy. The seeds are used to create 
                the rng method that control which points are selected to be in the training, testing and validation
                data sets. This argument is for reproducibility of results though usage of this does not garuentee 
                that the results are reproducable as it has been seen with usage of a computer cluster that even 
                when giving the same seed value, the randomly determined values from the seeded rng machine can be 
                different between one run to another. Specifically on said cluster for a job that was run over 4 
                GPUs, when the speed argument was passed from here was passed to the SeedSequence() function 
                with in the numpy.random module, the resulting entropies was not the same on each of the GPUs,
            to_float - boolean parameter that indicates whether the data when it is eventually passed to torch needs
                to have a dtype of toch.float32 - True (normal 32 bit float data type) or if it needs to be 
                torch.float64 - False (normal 64 bit data type i.e double)

        TODO: 08/02/2024 - 
            (1). Finish input argument checking
            (2). Better method for the fname checking or just maybe looking to path library and os library
                 stuff to see if the file exists and things like that. 
            (3) Change up the rng seed stuff. It may be best not to have a seed value passed by a user and 
                instead get the seeds using the SeedSequence().entropy thing, save it to a class variables
                and then have as a **kwargs arguments for restarting that involve seed values.  ¯ \\ _ (ツ) _ // ¯
        """
        # TODO: Finish input argument checking
        if not isinstance(fname, str):
            raise TypeError(f"The fname argument/paramter is expected to be a str not a {type(fname).__name__}")
        elif '.mat' not in fname:
            raise ValueError(f"The given fname str parameter is not a .mat file which is the expected file type.")
        if not isinstance(Sptldims, int):
            raise TypeError(f"The Sptldims argument/paramter is expected to be a int type object not a {type(Sptldims).__name__}")
        elif Sptldims<=0 or Sptldims>=4:
            raise ValueError(f"The given Sptldims parameter needs be a value of 1,2, or 3. What was passed = {Sptldims}.")
        if not isinstance(split, float):
            msg = (f"The split input argument that splits the learning data set into training and testing needs to\n"
                   f"be a float argument between 0 and 1.0. What you gave is a {type(split).__name__} type argument")
            raise TypeError(msg)
        if not 0.0<split<1.0:
            msg = (f"The split input argument needs to be a float be in the inteval (0.0, 1.0) and what you gave {split}\n"
                   "is not within  that interval.")
            raise ValueError(msg)
        if not isinstance(smpleprcnt, float):
            msg = (f"The smpleprcnt input argument that splits the data set into learning and validation sets needs to\n"
                   f"be a float argument between 0 and 1.0. What you gave is a {type(smpleprcnt).__name__} type argument")
            raise TypeError(msg)
        if not 0.0<smpleprcnt<1.0:
            msg = (f"The smpleprcnt input argument needs to be a float be in the inteval (0.0, 1.0) and what you gave {smpleprcnt}\n"
                   "is not within that interval.")
        if not isinstance(Ncp, int):
            raise TypeError(f"The collolcat_pts argument NEEDS to be a positive integer value not a {type(Ncp).__name__}")
        elif Ncp <=0:
            msg = ("WARNING - The user given/passed value of the collolcat_pts agument is a non-positive integer\n"
                   "The value should be a positive integer. To handle this problem, the number of collocation point\n"
                   "will be set to the same number of data training points.")
            warnings.warn(message=msg, stacklevel=2)
            Ncp=1
        if not isinstance(qrng, str):
            raise TypeError("The qrng argument NEEDS to be an str argument 0 either sobol, halton or lhc!")
        elif qrng not in ['halton', 'sobol', 'lhc']:
            msg = ("WARNING - The user given/passed value of the qrng agument is not one of the acceptable values.\n"
                   "The value should be \'halton\', \'sobol\' or \'lhc\' and nothing else. To handle this problem,\n"
                   "default value of \'halton\' will be used.")
            warnings.warn(message=msg, stacklevel=2)
            qrng = 'halton'
        if not isinstance(seeds, list) and seeds!=None:
            msg = ("WARNING - The User given argument for the seeds is {} which is not a list or None\n".format(type(seeds)),
                   "To handle this Fuck Up, no seed values will be passed to the generators")
            warnings.warn(msg, stacklevel=2)
            seeds=None
        # only reach this point seeds is either a list of just None
        if isinstance(seeds, list):
            # so is a list and thus check that the seeds positive integers and at minimum 1 seed
            ints =  np.any(False == np.array([isinstance(el, int) for el in seeds]))
            if ints and len(seeds)==1: np_seed, scipy_seed = np.abs(seeds[0]), np.abs(seeds[0])
            elif ints and len(seeds)>=2: np_seed, scipy_seed = np.abs(seeds[0]), np.abs(seeds[1])
            else: 
                msg = ("WARNING - The User given argument for the seeds is a list but it either has a length of 0\n"
                   "or one of its elements is not an integer. So we will be using the default value of None for the seeds")
                warnings.warn(msg, stacklevel=2)
                np_seed, scipy_seed = SeedSequence().entropy, SeedSequence().entropy
        else: # so not a list and thus is equal to None
            np_seed, scipy_seed = SeedSequence().entropy, SeedSequence().entropy
        ss1 = SeedSequence(np_seed)
        ss2 = scipy_seed
        print(f"The entropy or np seed used for seeding the default rng routine is {ss1.entropy}")
        print(f"While the scipy seed used for seeding the QuasiMonteCarlo scipy genorator is {ss2}")
        rng = default_rng(seed=ss1)
        if not isinstance(to_float, bool):
            raise TypeError("The to_float argument needs to be a boolean valued argument!")
        data = sio.loadmat(os.getcwd()+'/'+fname)
        # get the x and t points at column vectors/arrays. Then create a mesh grid of the points
        x = np.real(data['x'].flatten()[:, None])
        t = np.real(data['t'].flatten()[:, None])
        sol = np.real(data['usol'])
        if isinstance(noisePrcntg, (int, float)):
            if noisePrcntg >=0 and noisePrcntg <=1.0:
                sol = sol + rng.normal(loc=0, scale=noisePrcntg*np.std(sol) , size=sol.shape)
                noisePrcntg = noisePrcntg
            elif noisePrcntg>1.0:
                msg = ('WARNING; Given float noisePrcntg value is either is greater than 1.0 and so to convert it to\n'
                       'a decimal value it will be divided by 100 as we are interpretting any value greater than 1.0\n'
                       'to be a percentage and thus to convert out of a percentage (i.e p%) divided by 100 and use \n'
                       'that value (i.e p/100).')
                warnings.warn(msg, stacklevel=2)
                sol = sol + rng.normal(loc=0, scale=(noisePrcntg/100)*np.std(sol) , size=sol.shape)
                noisePrcntg = (noisePrcntg/100)
            else:
                raise ValueError(f"The noisePrcntg input value that was given was a negative number and it needs to be positive")
        else:
            msg = ('The user given noisePrcntg function argument/parameter is not a floating number - Will not use any noise')
            warnings.warn(msg, stacklevel=2)
        # All the data fines regardless of spatial dims has x, t, and sol in the data file now read if there is more. 
        if Sptldims==2:
            y = np.real(data['y'].flatten()[:, None])
            t = np.real(data['t'].flatten()[:, None])
            n_x, n_y, n_t = x.shape[0], y.shape[0], t.shape[0]
            Nsp =n_x*n_y
            X, Y, T = np.meshgrid(x, y, t)  # with indexing parameter. value shape is (n_y, n_x, n_t)
            if X.shape!=sol.shape:
                lst = []
                trgts = np.array(X.shape, dtype=int)
                inpts = np.array(sol.shape, dtype=int)
                for i in range(trgts.shape[0]):
                    lst.append(np.nonzero(inpts[i]==trgts)[0].item())
                sol= np.transpose(sol, axes=tuple(lst))
                del lst
            pts = np.hstack((np.copy(X.reshape(-1,1,order='F')), np.copy(Y.reshape(-1,1,order='F')), np.copy(T.reshape(-1,1,order='F'))))
            U = np.copy(sol.reshape(-1,1, order='F'))
        elif Sptldims==3:
            y = np.real(data['y'].flatten()[:, None])
            z = np.real(data['z'].flatten()[:, None])
            t = np.real(data['t'].flatten()[:, None])
            n_x, n_y, n_z, n_t = x.shape[0], y.shape[0], z.shape[0], t.shape[0]
            Nsp =n_x*n_y*n_z
            Y, Z, X, T = np.meshgrid(y, z, x, t)  # with indexing parameter. value shape is (n_z, n_y, n_x, n_t)
            if X.shape!=sol.shape:
                lst = []
                trgts = np.array(X.shape, dtype=int)
                inpts = np.array(sol.shape, dtype=int)
                for i in range(trgts.shape[0]):
                    lst.append(np.nonzero(inpts[i]==trgts)[0].item())
                sol= np.transpose(sol, axes=tuple(lst))
                del lst
            pts = np.hstack((np.copy(X.reshape(-1,1,order='F')), np.copy(Y.reshape(-1,1,order='F')), np.copy(Z.reshape(-1,1,order='F')), np.copy(T.reshape(-1,1,order='F'))))
            U = np.copy(sol.reshape(-1,1, order='F'))
        else: # just revert back down to one spatial dim
            n_x, n_t = x.shape[0], t.shape[0]
            Nsp =n_x
            X, T = np.meshgrid(x, t) # the shape of each of these is n_t by n_x
            if X.shape!=sol.shape:
                sol = sol.T
            pts = np.hstack((np.copy(X.reshape(-1,1)), np.copy(T.reshape(-1,1))))
            U = np.copy(sol.reshape(-1,1,order='C'))
        
        num_smpls = int(np.around(Nsp*smpleprcnt))
        N_trn = int(np.around(num_smpls*split))
        N_tst = int(num_smpls-N_trn)
        innerIDs = rng.choice(Nsp, num_smpls, replace=False).astype(np.int32)
        innerIDs.sort() # NOTE - may not need this line
        X_meas = np.empty((num_smpls*n_t,Sptldims+1), dtype=pts.dtype)
        U_meas = np.empty((num_smpls*n_t,1), dtype=sol.dtype)
        for i in range(0, n_t):
            X_meas[i*num_smpls:num_smpls*(i+1), :] = pts[innerIDs+Nsp*i,:]
            U_meas[i*num_smpls:num_smpls*(i+1), :] = U[innerIDs+Nsp*i,:]
        # Now split the data that will be use to train the model into a train and test sets.
        X_trn = np.empty((N_trn*n_t, Sptldims+1), dtype=X_meas.dtype,)
        U_trn = np.empty((N_trn*n_t, 1), dtype=U_meas.dtype,)
        X_tst = np.empty(((num_smpls-N_trn)*n_t, Sptldims+1), dtype=X_meas.dtype,)
        U_tst = np.empty(((num_smpls-N_trn)*n_t, 1), dtype=U_meas.dtype,)
        for i in range(0,n_t): 
            trnIDs = rng.choice(num_smpls, N_trn, replace=False).astype(np.int32)
            trnIDs.sort()
            tstIDs = np.setdiff1d(np.arange(num_smpls), trnIDs, assume_unique=True)
            X_trn[i*N_trn:N_trn*(i+1), :] = X_meas[trnIDs+num_smpls*i,:]
            U_trn[i*N_trn:N_trn*(i+1), :] = U_meas[trnIDs+num_smpls*i,:]
            X_tst[i*N_tst:N_tst*(i+1), :] = X_meas[tstIDs+num_smpls*i,:]
            U_tst[i*N_tst:N_tst*(i+1), :] = U_meas[tstIDs+num_smpls*i,:]
        # get both the time and spatial upper and lower bounds and use them to get the collocation points
        ub = pts.max(0) # or X_meas.max(0)
        lb = pts.min(0) # or X_meas.min(0)
        num_sensors = num_smpls
        Ns = (N_trn, N_tst)
        subsample_prcntg = smpleprcnt
        length = N_trn*n_t
        NPentropy = ss1.entropy
        SciPyEntropy  = ss2
        scipy_seed = scipy_seed
        ub = pts.max(0) # or X_meas.max(0)
        lb = pts.min(0) # or X_meas.min(0)
        Ns = (N_trn, N_tst)
        subsample_prcntg = smpleprcnt
        col_num = int(Ncp)
        if Ncp < length:
            msg = ("Warning - The number of collocation points is less than the number of data points for learning function\n"
                   "Will increase the number of collcation points to be the same number of data points.")
            warnings.warn(message=msg, stacklevel=2)
            col_num = int(length)
        if qrng =='sobol':
            sampler = qmc.Sobol(Sptldims+1, seed=ss2)
            col_num = int(np.ceil(col_num/length)*length)
            col_rat = int(np.ceil(col_num/length))
            m = int(np.ceil(np.log2(col_num)))
            colpts = sampler.random_base2(m=m)
        elif qrng=='halton':
            sampler = qmc.Halton(d=Sptldims+1, optimization="random-cd", seed=ss2)
            col_num = int(np.ceil(col_num/length)*length)
            col_rat = int(np.ceil(col_num/length))
            colpts = sampler.random(n=col_num, workers=-1)
        elif qrng=='lhc':
            if 'strength' in kwargs.keys():
                strength = kwargs['strength']
                if not isinstance(strength, int):
                    msg = ("Warning - User indicated Latin HyperCube Sampling and the passed value\n"
                           "for \'strength\' was not the correct/expected object type of int. So\n"
                            "we will use the default value of 1 for \'strength\'")
                    warnings.warn(message=msg, stacklevel=2)
                    strength=1
            else: # no strength argument passed:
                msg = ("Warning - User indicated Latin HyperCube Sampling but did not pass a \n"
                       "value for the 'strength' argument. To handle this error the default \n"
                       "value of one will be used.")
                warnings.warn(message=msg, stacklevel=2)
                strength=1
            # not checking the strength value - at this point konw it is an integer value thing so let scipy handle the error
            if col_num % length !=0:
                col_num = int(np.ceil(col_num/length)*length)
            col_rat = int(np.ceil(col_num/length))
            strength = strength
            sampler = qmc.LatinHypercube(Sptldims+1,strength=strength, seed=ss2)
            if strength==1:
                colpts = sampler.random(n=col_num)
            elif strength==2:
                # Need to find prime p such that p^2 >= col_num
                # p = n_primes(1, np.floor(np.sqrt(col_num))).item()
                p = 5
                colpts = sampler.random(n=p**2)
        
        matDic = {"X_trn":X_trn, "U_trn":U_trn, "X_tst":X_tst, "F_tst":U_tst, "length":length, "colpts":colpts,
                  "col_num":col_num, "col_rat":col_rat, "num_sensors":num_sensors, "Ns":Ns, 
                  "subsample_prcntg":subsample_prcntg, "noisePrcntg":noisePrcntg,"to_float":to_float, 
                  "NPentropy":str(NPentropy), "SciPyEntropy":str(SciPyEntropy), "ub":ub, "lb":lb,}
        og_dir = os.getcwd()
        try:
            os.mkdir('TempDataFiles')
        except FileExistsError:
            print('TempDataFiles Directory already exists so did not create it')
        try:
            os.chdir('TempDataFiles')
        except (OSError, FileNotFoundError, PermissionError, NotADirectoryError):
            print('Could not change directory to TempDataFiles. Results will be writen to file in CWD')

        sio.savemat(file_name=tmpName+'.mat', mdict=matDic)
        os.chdir(og_dir)

class ParallelPDELearningMatData(Dataset):
    """
        Equivalent class for as the similiarly titled class seen above that if it was not obvious 
        as this one's tilted is that one's preappened with the Parallel world indicating that this
        class is meant to be used for Parallel models - models that are using multiple GPUS to do
        the learning and training. However this class really do not do any work. The main work is 
        done in the ParallelPDELearningMatDataMaker function so as to ensure that the dataset is 
        properly parallelized. This has been done through the creation of a temporary data file
        that contains the partitioned training and testing sets, as well as all parameters used to 
        in the ParallelPDELearningMatDataMaker function. This was done because when a user wanted 
        have the model trained using X% of the data set the selected X% was properly split across
        the GPUs. The previous method for how this was done had problems in randomly selecting the 
        same spatio temporal data points that constitute X% of the data on the GPUs when the method
        for randomly selecting the points had been given the same seed value. 
    """
    def __init__(self, folder:str, readFile:str) -> None:
        """
        Initialization method for the class. Function parameters (input arguments) are as follows:
            folder - The folder that contains the .mat data file that was created from running the
                ParallelPDELearningMatDataMaker function.
            readFile - The .mat file name from where to read the data from. Should be the same name
                as that used as the tmpName parameter in the ParallelPDELearningMatDataMaker 
                function. 
        
        TODO 09/02/2024:
            (1) Input argument checks
            (2) see if we can use the pathlib or os library method to check that files and stuff exit
        """
        super().__init__()
        # TODO: Finish input argument checking
        fname = folder+'/'+readFile
        data = sio.loadmat(os.getcwd()+'/'+fname)
        self.X_trn = np.real(data['X_trn'])
        self.U_trn = np.real(data['U_trn'])
        self.X_tst = np.real(data['X_tst'])
        self.F_tst = np.real(data['F_tst'])
        self.length = int(data['length'])
        self.colpts = np.real(data['colpts'])
        self.col_num = int(data['col_num'])
        self.col_rat = int(data['col_rat'])
        self.num_sensors = int(data['num_sensors'])
        self.Ns = int(data['Ns'])
        self.subsample_prcntg = float(data['subsample_prcntg'])
        self.noisePrcntg = float(data['noisePrcntg'])
        self.to_float = bool(data['to_float'])
        self.NPentropy = int(data['NPentropy'])
        self.SciPyEntropy = int(data['SciPyEntropy'])
        self.scipy_seed = int(data['scipy_seed'])
        self.ub = np.real(data['ub'])
        self.lb = np.real(data['lb'])
        print(f"Loaded Data from the tempory file was determined using a Numpy seed/entropy value of {self.NPentropy}")
        print(f"Collocation Points where determined using the a Numpy seed/entropy value of {self.NPentropy}")

    def __len__(self):
        return self.length
    
    def __getitem__(self,  idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self._to_float:
            return self.X_trn[idx, :].astype(np.float32), self.U_trn[idx, :].astype(np.float32), self.colpts[idx*self.col_rat:(idx+1)*self.col_rat, :].astype(np.float32)
        else:
            return self.X_trn[idx, :], self.U_trn[idx, :], self.colpts[idx*self.col_rat:(idx+1)*self.col_rat, :]

class PdeReadLearnerMatDataVerB(Dataset):   
    def __init__(self, fname:str, Sptldims:int, split:float=0.80, smpleprcnt:float=0.20, Ncp:int=10000, noise:float=0.15, 
                 npSeed:int=None, to_float:bool=True, **kwargs) -> None:
        """
        Initialization method for the class. The Initializaton parameters/arguments are defined 
        as follows:
            fname -  file name for the .mat file that contains the training data. The data is expectd to be stored 
                or contained within the .mat file in a specific way; the spatial variable coordinates saved as a 
                variable named 'x', 'y' 'z' and the temporal values stored as a matrix/vector named 't' and the 
                data values at the (x,y,z,t) points saved to a matrix variable named 'usol'. 
            Sptldims - The number of spatial dimension of the data. This argument is used to determine whether
                or not to read the fname data file for variables y,z if it's value is greater than one. The 
                values this parameter can take 1, 2 or 3 currently. 
            Ntrn - The number of data points to train on/with. Needs to be an integer. If the number combined
                with the Ntst number is greater than the number of data points found in the .mat file a warning
                with be given and the number used to train will be 80 percent of the maximum available. 
            Ntrn - The number of data points to test on/with. Needs to be an integer. If the number combined
                with the Ntrn number is greater than the number of data points found in the .mat file a warning
                with be given and the number used to train will be 20 percent of the maximum available. 
            noisePrcntg - The amount of noise to add to the data set. The noise is random sample from normal 
                distibution with 0 mean (mu = 0) and standard deviation equal to the standard deviation of 
                the entire usol data (may be changed later to just the traing and testing data).
            Ncp -  the number of collocation points used in eveluating the models candidate libray terms
                and thus for determining/infering a (reduced order or surogate)  modeling PDE equation for the 
                data.
            to_float -  boolean argument indicating to have torch tensors be of torch.float32 or (True) or 
                torch.float64 data types when transfering from numpy to torch. 
            seeds - list argument that contains the seeds values for numpy and scipy. The seeds are used to create 
                the rng method that control which points are selected to be in the training, testing and validation
                data sets. This argument is for reproducibility of results though usage of this does not garuentee 
                that the results are reproducable as it has been seen with usage of a computer cluster that even 
                when giving the same seed value, the randomly determined values from the seeded rng machine can be 
                different between one run to another. Specifically on said cluster for a job that was run over 4 
                GPUs, when the speed argument was passed from here was passed to the SeedSequence() function 
                with in the numpy.random module, the resulting entropies was not the same on each of the GPUs,
            to_float - boolean parameter that indicates whether the data when it is eventually passed to torch needs
                to have a dtype of toch.float32 - True (normal 32 bit float data type) or if it needs to be 
                torch.float64 - False (normal 64 bit data type i.e double)

        TODO: 08/02/2024 - 
            (1). Finish input argument checking
            (2). Better method for the fname checking or just maybe looking to path library and os library
                 stuff to see if the file exists and things like that. 
            (3). Change up the rng seed stuff. It may be best not to have a seed value passed by a user and 
                instead get the seeds using the SeedSequence().entropy thing, save it to a class variables
                and then have as a **kwargs arguments for restarting that involve seed values.  ¯ \\ _ (ツ) _ // ¯
        """
        super().__init__()
        # TODO: Finish input argument checking
        if not isinstance(fname, str):
            raise TypeError(f"The fname argument/paramter is expected to be a str not a {type(fname).__name__}")
        elif '.mat' not in fname:
            raise ValueError(f"The given fname str parameter is not a .mat file which is the expected file type.")
        if not isinstance(Sptldims, int):
            raise TypeError(f"The Sptldims argument/paramter is expected to be a int type object not a {type(Sptldims).__name__}")
        elif Sptldims<=0 or Sptldims>=4:
            raise ValueError(f"The given Sptldims parameter needs be a value of 1,2, or 3. What was passed = {Sptldims}.")
        if not isinstance(split, float):
            msg = (f"The split input argument that splits the learning data set into training and testing needs to\n"
                   f"be a float argument between 0 and 1.0. What you gave is a {type(split).__name__} type argument")
            raise TypeError(msg)
        if not 0.0<split<1.0:
            msg = (f"The split input argument needs to be a float be in the inteval (0.0, 1.0) and what you gave {split}\n"
                   "is not within  that interval.")
            raise ValueError(msg)
        if not isinstance(smpleprcnt, float):
            msg = (f"The smpleprcnt input argument that splits the data set into learning and validation sets needs to\n"
                   f"be a float argument between 0 and 1.0. What you gave is a {type(smpleprcnt).__name__} type argument")
            raise TypeError(msg)
        if not 0.0<smpleprcnt<1.0:
            msg = (f"The smpleprcnt input argument needs to be a float be in the inteval (0.0, 1.0) and what you gave {smpleprcnt}\n"
                   "is not within that interval.")
        if not isinstance(Ncp, int):
            raise TypeError(f"The Ncp argument NEEDS to be a positive integer value not a {type(Ncp).__name__}")
        elif Ncp <=0:
            msg = ("WARNING - The user given/passed value of the collolcat_pts agument is a non-positive integer\n"
                   "The value should be a positive integer. To handle this problem, the number of collocation point\n"
                   "will be set to the same number of data training points.")
            warnings.warn(message=msg, stacklevel=2)
            Ncp=1
        if not isinstance(npSeed, int) and npSeed!=None:
            msg = ("WARNING - The User given argument for the npSeed is {} which is not an integer (int) \n".format(type(npSeed)),
                   "To handle this Fuck Up, no seed values will be passed to the generators")
            warnings.warn(msg, stacklevel=2)
            npSeed=None
        # only reach this point seeds is either a list of just None
        if isinstance(npSeed, int):
            # so is an int and thus check that the seed is non-negative integers 
            if npSeed<0:
                msg = ("WARNING - The User given argument for the seeds is an integer but its value is less than 0.\n"
                   "So a seed value will be chosen through the numpy SeedSequence().entropy value ")
                warnings.warn(msg, stacklevel=2)
                npSeed =  SeedSequence().entropy
        else: # so not a list and thus is equal to None
            npSeed =  SeedSequence().entropy
        ss1 = SeedSequence(npSeed)
        print(f"The entropy or np seed used for seeding the default rng routine is {ss1.entropy}")
        self.NPentropy = ss1.entropy
        rng = default_rng(seed=ss1)
        if not isinstance(to_float, bool):
            raise TypeError("The to_float argument needs to be a boolean valued argument!")
        self._to_float = to_float
        # data = sio.loadmat(os.getcwd()+'/'+fname)
        data = sio.loadmat(fname)
        # get the x and t points at column vectors/arrays. Then create a mesh grid of the points
        x = np.real(data['x'].flatten()[:, None])
        t = np.real(data['t'].flatten()[:, None])
        sol = np.real(data['usol'])
        if isinstance(noise, float):
            if noise >=0 and noise <=1.0:
                sol = sol + rng.normal(loc=0, scale=noise*np.std(sol) , size=sol.shape)
                self.noisePrcntg = noise
            elif noise>1.0:
                msg = ('WARNING; Given float noise value is either is greater than 1.0 and so to convert it to\n'
                       'a decimal value it will be divided by 100 as we are interpretting any value greater than 1.0\n'
                       'to be a percentage and thus to convert out of a percentage (i.e p%) divided by 100 and use \n'
                       'that value (i.e p/100).')
                warnings.warn(msg, stacklevel=2)
                sol = sol + rng.normal(loc=0, scale=(noise/100)*np.std(sol) , size=sol.shape)
                self.noisePrcntg = (noise/100)
            else:
                raise ValueError(f"The noisePrcntg input value that was given was a negative number and it needs to be positive")
        else:
            msg = ('The user given noisePrcntg function argument/parameter is not a floating number - Will not use any noise')
            warnings.warn(msg, stacklevel=2)
        # All the data fines regardless of spatial dims has x, t, and sol in the data file now read if there is more. 
        if Sptldims==2:
            y = np.real(data['y'].flatten()[:, None])
            t = np.real(data['t'].flatten()[:, None])
            n_x, n_y, n_t = x.shape[0], y.shape[0], t.shape[0]
            Nsp =n_x*n_y
            X, Y, T = np.meshgrid(x, y, t)  # with indexing parameter. value shape is (n_y, n_x, n_t)
            if X.shape!=sol.shape:
                lst = []
                trgts = np.array(X.shape, dtype=int)
                inpts = np.array(sol.shape, dtype=int)
                for i in range(trgts.shape[0]):
                    lst.append(np.nonzero(inpts[i]==trgts)[0].item())
                sol= np.transpose(sol, axes=tuple(lst))
                del lst
            pts = np.hstack((np.copy(X.reshape(-1,1,order='F')), np.copy(Y.reshape(-1,1,order='F')), np.copy(T.reshape(-1,1,order='F'))))
            U = np.copy(sol.reshape(-1,1, order='F'))
        elif Sptldims==3:
            y = np.real(data['y'].flatten()[:, None])
            z = np.real(data['z'].flatten()[:, None])
            t = np.real(data['t'].flatten()[:, None])
            n_x, n_y, n_z, n_t = x.shape[0], y.shape[0], z.shape[0], t.shape[0]
            Nsp =n_x*n_y*n_z
            Y, Z, X, T = np.meshgrid(y, z, x, t)  # with indexing parameter. value shape is (n_z, n_y, n_x, n_t)
            if X.shape!=sol.shape:
                lst = []
                trgts = np.array(X.shape, dtype=int)
                inpts = np.array(sol.shape, dtype=int)
                for i in range(trgts.shape[0]):
                    lst.append(np.nonzero(inpts[i]==trgts)[0].item())
                sol= np.transpose(sol, axes=tuple(lst))
                del lst
            pts = np.hstack((np.copy(X.reshape(-1,1,order='F')), np.copy(Y.reshape(-1,1,order='F')), np.copy(Z.reshape(-1,1,order='F')), np.copy(T.reshape(-1,1,order='F'))))
            U = np.copy(sol.reshape(-1,1, order='F'))
        else: # just revert back down to one spatial dim
            n_x, n_t = x.shape[0], t.shape[0]
            Nsp =n_x
            X, T = np.meshgrid(x, t) # the shape of each of these is n_t by n_x
            if X.shape!=sol.shape:
                sol = sol.T
            pts = np.hstack((np.copy(X.reshape(-1,1, order='C')), np.copy(T.reshape(-1,1, order='C'))))
            U = np.copy(sol.reshape(-1,1,order='C'))
        
        num_smpls = int(np.around(Nsp*smpleprcnt))
        val_num = int(Nsp - num_smpls)
        Ntrn = int(np.around(num_smpls*split))
        Ntst = int(num_smpls-Ntrn)
        innerIDs = rng.choice(Nsp, num_smpls, replace=False).astype(np.int32)
        innerIDs.sort() # NOTE - may not need this line
        valIDS = np.setdiff1d(np.arange(Nsp), innerIDs, assume_unique=True)
        X_meas = np.empty((num_smpls*n_t,Sptldims+1), dtype=pts.dtype)
        self.X_val = np.empty((val_num*n_t,Sptldims+1), dtype=pts.dtype)
        U_meas = np.empty((num_smpls*n_t,1), dtype=pts.dtype)
        self.U_val = np.empty((val_num*n_t,1), dtype=pts.dtype)
        for i in range(0, n_t):
            X_meas[i*num_smpls:num_smpls*(i+1), :] = pts[innerIDs+Nsp*i,:]
            self.X_val[i*val_num:val_num*(i+1), :] = pts[valIDS+Nsp*i,:]
            U_meas[i*num_smpls:num_smpls*(i+1), :] = U[innerIDs+Nsp*i,:]
            self.U_val[i*val_num:val_num*(i+1), :] = U[valIDS+Nsp*i,:]
        # Now split the data that will be use to train the model into a train and test sets.
        self.X_trn = np.empty((Ntrn*n_t, Sptldims+1), dtype=X_meas.dtype,)
        self.U_trn = np.empty((Ntrn*n_t, 1), dtype=U_meas.dtype,)
        self.X_tst = np.empty(((num_smpls-Ntrn)*n_t, Sptldims+1), dtype=X_meas.dtype,)
        self.U_tst = np.empty(((num_smpls-Ntrn)*n_t, 1), dtype=U_meas.dtype,)
        for i in range(0,n_t): 
            trnIDs = rng.choice(num_smpls, Ntrn, replace=False).astype(np.int32)
            trnIDs.sort()
            tstIDs = np.setdiff1d(np.arange(num_smpls), trnIDs, assume_unique=True)
            self.X_trn[i*Ntrn:Ntrn*(i+1), :] = X_meas[trnIDs+num_smpls*i,:]
            self.U_trn[i*Ntrn:Ntrn*(i+1), :] = U_meas[trnIDs+num_smpls*i,:]
            self.X_tst[i*Ntst:Ntst*(i+1), :] = X_meas[tstIDs+num_smpls*i,:]
            self.U_tst[i*Ntst:Ntst*(i+1), :] = U_meas[tstIDs+num_smpls*i,:]

        # get both the time and spatial upper and lower bounds and use them to get the collocation points
        self.ub = pts.max(0) 
        self.lb = pts.min(0) 
        self.Ntrn = Ntrn*n_t
        self.Ntst = Ntst*n_t
        self.Sptldims = Sptldims
        self.length = self.X_trn.shape[0]
        self.NumColPnts = int(Ncp)
        if 'col_ratio' in kwargs.keys():
            self.NumColPnts = self.length*kwargs['col_ratio']
            Ncp = self.length * kwargs['col_ratio']
        if Ncp < self.length:
            msg = ("Warning - The number of collocation points is less than the number of data points for learning function\n"
                   "Will increase the number of collcation points to be the same number of data points.")
            warnings.warn(message=msg, stacklevel=2)
            self.NumColPnts = int(self.length)
        self.NumColPnts = int(np.ceil(self.NumColPnts/self.length)*self.length)
        self.col_rat = int(np.ceil(self.NumColPnts/self.length))
        self.ColPnts = uniform.rvs(size=(self.NumColPnts, Sptldims+1), loc=self.lb, scale=self.ub-self.lb, random_state=None)
        self.TstColPnts = uniform.rvs(size=(self.Ntst, Sptldims+1), loc=self.lb, scale=self.ub-self.lb, random_state=None)
        print(f"The number of collocation points is {self.NumColPnts} and the ratio between collo. points to training points is {self.col_rat}:{1}")
    def __len__(self):
        return self.length
    def __getitem__(self,  idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self._to_float:
            return self.X_trn[idx, :].astype(np.float32), self.U_trn[idx, :].astype(np.float32), self.ColPnts[idx*self.col_rat:(idx+1)*self.col_rat, :].astype(np.float32)
        else:
            return self.X_trn[idx, :], self.U_trn[idx, :], self.ColPnts[idx*self.col_rat:(idx+1)*self.col_rat, :]  
    def Resample(self)->None:
        """
        Method that resamples the collocation points if the user would like to do so. 
        TODO:
            Should add a private class variable/parameter that checks is the points have already been
            resampled or not since with these QMC sampling methods, to maintain some balance property they 
            can only be resampled for the same number of points once. To resample again after that, we 
            would have to like do the sum of all the previous number of samples (the number of sampled 
            points needs to basically be a power of two).
        """
        self.ColPnts = uniform.rvs(size=(self.NumColPnts,self.Sptldims+1), loc=self.lb, scale=self.ub-self.lb, random_state=None)

def main():
    """ Just a simple demonstration of some of the functions here"""
    
    dataset = 'burgers1D'
    dataset = 'KdV'
    # dataset = 'allencahn2DEx4Nue0'
    # dataset = 'HeatEq3D0Drchlt'
    # 'DataSets/'+Dname+'.mat'
    dset = PDELearningMatData(fname='DataSets/'+ dataset +'.mat', Sptldims=1, split=0.80, smpleprcnt=0.05,
                             noisePrcntg=0.15, seed=None, to_float=True)

    # print(dset.length)
    # x, u, col = next(iter(loader))
    # print('loader 3D stuff')
    # print(x.size())
    # print(u)
    # print(u.size())
    # print(col.shape)
    # print(col)
    # print(col.view(-1,2))

if __name__ == '__main__':
    main()