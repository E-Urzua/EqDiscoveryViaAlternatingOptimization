import os
from typing import Tuple, List
import numpy as np
import pandas as pd
from scipy.special import binom

def Is_A_Num(text:str):
    try:
        float(text)
    except ValueError:
        return False
    return True

def numMonomials(numVar:int, deg:int)->int:
    """The nummber of basis monomials for a polynomail of numVar
    variables of degree deg"""
    if not isinstance(numVar, int):
        raise TypeError(f"The numVar input arg must of an integer type argument")
    if not isinstance(deg, int):
        raise TypeError(f"The deg input arg must of an integer type argument")
    if numVar<1:
        raise ValueError(f"numVar input arg must be greater than 0 (at least 1")
    if deg<0:
        raise ValueError(f"deg input arg must be at least 0")
    return int(binom(deg+numVar-1,numVar-1))

def numPolyMonomialsUpto(numVar:int, maxDeg:int, bias:bool=False)->int:
    """The nummber of basis monomials for a polynomial of numVar
    variables of maxDegree 0, 1,...maxDeg"""
    if not isinstance(numVar, int):
        raise TypeError(f"The numVar input arg must of an integer type argument")
    if not isinstance(maxDeg, int):
        raise TypeError(f"The maxDeg input arg must of an integer type argument")
    if numVar<1:
        raise ValueError(f"numVar input arg must be greater than 0 (at least 1")
    if maxDeg<0:
        raise ValueError(f"maxDeg input arg must be at least 0")
    return int(binom(maxDeg+numVar,numVar)) - int( not bias )
    
def EquationCoefsAndLib(EqName:str, noise:int)->Tuple[int, List[np.ndarray], List[List[str]], int]:
    """
    
    """

    # Now need to "populate" the correct entries in this array with the 
    # correct values based off of the Equation that is being learned.
    # Here come a bunch of if else statements...
    if EqName=='advection1DRight':
        EqTerms = [['u_x0']]    
        Eq1Coeffs = np.zeros((len(EqTerms[0]),), )
        Eq1Coeffs[0] = -1.0
        EqCoeffs = [Eq1Coeffs]
    elif EqName=='advection1DLeft':
        EqTerms = [['u_x0']]    
        Eq1Coeffs = np.zeros((len(EqTerms[0]),), )
        Eq1Coeffs[0] = 1.0
        EqCoeffs = [Eq1Coeffs]
    elif EqName=='Allen_Cahn':
        sptl_ord = 2
        polyDeg = 3
        EqTerms = [['u', 'u_x0x0', '(u)^3']]
        Eq1Coeffs = np.zeros((len(EqTerms[0]),), )
        Eq1Coeffs[0] = 1.0
        Eq1Coeffs[1] = 0.003
        Eq1Coeffs[2] = -1.0
        EqCoeffs = [Eq1Coeffs]
    elif EqName=='allencahn1DEx1Nue0':
        EqTerms = [['u', 'u_x0x0', '(u)^3']]
        Eq1Coeffs = np.zeros((len(EqTerms[0]),), )
        Eq1Coeffs[0] = 1.0
        Eq1Coeffs[1] = 0.003
        Eq1Coeffs[2] = -1.0
        EqCoeffs = [Eq1Coeffs]
    elif EqName=='allencahn1DEx2Nue0':
        EqTerms = [['u', 'u_x0x0', '(u)^3']]
        Eq1Coeffs = np.zeros((len(EqTerms[0]),), )
        Eq1Coeffs[0] = 1.0
        Eq1Coeffs[1] = 0.003
        Eq1Coeffs[2] = -1.0
        EqCoeffs = [Eq1Coeffs]
    elif EqName=='Burgers_Exp':
        sptl_ord = 2
        polyDeg = 2
        EqTerms = [['u_x0x0', '(u)(u_x0)']]
        Eq1Coeffs = np.zeros((len(EqTerms[0]),), )
        Eq1Coeffs[0] = 0.1
        Eq1Coeffs[1] = -1.0
        EqCoeffs = [Eq1Coeffs]
    elif EqName=='burgers1D':
        EqTerms = [['u_x0x0', '(u)(u_x0)']]
        Eq1Coeffs = np.zeros((len(EqTerms[0]),), )
        Eq1Coeffs[0] = 0.1
        Eq1Coeffs[1] = -1.0
        EqCoeffs = [Eq1Coeffs]
    elif EqName=='Burgers_Sine':
        sptl_ord = 2
        polyDeg = 2
        EqTerms = [['u_x0x0', '(u)(u_x0)']]
        Eq1Coeffs = np.zeros((len(EqTerms[0]),), )
        Eq1Coeffs[0] = 0.1
        Eq1Coeffs[1] = -1.0
        EqCoeffs = [Eq1Coeffs]
    elif EqName=='Heat_Sine_Exp*':
        sptl_ord = 2
        polyDeg = 2
        EqTerms = [['u_x0x0']]
        Eq1Coeffs = np.zeros((len(EqTerms[0]),), )
        Eq1Coeffs[0] = 0.05
        EqCoeffs = [Eq1Coeffs]
    elif EqName=='heat_sine*':
        sptl_ord = 2
        polyDeg = 2
        EqTerms = [['u_x0x0']]
        Eq1Coeffs = np.zeros((len(EqTerms[0]),), )
        Eq1Coeffs[0] = 0.05
        EqCoeffs = [Eq1Coeffs]
    elif EqName=='inviscidburgersDir0':
        EqTerms = [['(u)(u_x0)']]
        Eq1Coeffs = np.zeros((len(EqTerms[0]),), )
        Eq1Coeffs[0] = -1.0
        EqCoeffs = [Eq1Coeffs]
    elif EqName=='kuramoto_sivishinky':
        EqTerms = [['u_x0x0', 'u_x0x0x0x0', '(u)(u)_x0']]
        Eq1Coeffs = np.zeros((len(EqTerms[0]),), )
        Eq1Coeffs[0] = -1.0
        Eq1Coeffs[1] = -1.0
        Eq1Coeffs[2] = -1.0
        EqCoeffs = [Eq1Coeffs]
    elif EqName=='KdV_Sine':
        if noise==10 or noise==60:
            sptl_ord = 3
            polyDeg = 5
        else:
            sptl_ord = 3
            polyDeg = 3
        EqTerms = [['u_x0x0x0', '(u)(u_x0)']]    
        Eq1Coeffs = np.zeros((len(EqTerms[0]),), )
        Eq1Coeffs[0] = -1.0
        Eq1Coeffs[1] = -1.0
        EqCoeffs = [Eq1Coeffs]
    elif EqName=='KG_Exp':
        sptl_ord = 2
        polyDeg = 2
        EqTerms = [['u', 'u_x0x0']]
        Eq1Coeffs = np.zeros((len(EqTerms[0]),), )
        Eq1Coeffs[0] = -5.0
        Eq1Coeffs[1] = 0.5
        EqCoeffs = [Eq1Coeffs]
    elif EqName=='Beam_Exp*':
        sptl_ord = 4
        polyDeg = 2 
        EqTerms = [['u_x0x0x0x0']]
        Eq1Coeffs = np.zeros((len(EqTerms[0]),), )
        Eq1Coeffs[0] = -0.1
        EqCoeffs = [Eq1Coeffs]

    nLibFuncs = numPolyMonomialsUpto(numVar=sptl_ord+1, maxDeg=polyDeg, bias=False)
    n_eqs = len(EqTerms)
    return (n_eqs, EqCoeffs, EqTerms, nLibFuncs)

def Read_N_Extract_Results_File(fname:str)->Tuple:
    """
    This function reads through a results file that has been created and been apprended to 
    by the WriteResults function defined within the PDEComparisionScript.py file. For this
    reason this function will almost surely work in the one case. Things are hard coded 
    in this file to look at certain lines to extract data results. 
    """
    file = open(fname, 'r')
    lns = file.read()
    file.close()
    idx = lns.find(32*'~')
    lngth = len(lns[:idx].split("\n"))
    lines = lns.split('\n')
    idx = fname.rfind('_N')
    noise_idx = fname[idx:].rfind('_')
    noise = int(fname[idx+2:idx+noise_idx])
    if fname.rfind('%2A')>-1:
        idx = fname.rfind('%2A')
        EqName = fname[:idx]
        EqName+='*'
    else:
        idx = fname.rfind('_N')
        EqName = fname[:idx]
        
    n_eqs, EqCoefs, EqTerms, num_lib_funcs = EquationCoefsAndLib(EqName=EqName, noise=noise)
    for q in range(n_eqs):
        for i in range(len(EqTerms[q])):
            EqTerms[q][i] =  '*'+EqTerms[q][i]
    n_eq_coefs = np.array([coefs.shape[0] for coefs in EqCoefs], dtype=int)
    num_coefs = n_eq_coefs.sum()
    # lngth = 27 + num_lib_funcs + 2
    num_results = int((len(lines)-1)/lngth)
    print(f"The number of results in this test are {num_results}")
    # rng seeds are too big to be stored as an np.ndarray and so just write
    # them into a list and do something with the list later
    seeds = ['~~~~~~~~~~~~~~~~~~~~~']*num_results
    coef_stuf = np.empty((num_results, 3*num_coefs), )
    coef_stuf.fill(np.nan)
    for q in range(n_eqs):
        # coef_stuf[:, 3*np.arange(num_coefs)+2] = EqCoefs
        if q==0:
            coef_stuf[:, 3*np.arange(num_coefs)+2 + 0] = EqCoefs[q]
        else:
            coef_stuf[:, 3*np.arange(num_coefs)+2 + 3*n_eq_coefs[q-1]] = EqCoefs[q]
    col_names  = ['']*3*num_coefs
    # for i in range(num_coefs):
    #     col_names[3*i] = f" Coef. {i+1} Rel. Error"
    #     col_names[3*i +1] = f" Lrnd Coef. {i+1} Value"
    #     col_names[3*i +2] = f" True Coef. {i+1} Value"
    col_names.append('Run Time (secs)')
    mean_errs = np.empty((num_results,n_eqs), )
    mean_errs.fill(np.nan)
    runtimes = np.empty((num_results, 1), )
    runtimes.fill(np.nan)
    jobIDs = np.empty((num_results, 1), )
    jobIDs.fill(np.nan)
    ado_iters = np.empty((num_results, 1), )
    ado_iters.fill(np.nan)
    fvu_values = np.empty((num_results, n_eqs))
    fvu_values.fill(np.nan)
    kfolds = np.empty((num_results, 1), )
    kfolds.fill(np.nan)
    i = 0
    # Note that there is a way of getting the learned coef values
    # using the .find(')') on the line and then from there looking
    # to find the next ' + ' in the line. This can be used to get 
    # the correct terms in the PDE and then once we have these we 
    # could look if these equations are int he 
    strt_coef_err_lines = 13 + 2*n_eqs + 2
    strt_eqs_mean_coef_errs = strt_coef_err_lines + n_eqs + 1
    fvu_line = strt_eqs_mean_coef_errs + 3 + num_lib_funcs + 2
    while i<=len(lines)-2:
        line = lines[i]
        k = i//lngth
        if i%lngth==0: # The first line - contains the job id number
            words = line.split(' ')
            try:
                jobIDs[k] = int(words[-8][:-1])
            except ValueError:
                print(words[-9:])
                print(fname)
                raise ValueError('invalid literal for int() with base 10:')
        elif i%lngth==3: # get the run time in seconds. 
            words = line.split(' ')
            runtimes[k] = float(words[-5])
        elif i%lngth==6: # the rng data selection number/seed
            words = line.split(' ')
            seeds[k] = words[-1]
        elif i%lngth==9: # get the number of ado iterations
            words = line.split(' ')
            ado_iters[k] = int(words[6][:-1])
        
        # if i%lngth==12: # get the number of collocation partitions
        #     words = line.split(' ')
        #     kfolds[k] = int(words[-1])

        elif i%lngth in range(13, 13+n_eqs): # Learned equation line(s) - Need to figure out what to do with case of systems of equations.
            # So now know which are the correct terms look at the learned equation
            # and see if these terms are in the equation. If not, given an error
            q = 13 - i%lngth
            col_strt = q if q==0 else 3*n_eq_coefs[q-1]
            for j, term in enumerate(EqTerms[q]):
                stp = line.find(term)
                if stp==-1:
                    # so the learned equation was not correct - missing a correct term
                    coef_stuf[k, 3*j+1] = 0.0 # The coefficent value is 0.0 but this doesn't feel right
                    # coef_stuf[k, 3*j+1] = np.inf
                else:
                    # something special needs to be done for the first term in the EQ since that is no ' + ' in front of it. 
                    spcl = line.find('*')
                    if spcl==stp:
                        beg = line[:stp].rfind('= ')
                    else:
                        beg = line[:stp].rfind(' + ') # go ahead by 2 indices to get the start of the coef value
                    # since in the library the correct term can be in a product with another term and thus need to check if is a number
                    coef_stuf[k, 3*j+1] = float(line[beg+2:stp]) if Is_A_Num(line[beg+2:stp]) else 0.0 
                    # coef_stuf[k, 3*j+1] = float(line[beg+2:stp]) if Is_A_Num(line[beg+2:stp]) else np.inf
                # Since we have the terms of the pde use them in the column headers/names
                col_names[3*j + col_strt] = term +" Rel. Error"
                col_names[3*j +1 + col_strt] = term + " Lrnd Coef. Value"
                col_names[3*j +2 + col_strt] = term + " True Coef. Value" 
        elif i%lngth in range(strt_coef_err_lines, strt_coef_err_lines+n_eqs):
            # individual coefficient errors. - FUCK its a tensor not a np.array. Damnt 
            # errs = line.replace(']','').replace('[','').lstrip().rstrip().split() # ONLY IF the result was an np.ndarray
            q = i%lngth - strt_coef_err_lines # q defines if the lines is the error for the first EQ (q=0), second EQ (q=1), etc
            strt = line.find('[') + 1
            stp = line.rfind(']') 
            errs = line[strt:stp].replace(',','').split()
            col_strt = q if q==0 else 3*n_eq_coefs[q-1]
            for j, val in enumerate(errs):
                coef_stuf[k, 3*j+col_strt] = float(val)
        elif i%lngth in range(strt_eqs_mean_coef_errs, strt_eqs_mean_coef_errs+n_eqs):
            # the mean coefficent error
            q = i%lngth - strt_eqs_mean_coef_errs
            mean_errs[k, q] = float(line)
        elif i%lngth==fvu_line:
            words = line.split(' ')
            strt = -1*(2+n_eqs)
            for q in range(n_eqs):
                fvu_values[k,q] = float(words[strt+q])
        i+=1

    scr_metrics = np.empty(shape=(num_results, 2*n_eqs), )
    scr_metrics[:, 2*np.arange(0,n_eqs)] = fvu_values
    scr_metrics[:, 2*np.arange(0,n_eqs)+1] = mean_errs
    data_out = np.concatenate((scr_metrics, coef_stuf, runtimes), axis=1)
    # alright take the averages and stds of the results and things
    # but only use the results for which the correct eq was learned
    eq_lrn_fracs = np.empty((1,n_eqs),)
    aves, stds = np.empty(shape=(1, data_out.shape[1]), ), np.empty(shape=(1, data_out.shape[1]), )
    aves[0,-1] = data_out[:,-1].mean()
    stds[0,-1] = data_out[:,-1].std()
    for q in range(n_eqs):
        col_names.insert(0, f"EQ. {n_eqs - q } Ave. Coef. Rel. Err.")
        col_names.insert(0, f"EQ. {n_eqs - q } FVU Val. ")
        if q==0:
            col_ids = list(range(2*q, 2*q+2)) + list(range(2*n_eqs, 2*n_eqs+3*n_eq_coefs[q]))
        else:
            col_ids = col_ids = list(range(2*q, 2*q+2)) + list(range(2*n_eqs, 2*n_eqs+3*n_eq_coefs[:q]))
        nzs = np.nonzero(np.isfinite(scr_metrics[:, 2*q + 1]))[0][:, np.newaxis]
        eq_lrn_fracs[0, q] = nzs.shape[0] / num_results
        aves[0,col_ids] = np.mean(data_out[nzs, col_ids], axis=0, keepdims=False)
        stds[0,col_ids] = np.std(data_out[nzs, col_ids], axis=0, keepdims=False)
    ave_col_names = ["Mean "  + name for name in col_names]
    std_col_names = ["STD "  + name for name in col_names]
    
    dframe = pd.DataFrame(data=data_out, index=[f"Run {i}" for i in range(1, num_results+1)], columns=col_names)
    dframe.insert(loc=data_out.shape[1], column='Data Seed', value=seeds)
    dframe.insert(loc=data_out.shape[1], column='Slurm Job ID', value=jobIDs)
    dframe.insert(loc=data_out.shape[1], column='Num. Ado Iters', value=ado_iters)
    ave_frame = pd.DataFrame(data=aves, columns=ave_col_names)
    std_frame = pd.DataFrame(data=stds, columns=std_col_names)
    lrn_frame = pd.DataFrame(data=eq_lrn_fracs, columns=[f"EQ {q+1} Crct Lrnd Fract." for q in range(n_eqs)])
    # dframe.insert(loc=data_out.shape[1], column='Kfold Value', value=kfolds)
    # dframe.reindex([f"Run {i}" for i in range(1, num_results+1)])
    # idx = fname.rfind('Results.txt')
    idx = fname.rfind('DefaultVer2SettingsResults.txt')
    # sheet_name = fname[:idx].replace('%2A', '*') # Windows and their apps don't like '*' as part of a name in things egh?
    sheet_name = fname[:idx]
    
    return (sheet_name, dframe, ave_frame, std_frame, lrn_frame)

def Read_N_Extract_Results_File_V2(fname:str)->Tuple:
    """
    This function reads through a results file that has been created and been apprended to 
    by the WriteResults function defined within the PDEComparisionScript.py file. For this
    reason this function will almost surely work in the one case. Things are hard coded 
    in this file to look at certain lines to extract data results. 
    """
    file = open(fname, 'r')
    lns = file.read()
    file.close()
    idx = lns.find(32*'~')
    lngth = len(lns[:idx].split("\n"))
    lines = lns.split('\n')
    idx = fname.rfind('_N')
    noise_idx = fname[idx:].rfind('_')
    noise = int(fname[idx+2:idx+noise_idx])
    if fname.rfind('%2A')>-1:
        idx = fname.rfind('%2A')
        EqName = fname[:idx]
        EqName+='*'
    else:
        idx = fname.rfind('_N')
        EqName = fname[:idx]
        
    n_eqs, EqCoefs, EqTerms, num_lib_funcs = EquationCoefsAndLib(EqName=EqName, noise=noise)
    for i in range(len(EqTerms)):
        EqTerms[i] =  '*'+EqTerms[i]
    num_coefs = EqCoefs.shape[0]
    # lngth = 27 + num_lib_funcs + 2
    num_results = int((len(lines)-1)/num_lib_funcs)

    # rng seeds are too big to be stored as an np.ndarray and so just write
    # them into a list and do something with the list later
    seeds = ['~~~~~~~~~~~~~~~~~~~~~']*num_results
    coef_stuf = np.empty((num_results, 3*num_coefs), )
    coef_stuf.fill(np.nan)
    coef_stuf[:, 3*np.arange(num_coefs)+2] = EqCoefs
    col_names  = ['']*3*num_coefs
    # for i in range(num_coefs):
    #     col_names[3*i] = f" Coef. {i+1} Rel. Error"
    #     col_names[3*i +1] = f" Lrnd Coef. {i+1} Value"
    #     col_names[3*i +2] = f" True Coef. {i+1} Value"
    col_names.append('Run Time (secs)')
    mean_errs = np.empty((num_results,n_eqs), )
    mean_errs.fill(np.nan)
    runtimes = np.empty((num_results, 1), )
    runtimes.fill(np.nan)
    jobIDs = np.empty((num_results, 1), )
    jobIDs.fill(np.nan)
    ado_iters = np.empty((num_results, 1), )
    ado_iters.fill(np.nan)
    fvu_values = np.empty((num_results, n_eqs))
    fvu_values.fill(np.nan)
    kfolds = np.empty((num_results, 1), )
    kfolds.fill(np.nan)
    i = 0
    # Note that there is a way of getting the learned coef values
    # using the .find(')') on the line and then from there looking
    # to find the next ' + ' in the line. This can be used to get 
    # the correct terms in the PDE and then once we have these we 
    # could look if these equations are int he 
    strt_coef_err_lines = 13 + 2*n_eqs + 2
    strt_eqs_mean_coef_errs = strt_coef_err_lines + n_eqs + 1
    fvu_line = strt_eqs_mean_coef_errs + 3 + num_lib_funcs + 2
    while i<=len(lines)-2:
        line = lines[i]
        k = i//lngth
        if i%lngth==0: # The first line - contains the job id number
            words = line.split(' ')
            try:
                jobIDs[k] = int(words[-8][:-1])
            except ValueError:
                print(words[-9:])
                print(fname)
                raise ValueError('invalid literal for int() with base 10:')
        elif i%lngth==3: # get the run time in seconds. 
            words = line.split(' ')
            runtimes[k] = float(words[-5])
        elif i%lngth==6: # the rng data selection number/seed
            words = line.split(' ')
            seeds[k] = words[-1]
        elif i%lngth==9: # get the number of ado iterations
            words = line.split(' ')
            ado_iters[k] = int(words[6][:-1])
        # if i%lngth==12: # get the number of collocation partitions
        #     words = line.split(' ')
        #     kfolds[k] = int(words[-1])

        elif i%lngth==13: # Learned equation line(s).
            # So now know which are the correct terms look at the learned equation
            # and see if these terms are in the equation. If not, given an error
            for j, term in enumerate(EqTerms):
                stp = line.find(term)
                if stp==-1:
                    # so the learned equation was not correct - missing a correct term
                    coef_stuf[k, 3*j+1] = 0.0 # The coefficent value is 0.0 but this doesn't feel right
                    # coef_stuf[k, 3*j+1] = np.inf
                else:
                    # something special needs to be done for the first term in the EQ since that is no ' + ' in front of it. 
                    spcl = line.find('*')
                    if spcl==stp:
                        beg = line[:stp].rfind('= ')
                    else:
                        beg = line[:stp].rfind(' + ') # go ahead by 2 indices to get the start of the coef value
                    # since in the library the correct term can be in a product with another term and thus need to check if is a number
                    coef_stuf[k, 3*j+1] = float(line[beg+2:stp]) if Is_A_Num(line[beg+2:stp]) else 0.0 
                    # coef_stuf[k, 3*j+1] = float(line[beg+2:stp]) if Is_A_Num(line[beg+2:stp]) else np.inf
                # Since we have the terms of the pde use them in the column headers/names
                col_names[3*j] = term +" Rel. Error"
                col_names[3*j +1] = term + " Lrnd Coef. Value"
                col_names[3*j +2] = term + " True Coef. Value" 
        elif i%lngth==strt_coef_err_lines:
            # individual coefficient errors. - FUCK its a tensor not a np.array. Damnt 
            # errs = line.replace(']','').replace('[','').lstrip().rstrip().split() # ONLY IF the result was an np.ndarray
            strt = line.find('[') + 1
            stp = line.rfind(']') 
            errs = line[strt:stp].replace(',','').split()
            for j, val in enumerate(errs):
                coef_stuf[k, 3*j] = float(val)
        elif i%lngth==strt_eqs_mean_coef_errs:
            # the mean coefficent error
            for q in range(n_eqs):
                mean_errs[k, q] = float(line)
        elif i%lngth==fvu_line:
            words = line.split(' ')
            strt = -1*(2+n_eqs)
            for q in range(n_eqs):
                fvu_values[k,q] = float(words[strt+q])
        i+=1
    col_names.insert(0, 'Ave. Coef. Rel. Err.')
    data_out = np.concatenate((mean_errs, coef_stuf, runtimes), axis=1)
    dframe = pd.DataFrame(data=data_out, index=[f"Run {i}" for i in range(1, num_results+1)], columns=col_names)
    dframe.insert(loc=data_out.shape[1], column='Data Seed', value=seeds)
    dframe.insert(loc=data_out.shape[1], column='Slurm Job ID', value=jobIDs)
    dframe.insert(loc=data_out.shape[1], column='Num. Ado Iters', value=ado_iters)
    dframe.insert(loc=data_out.shape[1], column='Kfold Value', value=kfolds)
    # dframe.reindex([f"Run {i}" for i in range(1, num_results+1)])
    # idx = fname.rfind('Results.txt')
    idx = fname.rfind('DefaultVer2SettingsResults.txt')
    # sheet_name = fname[:idx].replace('%2A', '*') # Windows and their apps don't like '*' as part of a name in things egh?
    sheet_name = fname[:idx]
    
    return (dframe, sheet_name)

def Read_N_Extract_Results_File_V3(fname:str)->Tuple:
    """
    This function reads through a results file that has been created and been apprended to 
    by the WriteResults function defined within the PDEComparisionScript.py file. For this
    reason this function will almost surely work in the one case. Things are hard coded 
    in this file to look at certain lines to extract data results. 
    """
    file = open(fname, 'r')
    lns = file.read()
    file.close()

    indvd_sim_txt_res = lns.split('~'*150+'\n')[:-1]
    num_results = len(indvd_sim_txt_res)

    # lines = lns.split('\n')

    idx = fname.rfind('_N')
    noise_idx = fname[idx:].rfind('_')
    noise = int(fname[idx+2:idx+noise_idx])
    if fname.rfind('%2A')>-1:
        idx = fname.rfind('%2A')
        EqName = fname[:idx]
        EqName+='*'
    else:
        idx = fname.rfind('_N')
        EqName = fname[:idx]

    n_eqs, EqCoefs, EqTerms, num_lib_funcs = EquationCoefsAndLib(EqName=EqName, noise=noise)
    for q in range(n_eqs):
        for i in range(len(EqTerms[q])):
            EqTerms[q][i] =  '*'+EqTerms[q][i]
    n_eq_coefs = np.array([coefs.shape[0] for coefs in EqCoefs], dtype=int)
    num_coefs = n_eq_coefs.sum()
    # rng seeds are too big to be stored as an np.ndarray and so just write
    # them into a list and do something with the list later
    seeds = ['~~~~~~~~~~~~~~~~~~~~~']*num_results
    coef_stuf = np.empty((num_results, 3*num_coefs), )
    coef_stuf.fill(np.nan)
    for q in range(n_eqs):
        # coef_stuf[:, 3*np.arange(num_coefs)+2] = EqCoefs
        if q==0:
            coef_stuf[:, 3*np.arange(num_coefs)+2 + 0] = EqCoefs[q]
        else:
            coef_stuf[:, 3*np.arange(num_coefs)+2 + 3*n_eq_coefs[q-1]] = EqCoefs[q]
    col_names  = ['']*3*num_coefs
    # for i in range(num_coefs):
    #     col_names[3*i] = f" Coef. {i+1} Rel. Error"
    #     col_names[3*i +1] = f" Lrnd Coef. {i+1} Value"
    #     col_names[3*i +2] = f" True Coef. {i+1} Value"
    col_names.append('Run Time (secs)')
    mean_errs = np.empty((num_results,n_eqs), )
    mean_errs.fill(np.nan)
    runtimes = np.empty((num_results, 1), )
    runtimes.fill(np.nan)
    jobIDs = np.empty((num_results, 1), )
    jobIDs.fill(np.nan)
    ado_iters = np.empty((num_results, 1), )
    ado_iters.fill(np.nan)
    fvu_values = np.empty((num_results, n_eqs))
    fvu_values.fill(np.nan)
    kfolds = np.empty((num_results, 1), )
    kfolds.fill(np.nan)

    i = 0
    # Note that there is a way of getting the learned coef values
    # using the .find(')') on the line and then from there looking
    # to find the next ' + ' in the line. This can be used to get 
    # the correct terms in the PDE and then once we have these we 
    # could look if these equations are int he 
    strt_coef_err_lines = 13 + 2*n_eqs + 2
    strt_eqs_mean_coef_errs = strt_coef_err_lines + n_eqs + 1
    fvu_line = strt_eqs_mean_coef_errs + 3 + num_lib_funcs + 2

    for k in range(num_results):
        sim_res_lines = indvd_sim_txt_res[k].split('\n')
        for i in range(fvu_line+1):
            line = sim_res_lines[i]
            if i==0: # The first line - contains the job id number
                words = line.split(' ')
                try:
                    jobIDs[k] = int(words[-8][:-1])
                except ValueError:
                    print(words[-9:])
                    print(fname)
                    raise ValueError('invalid literal for int() with base 10:')
            elif i==3: # get the run time in seconds. 
                words = line.split(' ')
                runtimes[k] = float(words[-5])
            elif i==6: # the rng data selection number/seed
                words = line.split(' ')
                seeds[k] = words[-1]
            elif i==9: # get the number of ado iterations
                words = line.split(' ')
                ado_iters[k] = int(words[6][:-1])

            # if i==12: # get the number of collocation partitions
            #     words = line.split(' ')
            #     kfolds[k] = int(words[-1])

            elif i in range(13, 13+n_eqs): # Learned equation line(s) - Need to figure out what to do with case of systems of equations.
                # So now know which are the correct terms look at the learned equation
                # and see if these terms are in the equation. If not, given an error
                q = 13 - i
                col_strt = q if q==0 else 3*n_eq_coefs[q-1]
                for j, term in enumerate(EqTerms[q]):
                    stp = line.find(term)
                    if stp==-1:
                        # so the learned equation was not correct - missing a correct term
                        coef_stuf[k, 3*j+1] = 0.0 # The coefficent value is 0.0 but this doesn't feel right
                        # coef_stuf[k, 3*j+1] = np.inf
                    else:
                        # something special needs to be done for the first term in the EQ since that is no ' + ' in front of it. 
                        spcl = line.find('*')
                        if spcl==stp:
                            beg = line[:stp].rfind('= ')
                        else:
                            beg = line[:stp].rfind(' + ') # go ahead by 2 indices to get the start of the coef value
                        # since in the library the correct term can be in a product with another term and thus need to check if is a number
                        coef_stuf[k, 3*j+1] = float(line[beg+2:stp]) if Is_A_Num(line[beg+2:stp]) else 0.0 
                        # coef_stuf[k, 3*j+1] = float(line[beg+2:stp]) if Is_A_Num(line[beg+2:stp]) else np.inf
                    # Since we have the terms of the pde use them in the column headers/names
                    col_names[3*j + col_strt] = term +" Rel. Error"
                    col_names[3*j +1 + col_strt] = term + " Lrnd Coef. Value"
                    col_names[3*j +2 + col_strt] = term + " True Coef. Value" 
            elif i in range(strt_coef_err_lines, strt_coef_err_lines+n_eqs):
                # individual coefficient errors. - FUCK its a tensor not a np.array. Damnt 
                # errs = line.replace(']','').replace('[','').lstrip().rstrip().split() # ONLY IF the result was an np.ndarray
                q = i - strt_coef_err_lines # q defines if the lines is the error for the first EQ (q=0), second EQ (q=1), etc
                strt = line.find('[') + 1
                stp = line.rfind(']') 
                errs = line[strt:stp].replace(',','').split()
                col_strt = q if q==0 else 3*n_eq_coefs[q-1]
                for j, val in enumerate(errs):
                    coef_stuf[k, 3*j+col_strt] = float(val)
            elif i in range(strt_eqs_mean_coef_errs, strt_eqs_mean_coef_errs+n_eqs):
                # the mean coefficent error
                q = i - strt_eqs_mean_coef_errs
                mean_errs[k, q] = float(line)
            elif i==fvu_line:
                words = line.split(' ')
                strt = -1*(2+n_eqs)
                for q in range(n_eqs):
                    fvu_values[k,q] = float(words[strt+q])
    
    scr_metrics = np.empty(shape=(num_results, 2*n_eqs), )
    scr_metrics[:, 2*np.arange(0,n_eqs)] = fvu_values
    scr_metrics[:, 2*np.arange(0,n_eqs)+1] = mean_errs
    data_out = np.concatenate((scr_metrics, coef_stuf, runtimes), axis=1)
    # alright take the averages and stds of the results and things
    # but only use the results for which the correct eq was learned
    eq_lrn_fracs = np.empty((1,n_eqs),)
    aves, stds = np.empty(shape=(1, data_out.shape[1]), ), np.empty(shape=(1, data_out.shape[1]), )
    aves[0,-1] = data_out[:,-1].mean()
    stds[0,-1] = data_out[:,-1].std()
    for q in range(n_eqs):
        col_names.insert(0, f"EQ. {n_eqs - q } Ave. Coef. Rel. Err.")
        col_names.insert(0, f"EQ. {n_eqs - q } FVU Val. ")
        if q==0:
            col_ids = list(range(2*q, 2*q+2)) + list(range(2*n_eqs, 2*n_eqs+3*n_eq_coefs[q]))
        else:
            col_ids = col_ids = list(range(2*q, 2*q+2)) + list(range(2*n_eqs, 2*n_eqs+3*n_eq_coefs[:q]))
        nzs = np.nonzero(np.isfinite(scr_metrics[:, 2*q + 1]))[0][:, np.newaxis]
        eq_lrn_fracs[0, q] = nzs.shape[0] / num_results
        aves[0,col_ids] = np.mean(data_out[nzs, col_ids], axis=0, keepdims=False)
        stds[0,col_ids] = np.std(data_out[nzs, col_ids], axis=0, keepdims=False)
    ave_col_names = ["Mean "  + name for name in col_names]
    std_col_names = ["STD "  + name for name in col_names]

    dframe = pd.DataFrame(data=data_out, index=[f"Run {i}" for i in range(1, num_results+1)], columns=col_names)
    dframe.insert(loc=data_out.shape[1], column='Data Seed', value=seeds)
    dframe.insert(loc=data_out.shape[1], column='Slurm Job ID', value=jobIDs)
    dframe.insert(loc=data_out.shape[1], column='Num. Ado Iters', value=ado_iters)
    ave_frame = pd.DataFrame(data=aves, columns=ave_col_names)
    std_frame = pd.DataFrame(data=stds, columns=std_col_names)
    lrn_frame = pd.DataFrame(data=eq_lrn_fracs, columns=[f"EQ {q+1} Crct Lrnd Fract." for q in range(n_eqs)])
    # dframe.insert(loc=data_out.shape[1], column='Kfold Value', value=kfolds)
    # dframe.reindex([f"Run {i}" for i in range(1, num_results+1)])
    idx = fname.rfind('Results.txt')
    if fname.rfind("OptimalResults.txt")>-1:
        idx = fname.rfind("OptimalResults.txt")
    # sheet_name = fname[:idx].replace('%2A', '*') # Windows and their apps don't like '*' as part of a name in things egh?
    sheet_name = fname[:idx]
    
    return (sheet_name, dframe, ave_frame, std_frame, lrn_frame)

def Excel_Results_File(res_types:str=None)->None:
    """
    Descriptive Text Goes here. 
    SHOULD ONLY BY USED IN TO GET THE RESULTS AND MORE RUN FROM THE 
    CMD-LINE WHILE IN THE FOLDER/DIRECTORY WITH THE ...Results.txt
    FILES. 
    """
    dir_files = os.listdir('.') 
    if res_types=="Optimal":
        ResFiles = [f for f in dir_files if f.rfind('.txt')>1 and f.rfind("Optimal")>-1]
        xcl_name = 'PinnsSrPlusCompiledOptimalResults.xlsx'
    else:
        ResFiles = [f for f in dir_files if f.rfind('.txt')>1 and "Optimal" not in f]
        xcl_name = 'PinnsSrPlusCompiledResults.xlsx'
    
    dframes = {}
    ave_frames = []
    std_frames = []
    lrn_frac = []
    n_rslts = []
    for file in ResFiles:
        print("Putting Results in a pandas data frame for "+file)
        # temp = Read_N_Extract_Results_File(fname=file)
        temp = Read_N_Extract_Results_File_V3(fname=file)
        dframes[temp[0]] = temp[1]
        ave_frames.append(temp[2])
        std_frames.append(temp[3])
        lrn_frac.append(temp[4])
        n_rslts.append(temp[1].shape[0])
    
    # # with pd.ExcelWriter('PdeReadCompiledResults.xlsx', mode='a', if_sheet_exists='new') as writer:
    # with pd.ExcelWriter('PinnsSrPlusCompiledResults.xlsx', mode='w',) as writer:
    #     for key in dframes.keys():
    #         dframes[key].to_excel(writer, sheet_name=key, float_format="%.8f", header=True,
    #                               startrow=1, startcol=1, )
    
    with pd.ExcelWriter(xcl_name, mode='w',) as writer:
    # with pd.ExcelWriter('PdeReadCompiledResults.xlsx', mode='a', if_sheet_exists='overlay') as writer:
        for i, key in enumerate(dframes.keys()):
            dframes[key].to_excel(writer, sheet_name=key, float_format="%.8f", header=True,
                                  startrow=1, startcol=1, )
            ave_frames[i].to_excel(writer, sheet_name=key, float_format="%.8f", header=True, index=False,
                                  startrow=1+n_rslts[i]+3, startcol=2, )
            std_frames[i].to_excel(writer, sheet_name=key, float_format="%.8f", header=True, index=False,
                                  startrow=1+n_rslts[i]+5, startcol=2, )
            lrn_frac[i].to_excel(writer, sheet_name=key, float_format="%.8f", header=True, index=False,
                                  startrow=1+n_rslts[i]+9, startcol=2, )
    # return dframes
    return None

if __name__=='__main__':
    print("Doing Results_File Function")
    Excel_Results_File()
