from warnings import warn
import numpy as np
import torch
from typing import Union

# I do not doubt that one could create one single function that gives the the solutions and error
# instead of me either having 3 functions (one for each number of spatial dimensions) or one
# that has a massive if-else block for all the solutions. But hey the latter option may just be it.

def OneDimSols(dataset:str, lib:list, lrnd_sol:Union[torch.Tensor, np.ndarray]) -> Union[float, torch.Tensor, np.ndarray, None]:
    """
    Give the name of dataset for any of the one spatial and one temporal datasets used as examples in PDE learning, 
    the used candidate library and the vector/1D-tensor/numpy array that constitutes the learned equation, determine 
    the 
    TODO: Finish the description here and detailing the input arguments and the return arguments
    """
    # Some basic input argument checking
    if not isinstance(lrnd_sol, (torch.Tensor, np.ndarray)):
        print('ERROR - Wrong data type for the lrnd_sol argument. Need be a torch.Tensor or np.ndarray not {}'.format(type(lrnd_sol)))
        print('Returning a tuple of Nones')
        return (None, None, None)
    if not isinstance(dataset, str):
        print('ERROR - Wrong data type for the dataset argument. Need be aa string (str) not a {}'.format(type(lrnd_sol)))
        print('Returning a tuple of Nones')
        return (None, None, None)
    if not all(isinstance(term, str) for term in lib):
        print('ERROR - Wrong data type for an element in lib argument. Need be list of strings (str)')
        print('Returning a tuple of Nones')
        return (None, None, None)
    # Now know that things are of the correct type so will create empty tensor/arrays for the correct PDE equation with coefficients
    # and stuff
    if isinstance(lrnd_sol, (torch.Tensor)):
        EQ_sol = torch.zeros_like(lrnd_sol)
    else:
        EQ_sol = np.zeros_like(lrnd_sol)
    TruCoeffs = np.zeros((len(lib),) )
    # Now determine which data set we have and then the terms it should have and the corresponding coefficients on the terms
    # need this to create the correct solution vector
    if dataset=='advection1DRight':
        EqTerms = ['u_x0']    
        EqCoeffs = np.zeros((len(EqTerms),), )
        EqCoeffs[0] = -1.0
    elif dataset=='advection1DLeft':
        EqTerms = ['u_x0']    
        EqCoeffs = np.zeros((len(EqTerms),), )
        EqCoeffs[0] = 1.0
    elif dataset=='Allen_Cahn':
        EqTerms = ['u', 'u_x0x0', '(u)^3']
        EqCoeffs = np.zeros((len(EqTerms),), )
        EqCoeffs[0] = 1.0
        EqCoeffs[1] = 0.003
        EqCoeffs[2] = -1.0
    elif dataset=='allencahn1DEx1Nue0':
        EqTerms = ['u', 'u_x0x0', '(u)^3']
        EqCoeffs = np.zeros((len(EqTerms),), )
        EqCoeffs[0] = 1.0
        EqCoeffs[1] = 0.003
        EqCoeffs[2] = -1.0
    elif dataset=='allencahn1DEx2Nue0':
        EqTerms = ['u', 'u_x0x0', '(u)^3']
        EqCoeffs = np.zeros((len(EqTerms),), )
        EqCoeffs[0] = 1.0
        EqCoeffs[1] = 0.003
        EqCoeffs[2] = -1.0
    elif dataset=='Burgers_Exp':
        EqTerms = ['u_x0x0', '(u)(u_x0)']
        EqCoeffs = np.zeros((len(EqTerms),), )
        EqCoeffs[0] = 0.1
        EqCoeffs[1] = -1.0
    elif dataset=='burgers1D':
        EqTerms = ['u_x0x0', '(u)(u_x0)']
        EqCoeffs = np.zeros((len(EqTerms),), )
        EqCoeffs[0] = 0.1
        EqCoeffs[1] = -1.0
    elif dataset=='Burgers_Sine':
        EqTerms = ['u_x0x0', '(u)(u_x0)']
        EqCoeffs = np.zeros((len(EqTerms),), )
        EqCoeffs[0] = 0.1
        EqCoeffs[1] = -1.0
    elif dataset=='Heat_Sine_Exp*':
        EqTerms = ['u_x0x0']
        EqCoeffs = np.zeros((len(EqTerms),), )
        EqCoeffs[0] = 0.05
    elif dataset=='heat_sine*':
        EqTerms = ['u_x0x0']
        EqCoeffs = np.zeros((len(EqTerms),), )
        EqCoeffs[0] = 0.05
    elif dataset=='inviscidburgersDir0':
        EqTerms = ['(u)(u_x0)']
        EqCoeffs = np.zeros((len(EqTerms),), )
        EqCoeffs[0] = -1.0
    elif dataset=='kuramoto_sivishinky':
        EqTerms = ['u_x0x0', 'u_x0x0x0x0', '(u)(u_x0)']
        EqCoeffs = np.zeros((len(EqTerms),), )
        EqCoeffs[0] = -1.0
        EqCoeffs[1] = -1.0
        EqCoeffs[2] = -1.0
    elif dataset=='KdV_Sine':
        EqTerms = ['u_x0x0x0', '(u)(u_x0)']    
        EqCoeffs = np.zeros((len(EqTerms),), )
        EqCoeffs[0] = -1.0
        EqCoeffs[1] = -1.0
    elif dataset=='KG_Exp':
        EqTerms = ['u', 'u_x0x0']
        EqCoeffs = np.zeros((len(EqTerms),), )
        EqCoeffs[0] = -5.0
        EqCoeffs[1] = 0.5
    elif dataset=='Beam_Exp*':
        EqTerms = ['u_x0x0x0x0']
        EqCoeffs = np.zeros((len(EqTerms),), )
        EqCoeffs[0] = -0.1
    # enter the correct coefficient values in to the TruCoeffs array 
    else:
        print('Error - The given dataset was not found. Hence no solution Infor. can be found and thus no Error Info.')
        print('Will return a tuple of Nones')
        return (None, None, None)
    TruCoeffs = np.zeros((len(lib),) ) if isinstance(lrnd_sol, (np.ndarray)) else torch.zeros((len(lib),) )
    for i, term in enumerate(EqTerms):
        try:
            idx = lib.index(term)
        except(ValueError):
            msg = (f"For some reason, one of the function/terms in the {dataset} was not in the\n"
                   "Library based off the given derivative order and polynomial degree. Returning\n"
                   "inf values.")
            warn(message=msg, stacklevel=2)
            return (np.inf, np.inf*np.ones(len(EqTerms)), None)
        TruCoeffs[idx] = EqCoeffs[i]
    # Now compare the TruCoeffs array to the LrndCoeffs - They should have the exact same zeros (or nonzeros)
    # I am too lazy and not wanting to figure out how to make this more concise than using an if statement 
    # for the numpy func calls and the torch call. 
    if isinstance(lrnd_sol, (np.ndarray)): 
        LrndNonZeros = np.nonzero(lrnd_sol)[0]  if isinstance(lrnd_sol, (np.ndarray)) else lrnd_sol.nonzero(as_tuple=True)[0]
        TruNonZeros = np.nonzero(TruCoeffs)[0]  if isinstance(lrnd_sol, (np.ndarray)) else TruCoeffs.nonzero(as_tuple=True)[0]
        errs = np.empty_like(EqCoeffs) if isinstance(lrnd_sol, (np.ndarray)) else torch.empty((len(EqCoeffs),),)
        # Go through term of the correct equation and get the relative error
        # If the learned equation is missing one function/term in the actual
        # equation, then report a negative inf relative coeff error. 
        for i, idx in enumerate(TruNonZeros):
            if idx in LrndNonZeros:
                errs[i] = np.abs(TruCoeffs[idx] - lrnd_sol[idx])/np.abs(TruCoeffs[idx])
            else:
                errs[i] = -np.inf
        if LrndNonZeros.shape[0]< TruNonZeros.shape[0]:
            # learned equation is definitely missing a term so the reported
            # average relative error should be -inf. This is because with 
            # missing a term you are always under or less than and we
            # need feel negative inf properly reflects this. 
            err = -np.inf
        elif LrndNonZeros.shape[0] > TruNonZeros.shape[0]:
            # learned equation definitely has an extra a term so the reported 
            # average relative error should be inf. This is because with
            # having an extra term we are always over or greater than what
            # actually is the real value/equation
            err = np.inf
        else:
            # learned equation has the correct number of terms but 
            # are the chosen terms all the correct ones. If they 
            # all are the correct ones, then the just take the mean
            # of the errors. If we are missing one than negative inf. 
            err = errs.mean()
    else: 
        LrndNonZeros = lrnd_sol.nonzero(as_tuple=True)[0]
        TruNonZeros = TruCoeffs.nonzero(as_tuple=True)[0]
        errs = torch.empty((len(EqCoeffs),),)
        # Go through term of the correct equation and get the relative error
        # If the learned equation is missing one function/term in the actual
        # equation, then report a negative inf relative coeff error. 
        for i, idx in enumerate(TruNonZeros):
            if idx in LrndNonZeros:
                errs[i] = torch.abs(TruCoeffs[idx] - lrnd_sol[idx])/torch.abs(TruCoeffs[idx])
            else:
                errs[i] = -torch.inf
        if LrndNonZeros.shape[0]< TruNonZeros.shape[0]:
            # learned equation is definitely missing a term so the reported
            # average relative error should be -inf. This is because with 
            # missing a term you are always under or less than and we
            # need feel negative inf properly reflects this. 
            err = -torch.inf
        elif LrndNonZeros.shape[0] > TruNonZeros.shape[0]:
            # learned equation definitely has an extra a term so the reported 
            # average relative error should be inf. This is because with
            # having an extra term we are always over or greater than what
            # actually is the real value/equation
            err = torch.inf
        else:
            # learned equation has the correct number of terms but 
            # are the chosen terms all the correct ones. If they 
            # all are the correct ones, then the just take the mean
            # of the errors. If we are missing one than negative inf. 
            err = errs.mean()
    tru_RHS = ''
    plus = ' + '
    n = lrnd_sol.shape[0]
    for i in range(n):
        num = str(TruCoeffs[i].item())
        if TruCoeffs[i].item() == 0:
            continue
        tru_RHS = tru_RHS + num +'*'+lib[i]
        if i != n-1:
            tru_RHS = tru_RHS + plus
    if tru_RHS[len(tru_RHS)-2] == '+':
        tru_RHS = tru_RHS[0:len(tru_RHS)-2] # has an extra space at the end of the sting
        # lrnd_eq = lrnd_eq[len(lrnd_eq)-3] # does not have a space at the end of the string

    return (err, errs, tru_RHS)
    