import os
from copy import deepcopy
from typing import List, Union, Tuple
import warnings
import datetime

import numpy as np
from numpy.random import SeedSequence
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm, Normalize#, ListedColormap, BoundaryNorm

import torch

from PartialDerivFunctions import Nth_temporal_prtls
from trackers import LossTracker
from extra_loss_functions import Torch_Lp_Loss
from func_libraries import BaseFuncLib, Poly_Deriv_Library
from sparse_regress_algs import SparseRegressAlg, RFE, Cross_Val_RFE, Cross_Val_RFE_V2, SSR
from data_sampling import Rand_Col_Sampler
from torch.utils.data import DataLoader, TensorDataset

def solve_svd(matrix:torch.Tensor, trgs:torch.Tensor, alphas:Union[float, torch.Tensor]=0.0)->torch.Tensor:
    """
    Torch function to finds/determines the solution to the ridge
    regression problem of 

        argmin(x) ||Ax - b||_{2}^{2} + alpha*||x||_{2}^{2}

    using the singular value decomposition of the matrix A. Note that
    this method can be used even if b (the targets) is a matrix itself.
    Note that if all of the alphas are zero, then should just use the
    normal least square functino/method. 

    Input arguments:

        * matrix (2D torch Tensor) - The M x N matrix A seen in the above 
                equation.

        * trgs (2D torch Tensor) - The M x P vector/matrix b in the above
                equation where P>=1 is the number of targets. 
        
        * alphas (float or 1D tensor of floats) - The scalar alpha found 
                in the above equation however if there are multiples 
                targets  (i.e P>1 or that b is a matrix in the above eq.)
                then it is a 1D P sized tensor of non-negative float. 

    Return type: 2D tensor minimizing X in the above equation.
        
        * res - X that is the argmin(x) in the above equation. Is a 2D 
                N x P tensor. 

    """
    dvc = matrix.device
    if isinstance(alphas, float):
       alphas = torch.full(size=(trgs.shape[1],), fill_value=alphas, device=dvc) 
    if isinstance(alphas, torch.Tensor):
        if alphas.ndim==0:
            alphas = torch.full(size=(trgs.shape[1],), fill_value=alphas, device=dvc) 
    if torch.all(alphas<0):
        raise ValueError(f"alpha needs to be non-negative!")
    zros_ids = torch.argwhere(alphas==0).flatten()
    non_zros_ids = [j for j in range(trgs.shape[1]) if j not in zros_ids]
    res = torch.zeros(size=(matrix.shape[1], trgs.shape[1]), device=dvc, dtype=matrix.dtype)
    tmp_res = torch.linalg.lstsq(matrix, trgs[:, zros_ids])[0]
    if tmp_res.nelement()!=0:
        res[:, zros_ids] = tmp_res
    if len(non_zros_ids)==0:
        return res
    U, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    ids = s>1e-15
    s_nnz = s[ids][:, None]
    UTy = U.T @ trgs[:, non_zros_ids]
    d = torch.zeros((s.shape[0], len(non_zros_ids)), dtype=matrix.dtype,device=dvc)
    d[ids] = s_nnz / (s_nnz**2 + alphas[non_zros_ids])
    d_UT_y = d * UTy
    res[:, non_zros_ids] = vh.T @ d_UT_y
    return res

def model_regress(mat:torch.Tensor, trgt:torch.Tensor,apha):
    l = mat.shape[1]
    xs = torch.zeros(size=(l,l), device=mat.device, dtype=mat.dtype)
    for j in range(l):
        updts = [k for k in range(l) if k!=j]
        xs[updts, j:j+1] = solve_svd(matrix=mat[:,updts], trgs=trgt, alphas=apha)
    # get the least important feature - removed on the next pass
    lst = torch.argmin(torch.linalg.vector_norm(mat@xs - trgt, ord=2, axis=0, keepdim=True).pow(2) + apha*torch.linalg.vector_norm(xs, ord=2, axis=0, keepdim=True).pow(2), dim=1)
    rs = solve_svd(matrix=mat, trgs=trgt, alphas=apha)
    return rs, lst

def Specified_Optim_Regres(A:torch.Tensor, b:torch.Tensor, activ_ids:Union[list[int],np.ndarray], alpha)->torch.Tensor:
    """
    
    """
    mags = torch.linalg.vector_norm(A, ord=2, dim=0)
    # mags = torch.ones_like(mags)
    x = A / mags
    if b.ndim==1:
        y = torch.clone(b)[:, np.newaxis]
    else:
        y = torch.clone(b)
    n = A.shape[1]
    if isinstance(activ_ids, list):
        ids = deepcopy(activ_ids)
    elif isinstance(activ_ids, np.ndarray):
        ids = activ_ids.tolist()
    else:
        raise TypeError(f"active_ids array needs to be a list or numpy.ndarray")
    max_iters = len(ids)

    # n_trgts = y.shape[-1]

    # There is probably a quicker way of doing x[trn_splts[q]][:,ids[j]] like using np.ix_

    cs = torch.zeros( size=(n, max_iters+1), device=A.device, dtype=A.dtype)
    # Rls = torch.empty( size=(max_iters+1, n_trgts), device=A.device, dtype=A.dtype)

    for i in range(max_iters):
            cp, idx = model_regress(x[:,ids], y[:, 0:0+1], apha=alpha)
            cs[ids, i:i+1] = cp
            ids.pop(idx)
    # Rls = (x @ cs - y).pow(2).sum(dim=1)
    
    
    return cs / mags[:, None]

def model_loss_plotter(type_loss:str,
                       losses_trn:np.ndarray, losses_tst:np.ndarray, 
                       optim_trn_epochs:np.ndarray, plt_lbls:List[str], 
                       plt_tle:str, fnt_sz:float):
    """
    
    """
    if type_loss.lower()=="lp":
        plot_func = plt.plot
    else:
        plot_func = plt.semilogy    
    nzs_epochs = np.nonzero(optim_trn_epochs)[0]
    plt_fig = plt.figure(figsize=(20, 10), layout='constrained')
    for k in range(len(plt_lbls)):

        indices = range(optim_trn_epochs[:k].sum(), optim_trn_epochs[:k+1].sum())
        sctr_inds = range(optim_trn_epochs[:k].sum(), optim_trn_epochs[:k].sum()+1)
        if len(indices)==0:
            continue

        if k==nzs_epochs[-1]:
            # plt.semilogy(indices, losses_tst[indices, 0], linestyle='solid', linewidth=1.5, color='black', label='Test Data Losses')
            if losses_tst is not None:
                plot_func(indices, losses_tst[indices, 0], linestyle='solid', linewidth=1.5, color='black', label='Test Data Losses')
        else:
            # plt.semilogy(indices, losses_tst[indices, 0], linestyle='solid', linewidth=1.5, color='black',)
            if losses_tst is not None:
                plot_func(indices, losses_tst[indices, 0], linestyle='solid', linewidth=1.5, color='black',)
        # l1 = plt.semilogy(indices, losses_trn[indices, 0], linestyle='solid', linewidth=1.5, label=plt_lbls[k])
        l1 = plot_func(indices, losses_trn[indices, 0], linestyle='solid', linewidth=1.5, label=plt_lbls[k])        
        if k+1!=len(plt_lbls):
            dsh_inds = range(optim_trn_epochs[:k].sum(), optim_trn_epochs[:k].sum()+2)
            # plt.semilogy(dsh_inds, losses_trn[dsh_inds, 0], linestyle='dashed', linewidth=1, color=l1[0].get_color())
            plot_func(dsh_inds, losses_trn[dsh_inds, 0], linestyle='dashed', linewidth=1, color=l1[0].get_color())            
        plt.scatter(x=sctr_inds, y=losses_trn[sctr_inds,0], s=15, c=1000)
        if losses_tst is not None:
            plt.scatter(x=sctr_inds, y=losses_tst[sctr_inds,0], s=15, c=1000)
        
    plt.xlabel('(Super) epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.title(plt_tle, fontsize=fnt_sz)
    return plt_fig
    
class EqLearner():
    """
    
    """
    def __init__(self,
        net:torch.nn.Module,
        Lmbda:torch.Tensor,
        lib_func:BaseFuncLib,
        sprs_slvr:SparseRegressAlg,
        data_dict:dict,
        tmprl_ords:List[int],
        col_pnts_smplr:Rand_Col_Sampler=Rand_Col_Sampler(),
        N_col_pnts:int=10000,
        ntwk_out_names:List[str]=None,
        device:torch.device=torch.device('cpu'),
        data_type:torch.dtype=None
    ):
        
        self.net = net
        self.lmbda = Lmbda
        self.lib_func = lib_func
        self.sprs_slvr = sprs_slvr
        self.data_dict = data_dict
        self.tmprl_ords = tmprl_ords
        self.col_pnts_smplr = col_pnts_smplr
        self.N_col_pnts = N_col_pnts
        self.ntwk_out_names = ntwk_out_names
        self.device = device
        self.data_type = data_type

        dtime = datetime.datetime.now()
        self.sv_fname = f"LearnerCreatedOnY{dtime.year}M{dtime.month}D{dtime.day}at{dtime.hour}Hr{dtime.minute}Min{dtime.second}Sec{dtime.microsecond}Misec"
        
        self._AdamsPreTrnEpochs : int = None
        self._AdamsPreTrnLrnRt : float = None
        self._AdamsPreTrnAlpha : float = None
        self._AdmasPreTrnGamma : float = None
        self.AdamsPreTrnLambdas :np.ndarray = None
        self.AdamsPreTrnFvus : np.ndarray = None
        self.AdamsPreTrnLoss : np.ndarray = None
        self.AdamsPreTstLoss : np.ndarray = None
        self.AdamsPreTrnDataLoss : np.ndarray = None
        self.AdamsPreTstDataLoss : np.ndarray = None
        self.AdamsPreTrnEqLoss : np.ndarray = None
        self.AdamsPreTstEqLoss : np.ndarray = None
        self.AdamsPreLpLosses : np.ndarray = None
        
        self._LBFGsPreTrnEpochs : int = None
        self._LBFGsPreTrnLrnRt : float = None
        self._LBFGsPreTrnAlpha : float = None
        self._LBFGsPreTrnGamma : float = None
        self.LbfgsPreTrnLoss : np.ndarray = None
        self.LbfgsPreTstLoss : np.ndarray = None
        self.LbfgsPreTrnDataLoss : np.ndarray = None
        self.LbfgsPreTstDataLoss : np.ndarray = None
        self.LbfgsPreTrnEqLoss : np.ndarray = None
        self.LbfgsPreTstEqLoss : np.ndarray = None
        self.LbfgsPreTrnLambdas : np.ndarray = None
        self.LbfgsPreTrnFvus : np.ndarray = None
        self.LbfgsPreLpLosses : np.ndarray = None
        
        self._ADO_iters : int = None
        self.earl_lmbds : np.ndarray = None
        self.earl_fvus : np.ndarray = None
        self._ADO_lambdas : np.ndarray = None
        self._ADO_FVUs : np.ndarray = None
        self._ADO_alphas : np.ndarray = None
        self._ADO_epchs : np.ndarray = None
        self.sprs_crs_val_folds : int = None
        self._ADOTrnLosses : np.ndarray = None
        self._ADOTstLosses : np.ndarray = None
        self._ADOtrnDataLs : np.ndarray = None
        self._ADOtstDataLs : np.ndarray = None
        self._ADOtrnColloLs : np.ndarray = None
        self._ADOtstColloLs : np.ndarray = None
        self._AdoLpLosses : np.ndarray = None
        self._model_losses : np.ndarray = None
        self._model_complxtys : np.ndarray = None
        self._model_scores : np.ndarray = None

        
        self._PstTrnAdamsEpochs : int = None
        self._PstTrnAdamsLrnRt : float = None
        self._PstTrnAdamsAlpha : float = None
        self.AdamsPstTrnLambdas :np.ndarray = None
        self.AdamsPstTrnFvus :np.ndarray = None
        self.AdamsPstTrnLoss : np.ndarray = None
        self.AdamsPstTstLoss : np.ndarray = None
        self.AdamsPstTrnDataLoss : np.ndarray = None
        self.AdamsPstTstDataLoss : np.ndarray = None
        self.AdamsPstTrnEqLoss : np.ndarray = None
        self.AdamsPstTstEqLoss : np.ndarray = None
        self.AdamsPstLpLosses : np.ndarray = None
        
        self._PstTrnLBFGsEpochs : int = None
        self._PstTrnLBFGsLrnRt : float = None
        self._PstTrnLBFGsAlpha : float = None
        self.LbfgsPstTrnLambdas :np.ndarray = None
        self.LbfgsPstTrnFvus :np.ndarray = None
        self.LbfgsPstTrnLoss : np.ndarray = None
        self.LbfgsPstTstLoss : np.ndarray = None
        self.LbfgsPstTrnDataLoss : np.ndarray = None
        self.LbfgsPstTstDataLoss : np.ndarray = None
        self.LbfgsPstTrnEqLoss : np.ndarray = None
        self.LbfgsPstTstEqLoss : np.ndarray = None
        self.LbfgsPstLpLosses : np.ndarray = None

    # function still is a work in progress
    def Save_Model(self, data_set:str, fname:str=None)->None:
        """
        
        """
        og_dir = os.getcwd()
        try:
            os.mkdir('Saved_Models')
        except FileExistsError:
            print('Saved_Models'+' Directory already exists so did not create it')
        try: 
            os.chdir('Saved_Models')
        except (OSError, FileNotFoundError,PermissionError,NotADirectoryError):
            print('Could not change directory to {}. Animated Learned Plots will be writen to mp4 file in CWD={}'.format('Saved_Models', os.getcwd()))
        try:
            os.mkdir(data_set)
        except FileExistsError:
            print('{} Directory already exists so did not create it'.format(data_set))
        try: 
            os.chdir(data_set)
        except (OSError, FileNotFoundError,PermissionError,NotADirectoryError):
            print('Could not change directory to {}. Animated Learned Plots will be writen to mp4 file in CWD={}'.format(data_set, os.getcwd()))
        chckPntDic = {
            'net_state_dict': self.net.state_dict(),        # First the model's parameters then all the class variables/parameters should be the exact same as the in init method.
            'lambda': self.lmbda.data.cpu().numpy(),        # Adams Pretrain variables
            "_AdamsPreTrnEpochs":self._AdamsPreTrnEpochs,
            "_AdamsPreTrnLrnRt":self._AdamsPreTrnLrnRt,
            "_AdamsPreTrnAlpha":self._AdamsPreTrnAlpha,
            "_AdmasPreTrnGamma":self._AdmasPreTrnGamma,
            "AdamsPreTrnLambdas":self.AdamsPreTrnLambdas,
            "AdamsPreTrnFvus":self.AdamsPreTrnFvus,
            "AdamsPreTrnLoss":self.AdamsPreTrnLoss,
            "AdamsPreTstLoss":self.AdamsPreTstLoss,
            "AdamsPreTrnDataLoss":self.AdamsPreTrnDataLoss,
            "AdamsPreTstDataLoss":self.AdamsPreTstDataLoss,
            "AdamsPreTrnEqLoss":self.AdamsPreTrnEqLoss,
            "AdamsPreTstEqLoss":self.AdamsPreTstEqLoss,
            "AdamsPreLpLosses":self.AdamsPreLpLosses,       # LBFGs Pretrain variables
            "_LBFGsPreTrnEpochs":self._LBFGsPreTrnEpochs,
            "_LBFGsPreTrnLrnRt":self._LBFGsPreTrnLrnRt,
            "_LBFGsPreTrnAlpha":self._LBFGsPreTrnAlpha,
            "_LBFGsPreTrnGamma":self._LBFGsPreTrnGamma,
            "LbfgsPreTrnLoss":self.LbfgsPreTrnLoss,
            "LbfgsPreTstLoss":self.LbfgsPreTstLoss,
            "LbfgsPreTrnDataLoss":self.LbfgsPreTrnDataLoss,
            "LbfgsPreTstDataLoss":self.LbfgsPreTstDataLoss,
            "LbfgsPreTrnEqLoss":self.LbfgsPreTrnEqLoss,
            "LbfgsPreTstEqLoss":self.LbfgsPreTstEqLoss,
            "LbfgsPreTrnLambdas":self.LbfgsPreTrnLambdas,
            "LbfgsPreTrnFvus":self.LbfgsPreTrnFvus,
            "LbfgsPreLpLosses":self.LbfgsPreLpLosses,
            'N_col_pnts': self.N_col_pnts,                  # Now the stuff for ADO-Like Alg
            "_ADO_iters": self._ADO_iters,
            "earl_lmbds": self.earl_lmbds,
            "earl_fvus": self.earl_fvus,
            "_ADO_lambdas": self._ADO_lambdas,
            "_ADO_FVUs": self._ADO_FVUs,
            "_ADO_alphas": self._ADO_alphas,
            "_ADO_epchs": self._ADO_epchs,
            "sprs_crs_val_folds": self.sprs_crs_val_folds,
            "_ADOTrnLosses": self._ADOTrnLosses,
            "_ADOTstLosses": self._ADOTstLosses,
            "_ADOtrnDataLs": self._ADOtrnDataLs,
            "_ADOtstDataLs": self._ADOtstDataLs,
            "_ADOtrnColloLs": self._ADOtrnColloLs,
            "_ADOtstColloLs": self._ADOtstColloLs,
            "_AdoLpLosses": self._AdoLpLosses,
            "_model_losses": self._model_losses,
            "_model_complxtys": self._model_complxtys,
            "_model_scores": self._model_scores,            # Now the post training stuff
            "_PstTrnAdamsEpochs": self._PstTrnAdamsEpochs,
            "_PstTrnAdamsLrnRt": self._PstTrnAdamsLrnRt,
            "_PstTrnAdamsAlpha": self._PstTrnAdamsAlpha,
            "AdamsPstTrnLambdas": self.AdamsPstTrnLambdas,
            "AdamsPstTrnFvus": self.AdamsPstTrnFvus,
            "AdamsPstTrnLoss": self.AdamsPstTrnLoss,
            "AdamsPstTstLoss": self.AdamsPstTstLoss,
            "AdamsPstTrnDataLoss": self.AdamsPstTrnDataLoss,
            "AdamsPstTstDataLoss": self.AdamsPstTstDataLoss,
            "AdamsPstTrnEqLoss": self.AdamsPstTrnEqLoss,
            "AdamsPstTstEqLoss": self.AdamsPstTstEqLoss,
            "AdamsPstLpLosses": self.AdamsPstLpLosses,
            "_PstTrnLBFGsEpochs": self._PstTrnLBFGsEpochs,
            "_PstTrnLBFGsLrnRt": self._PstTrnLBFGsLrnRt,
            "_PstTrnLBFGsAlpha": self._PstTrnLBFGsAlpha,    # LBFGs Post training stuff
            "LbfgsPstTrnLambdas": self.LbfgsPstTrnLambdas,
            "LbfgsPstTrnFvus": self.LbfgsPstTrnFvus,
            "LbfgsPstTrnLoss": self.LbfgsPstTrnLoss,
            "LbfgsPstTstLoss": self.LbfgsPstTstLoss,
            "LbfgsPstTrnDataLoss": self.LbfgsPstTrnDataLoss,
            "LbfgsPstTstDataLoss": self.LbfgsPstTstDataLoss,
            "LbfgsPstTrnEqLoss": self.LbfgsPstTrnEqLoss,
            "LbfgsPstTstEqLoss": self.LbfgsPstTstEqLoss,
            "LbfgsPstLpLosses": self.LbfgsPstLpLosses,
        }
        if fname is None:
            fname = self.sv_fname

        path = fname+'.tar'
        torch.save(chckPntDic, path)
        try:
            os.chdir(og_dir)
        except (OSError, FileNotFoundError, PermissionError, NotADirectoryError):
            print(f"Could not change back to the original working directory/folder after changing to save the model checkpoint.")
    # function still is a work in progress
    def Load_Model(self,folder_loc:str, fname:str)->None:
        """
        
        """
        path = folder_loc +'/'+fname+'.tar'
        state = torch.load(f=path, weights_only=False)

        self.net.load_state_dict(state['net_state_dict'])
        self.lmbda = torch.from_numpy(state['lambda']).to(device=self.device, dtype=self.data_type).requires_grad_(True)

        self._AdamsPreTrnEpochs = state["_AdamsPreTrnEpochs"]
        self._AdamsPreTrnLrnRt = state["_AdamsPreTrnLrnRt"]
        self._AdamsPreTrnAlpha = state["_AdamsPreTrnAlpha"]
        self._AdmasPreTrnGamma = state["_AdmasPreTrnGamma"]
        self.AdamsPreTrnLambdas = state["AdamsPreTrnLambdas"]
        self.AdamsPreTrnFvus = state["AdamsPreTrnFvus"]
        self.AdamsPreTrnLoss = state["AdamsPreTrnLoss"]
        self.AdamsPreTstLoss = state["AdamsPreTstLoss"]
        self.AdamsPreTrnDataLoss = state["AdamsPreTrnDataLoss"]
        self.AdamsPreTstDataLoss = state["AdamsPreTstDataLoss"]
        self.AdamsPreTrnEqLoss = state["AdamsPreTrnEqLoss"]
        self.AdamsPreTstEqLoss = state["AdamsPreTstEqLoss"]
        self.AdamsPreLpLosses = state["AdamsPreLpLosses"]
        self._LBFGsPreTrnEpochs = state["_LBFGsPreTrnEpochs"]
        self._LBFGsPreTrnLrnRt = state["_LBFGsPreTrnLrnRt"]
        self._LBFGsPreTrnAlpha = state["_LBFGsPreTrnAlpha"]
        self._LBFGsPreTrnGamma = state["_LBFGsPreTrnGamma"]
        self.LbfgsPreTrnLoss = state["LbfgsPreTrnLoss"]
        self.LbfgsPreTstLoss = state["LbfgsPreTstLoss"]
        self.LbfgsPreTrnDataLoss = state["LbfgsPreTrnDataLoss"]
        self.LbfgsPreTstDataLoss = state["LbfgsPreTstDataLoss"]
        self.LbfgsPreTrnEqLoss = state["LbfgsPreTrnEqLoss"]
        self.LbfgsPreTstEqLoss = state["LbfgsPreTstEqLoss"]
        self.LbfgsPreTrnLambdas = state["LbfgsPreTrnLambdas"]
        self.LbfgsPreTrnFvus = state["LbfgsPreTrnFvus"]
        self.LbfgsPreLpLosses = state["LbfgsPreLpLosses"]
        self.N_col_pnts = state["N_col_pnts"]
        self._ADO_iters = state["_ADO_iters"]
        self.earl_lmbds = state["earl_lmbds"]
        self.earl_fvus = state["earl_fvus"]
        self._ADO_lambdas = state["_ADO_lambdas"]
        self._ADO_FVUs = state["_ADO_FVUs"]
        self._ADO_alphas = state["_ADO_alphas"]
        self._ADO_epchs = state["_ADO_epchs"]
        self.sprs_crs_val_folds = state["sprs_crs_val_folds"]
        self._ADOTrnLosses = state["_ADOTrnLosses"]
        self._ADOTstLosses = state["_ADOTstLosses"]
        self._ADOtrnDataLs = state["_ADOtrnDataLs"]
        self._ADOtstDataLs = state["_ADOtstDataLs"]
        self._ADOtrnColloLs = state["_ADOtrnColloLs"]
        self._ADOtstColloLs = state["_ADOtstColloLs"]
        self._AdoLpLosses = state["_AdoLpLosses"]
        self._model_losses = state["_model_losses"]
        self._model_complxtys = state["_model_complxtys"]
        self._model_scores = state["_model_scores"]
        self._PstTrnAdamsEpochs = state["_PstTrnAdamsEpochs"]
        self._PstTrnAdamsLrnRt = state["_PstTrnAdamsLrnRt"]
        self._PstTrnAdamsAlpha = state["_PstTrnAdamsAlpha"]
        self.AdamsPstTrnLambdas = state["AdamsPstTrnLambdas"]
        self.AdamsPstTrnFvus = state["AdamsPstTrnFvus"]
        self.AdamsPstTrnLoss = state["AdamsPstTrnLoss"]
        self.AdamsPstTstLoss = state["AdamsPstTstLoss"]
        self.AdamsPstTrnDataLoss = state["AdamsPstTrnDataLoss"]
        self.AdamsPstTstDataLoss = state["AdamsPstTstDataLoss"]
        self.AdamsPstTrnEqLoss = state["AdamsPstTrnEqLoss"]
        self.AdamsPstTstEqLoss = state["AdamsPstTstEqLoss"]
        self.AdamsPstLpLosses = state["AdamsPstLpLosses"]
        self._PstTrnLBFGsEpochs = state["_PstTrnLBFGsEpochs"]
        self._PstTrnLBFGsLrnRt = state["_PstTrnLBFGsLrnRt"]
        self._PstTrnLBFGsAlpha = state["_PstTrnLBFGsAlpha"]
        self.LbfgsPstTrnLambdas = state["LbfgsPstTrnLambdas"]
        self.LbfgsPstTrnFvus = state["LbfgsPstTrnFvus"]
        self.LbfgsPstTrnLoss = state["LbfgsPstTrnLoss"]
        self.LbfgsPstTstLoss = state["LbfgsPstTstLoss"]
        self.LbfgsPstTrnDataLoss = state["LbfgsPstTrnDataLoss"]
        self.LbfgsPstTstDataLoss = state["LbfgsPstTstDataLoss"]
        self.LbfgsPstTrnEqLoss = state["LbfgsPstTrnEqLoss"]
        self.LbfgsPstTstEqLoss = state["LbfgsPstTstEqLoss"]
        self.LbfgsPstLpLosses = state["LbfgsPstLpLosses"]

    def Learned_EQ(self, coefs:torch.Tensor=None, output:bool=False, sup_zeros:bool=True, dec_rnd:int=8, prnt_sig_dif:int=None)->str:
        """
        Returned what the current learned equations it. This is done by using current values in the Lambda
        matrix/vector/tensor and the current library terms in the following way - 
            Lambda(0)libraryfuncs(0) + Lambda(1)libraryfuncs(1) + .... + Lambda(end)libraryfuncs(end)
        So the number of library terms needs to be the  same as the number of elements in Lambda. Input 
        arguments/parameters are the following -
            EqLHS: the Left Hand Side of the equation if you want it included in the returned Diff Eq.
                Optional but if included needs to be a string.
            output: Boolean on whether or not (False) to print the learned equation to standard out 
            sup_zeros: (Suppress Zeros) Whether or not to suppress the terms in the library that have a 
                0 coefficient that is multiplying them. If true any library function that would be 
                multiplied by a zero is not printed
            dec_rnd: (decimal round) the number of decimal places to round up to when printing the 
                current learned equation.
        return - The learned equation as a string
        TODO - Possibly changed the returned equation to have the Left Hand Side of the pde since right 
            now it only has the RHS returned but in the training I have it explicitly set to have the
            LHS as the first temporal partial so probably merits to having this function return u_t = ....
            Also maybe have this update a class variable that contains the current learned equation (03/11/2022)
        """

        library_names = self.lib_func.get_library_names(net_out_func_names=self.ntwk_out_names)
        n = len(library_names)
        if coefs is None:
            coefs = torch.round(torch.clone(self.lmbda.data.detach()), decimals=dec_rnd)

        if n != coefs.shape[0] or n==0:
            print('ERROR! - The number of Library function names is not the same as the number of Lambda elements')
            print('or the Library of function names is empty - Returning None')
            return None
        if not isinstance(dec_rnd, int):
            print('Given decimal rounding number is not a int. Will default to using 8 decimal places to round')
            dec_rnd=8
        if not isinstance(prnt_sig_dif, int):
            prnt_sig_dif=8
        # Now that the little amount of argument checking has been done lets print out current learned equation
        N_outs = coefs.shape[-1]
        if self.ntwk_out_names:
            EqLHs = self.ntwk_out_names
        else:
            EqLHs = ["O"+str(j+1) for j in range(N_outs)]
        
        lrnd_eq = ""
        plus = ' + '
        num_frmtr = "%3.{}e".format(prnt_sig_dif)
        for k in range(N_outs):
            lrnd_eq = lrnd_eq + EqLHs[k] + "_"+self.tmprl_ords[k]*"t" +" = "
            
            for i in range(n):
                # num = str(np.around(coefs[i,k].item(), decimals=dec_rnd))
                num = num_frmtr % (np.around(coefs[i,k].item(), decimals=dec_rnd))
                if coefs[i,k].item() == 0 and sup_zeros:
                    continue
                lrnd_eq = lrnd_eq + num +'*'+library_names[i]
                if i != n-1:
                    lrnd_eq = lrnd_eq + plus
            if lrnd_eq[len(lrnd_eq)-2] == '+':
                lrnd_eq = lrnd_eq[0:len(lrnd_eq)-2] # has an extra space at the end of the sting
                # lrnd_eq = lrnd_eq[len(lrnd_eq)-3] # does not have a space at the end of the string
            if k!=N_outs-1:
                lrnd_eq = lrnd_eq + "\n"
        
        if output:
            print('The current Learned Equation(s) is(are) ... ')
            print(lrnd_eq)
        return lrnd_eq

    def FVU_Calc(self, lib_ceofs:torch.Tensor):
        """
            Calculate the Fraction of Variance Unexplained (FVU) for the 
            linearly regressioned learned equation using the coefficient 
            values offound in the lmbda parameter at any moment. Data used 
            to find the FVU is the test/validation data set. 
        """
        # Set Up stuff
        pnts = torch.from_numpy(self.data_dict['Test_Inputs']).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        trgts = Nth_temporal_prtls(values=self.net(pnts), pts=pnts, orders=self.tmprl_ords).detach()
        trg_means = trgts.mean(dim=0)
        lib_evals = self.lib_func.Calc(network=self.net, inpts=pnts).detach()
        estimates = lib_evals @ lib_ceofs
        # Calculate the Sum of Squared pred errors (SSerr) and Total Sum of Squares (SStot)
        SSerr = torch.sum( (trgts - estimates)**2, dim=0)
        SStot = torch.sum( (trgts - trg_means)**2, dim=0)
        fvu = SSerr / SStot
        return fvu

    def AdamsOptimTraining(self, mode:str="pre", alpha:float=1.0, gamma:float=1e-5, 
                           min_epochs:int=None, max_epochs:int=1000,
                           lrn_rt:float=0.001, lp_ord:float=1.0, 
                           outputFreq:int=100,
                           betas:Tuple=(0.9, 0.99), eps:float=1e-8, 
                           wght_dcy:float=0, amsgrad:bool=False, 
                           threshold:bool=False, Save_File_Name:str=None):
        """
        
        """
        if mode.lower() not in ["pre", "post", "ado"]:
            raise ValueError(f"The \'mode\' input argument can only be the \'pre\', \'post\', or \'ado\'")
        if outputFreq is None or outputFreq <= 0:
            outputFreq = max_epochs//10

        # if Save_File_Name==None or (not isinstance(Save_File_Name, str)):
        #     msg = ("The Save_File_Name argument was None or was not a string object the file name  \n"
        #            "where the model will periodically be saved to is based off the day and time the \n" 
        #            "class object was created. ")
        #     warnings.warn(message=msg, stacklevel=1)
        #     Save_File_Name = self.sv_fname
        # Save_File_Name += ".pt"

        TrnLambdas = np.empty(shape=(2, *self.lmbda.data.shape), dtype=float)
        TrnFvus = np.empty(shape=(2, self.lmbda.data.shape[1]), dtype=float)
        TrnLambdas[0] = self.lmbda.data.cpu().numpy()
        TrnFvus[0] = self.FVU_Calc(lib_ceofs=self.lmbda.data).cpu().numpy()

        n_trgts = self.lmbda.size(-1)
        nzs = [ [i.item() for i in self.lmbda.data[:,j].nonzero()] for j in range(n_trgts)]

        optim = torch.optim.Adam(self.net.parameters(), lr=lrn_rt, betas=betas, eps=eps, weight_decay=wght_dcy, amsgrad=amsgrad)
        optim.param_groups[0]['params'].append(self.lmbda)
        # schdlr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.9, last_epoch=-1, ) 
        # schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10,
        #                                         threshold=0.001, threshold_mode='rel', cooldown=1, min_lr=0, 
        #                                             eps=1e-08, )

        train_inputs = torch.from_numpy(self.data_dict['Train_Inputs']).to(device=self.device, dtype=self.data_type)
        train_targets = torch.from_numpy(self.data_dict['Train_Targets']).to(device=self.device, dtype=self.data_type)
        test_inputs = torch.from_numpy(self.data_dict['Test_Inputs']).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        test_targets = torch.from_numpy(self.data_dict['Test_Targets']).to(device=self.device, dtype=self.data_type)
        np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        self.nDpnts = train_inputs.size(0)
        
        tot_loss = []
        data_loss = []
        pde_loss = []
        lp_loss = []
        tst_loss = []
        tst_data_loss = []
        tst_pde_loss = []

        if mode.lower()=="pre":
            nzs = [ list(range(self.lmbda.data.size(0))) for _ in range(n_trgts)]
            print("Beginning Adams Pre-Training Optimization Now")
            tracker = LossTracker(mode="decreasing", chng_mode="rel", patience=150, eps=1e-4, track_mode="network", min_iters=min_epochs)
            # tracker = LossTracker(mode="decreasing", chng_mode="abs", patience=150, eps=1e-6, track_mode="network", min_iters=min_epochs)

        elif mode.lower()=="ado":
            print("Beginning Adams ADO-Training Optimization Now")
            tracker = LossTracker(mode="decreasing", chng_mode="rel", patience=100, eps=1e-4, track_mode="network", min_iters=min_epochs)
            # tracker = LossTracker(mode="decreasing", chng_mode="abs", patience=100, eps=1e-6, track_mode="network", min_iters=min_epochs)

        elif mode.lower()=="post":
            print("Beginning Adams Post-Training Optimization Now")
            tracker = LossTracker(mode="increasing", chng_mode="rel", patience=200, eps=1e-4, track_mode="lp", min_iters=min_epochs)
            # tracker = LossTracker(mode="increasing", chng_mode="abs", patience=200, eps=1e-6, track_mode="lp", min_iters=min_epochs)

        print(f"            | Data  Type |  Data Loss  |  Pde Loss   | Lp Loss")
        print("-"*75)

        for i in range(max_epochs):
            self.net.train(True)
            
            # colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
            colpnts.grad = None
                
            optim.zero_grad(set_to_none=True)
            self.net.train(True)

            dataL = torch.mean((self.net(train_inputs) - train_targets)**2, dim=0)
            col_preds = self.net(colpnts)
            t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords)
            lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts)
            pdeL = torch.zeros(size=(n_trgts,), device=self.device, dtype=self.data_type)
            # lp_lss = Torch_Lp_Loss(Xis=self.lmbda.data, p=lp_ord)
            lp_lss = Torch_Lp_Loss(Xis=self.lmbda, p=lp_ord)
            for j in range(n_trgts): 
                pdeL[j] = torch.mean((t_prtls[:,j:j+1] - lib_evals[:, nzs[j]] @ self.lmbda[nzs[j],j:j+1])**2, dim=0)
            loss = torch.sum(dataL + alpha*pdeL + gamma*lp_lss)
            loss.backward()

            tot_loss.append(loss.detach())
            data_loss.append(dataL.detach())
            pde_loss.append(pdeL.detach())
            lp_loss.append(lp_lss.detach())

            optim.step()
            
            self.net.eval()
            tst_dataL = torch.mean((self.net(test_inputs) - test_targets)**2, dim=0).detach_()
            tst_col_preds = self.net(test_inputs)
            tst_t_prtls = Nth_temporal_prtls(values=tst_col_preds, pts=test_inputs, orders=self.tmprl_ords).detach_()
            tst_lib_evals = self.lib_func.Calc(network=self.net, inpts=test_inputs)
            tst_pdeL = torch.mean((tst_t_prtls - tst_lib_evals@self.lmbda)**2, dim=0,).detach_()
            ts_loss = torch.sum(tst_dataL + alpha*tst_pdeL).detach_()
            tst_loss.append( ts_loss.detach() )
            tst_data_loss.append( tst_dataL.detach() )
            tst_pde_loss.append( tst_pdeL.detach() )

            # if i%outputFreq==0 and i>0:
            if i%outputFreq==0:
                print(f"Epoch {i:5d} | Train Data | {dataL.sum().item():.5e} | {pdeL.sum().item():.5e} | {lp_lss.sum().item():.4e}")
                print(f"            | Test Data  | {tst_dataL.sum().item():.5e} | {tst_pdeL.sum().item():.5e} | ")
                # print(f"            | Test Data  | {tst_dataL.sum().item():.5e} | {tst_pdeL.sum().item():.5e} | {lp_lss.sum().item():.4e}")


            if tracker(data_loss=tst_dataL.sum().detach(), eq_loss=tst_pdeL.sum().detach(), lp_loss=Torch_Lp_Loss(self.lmbda.data, p=1.0).sum(), net=self.net, lib_coefs=self.lmbda):
                break

        if max_epochs>0:
            # Can only do this check if the above loop was run at least once - possible
            # for it to not have been run if epochs=0
            if (i+1)==max_epochs:
                if os.path.isfile(tracker.wghts_bs_file):
                    tracker._load_weights_bais(network=self.net, eq_coefs=self.lmbda)
                    os.remove(path=tracker.wghts_bs_file)

        optim.zero_grad(set_to_none=True)
        del train_inputs, train_targets, test_inputs, test_targets, colpnts

        if threshold:
            self.lmbda.data = torch.where(torch.abs(self.lmbda.data)<= 5e-4, torch.tensor(0.0, device=self.device, dtype=self.data_type), self.lmbda.data)
            self.thresheld = True
        
        self.trn_batch_size =1
        cmpltd_epochs = len(tst_loss)
        if cmpltd_epochs>0:
            TrnLosses = torch.vstack(tot_loss).cpu().numpy()
            TstLosses = torch.vstack(tst_loss).cpu().numpy()
            TrnDataLosses = torch.vstack(data_loss).cpu().numpy()
            TstDataLosses = torch.vstack(tst_data_loss).cpu().numpy()
            TrnEqLosses = torch.vstack(pde_loss).cpu().numpy()
            TstEqLosses = torch.vstack(tst_pde_loss).cpu().numpy()
            LpLosses = torch.vstack(lp_loss).cpu().numpy()
        else:
            TrnLosses = np.zeros(shape=(1, n_trgts))
            TstLosses = np.zeros(shape=(1, n_trgts))
            TrnDataLosses = np.zeros(shape=(1, n_trgts))
            TstDataLosses = np.zeros(shape=(1, n_trgts))
            TrnEqLosses = np.zeros(shape=(1, n_trgts))
            TstEqLosses = np.zeros(shape=(1, n_trgts))
            LpLosses = np.zeros(shape=(1, n_trgts))
        TrnLambdas[1] = self.lmbda.data.cpu().numpy()
        TrnFvus[1] = self.FVU_Calc(lib_ceofs=self.lmbda.data).cpu().numpy()

        if mode=="pre":
            self.AdamsPreTrnLambdas = TrnLambdas
            self.AdamsPreTrnFvus = TrnFvus
            self._AdamsPreTrnEpochs = cmpltd_epochs
            self._AdamsPreTrnLrnRt = lrn_rt
            self._AdamsPreTrnAlpha = alpha
            self._AdmasPreTrnGamma = gamma
            self.AdamsPreTrnLoss = TrnLosses
            self.AdamsPreTstLoss = TstLosses
            self.AdamsPreTrnDataLoss = TrnDataLosses
            self.AdamsPreTstDataLoss = TstDataLosses
            self.AdamsPreTrnEqLoss = TrnEqLosses
            self.AdamsPreTstEqLoss = TstEqLosses
            self.AdamsPreLpLosses = LpLosses

        elif mode=="post":
            print('Finished Adams Post Training')
            self.AdamsPstTrnLambdas = TrnLambdas
            self.AdamsPstTrnFvus = TrnFvus
            self._PstTrnAdamsEpochs = cmpltd_epochs
            self._PstTrnAdamsLrnRt = lrn_rt
            self._PstTrnAdamsAlpha = alpha
            self.AdamsPstTrnLoss = TrnLosses
            self.AdamsPstTstLoss = TstLosses
            self.AdamsPstTrnDataLoss = TrnDataLosses
            self.AdamsPstTstDataLoss = TstDataLosses
            self.AdamsPstTrnEqLoss = TrnEqLosses
            self.AdamsPstTstEqLoss = TstEqLosses
            self.AdamsPstLpLosses = LpLosses

        # elif mode=="ado":
        #     # do some stuff here or in the ADO method?

        return (TrnLosses, TstLosses, 
                TrnDataLosses, TstDataLosses, 
                TrnEqLosses, TstEqLosses, 
                LpLosses)

    def LbfgsOptimTraining(self, mode:str="pre", alpha:float=1.0, gamma:float=1e-5, 
                           min_epochs:int=None, max_epochs:int=1000,
                           lrn_rt:float=0.001, lp_ord:float=1.0, 
                           outputFreq:int=100,
                           max_it:int=20, max_evl:int=None, grad_tol:float=1e-07, tol_change:float=1e-09,
                           history_size:int=100, line_srch_fn:str=None,
                           threshold:bool=False, Save_File_Name:str=None):
        """
        
        """
        if mode.lower() not in ["pre", "post", "ado"]:
            raise ValueError(f"The \'mode\' input argument can only be the \'pre\', \'post\', or \'ado\'")
        if outputFreq is None or outputFreq <= 0:
            outputFreq = max_epochs//10
            
        TrnLambdas = np.empty(shape=(2, *self.lmbda.data.shape), dtype=float)
        TrnFvus = np.empty(shape=(2, self.lmbda.data.shape[1]), dtype=float)
        TrnLambdas[0] = self.lmbda.data.cpu().numpy()
        TrnFvus[0] = self.FVU_Calc(lib_ceofs=self.lmbda.data).cpu().numpy()

        n_trgts = self.lmbda.size(-1)
        nzs = [ [i.item() for i in self.lmbda.data[:,j].nonzero()] for j in range(n_trgts)]

        optim = torch.optim.LBFGS(params=self.net.parameters(), lr=lrn_rt, max_iter=max_it, max_eval=max_evl,
            tolerance_grad=grad_tol, tolerance_change=tol_change, history_size=history_size, line_search_fn=line_srch_fn)
        optim.param_groups[0]['params'].append(self.lmbda)
        # schdlr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.9, last_epoch=-1, ) 
        # schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10,
        #                                         threshold=0.001, threshold_mode='rel', cooldown=1, min_lr=0, 
        #                                             eps=1e-08, )

        train_inputs = torch.from_numpy(self.data_dict['Train_Inputs']).to(device=self.device, dtype=self.data_type)
        train_targets = torch.from_numpy(self.data_dict['Train_Targets']).to(device=self.device, dtype=self.data_type)
        test_inputs = torch.from_numpy(self.data_dict['Test_Inputs']).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        test_targets = torch.from_numpy(self.data_dict['Test_Targets']).to(device=self.device, dtype=self.data_type)
        np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        self.nDpnts = train_inputs.size(0)

        def closure():
            colpnts.grad = None
            optim.zero_grad(set_to_none=True)

            dataL = torch.mean((self.net(train_inputs) - train_targets)**2, dim=0)
            col_preds = self.net(colpnts)
            t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords)
            lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts)
            pdeL = torch.zeros(size=(n_trgts,), device=self.device, dtype=self.data_type)
            for j in range(n_trgts): 
                pdeL[j] = torch.mean((t_prtls[:,j:j+1] - lib_evals[:, nzs[j]] @ self.lmbda[nzs[j],j:j+1])**2, dim=0)
            # lp_lss = Torch_Lp_Loss(Xis=self.lmbda.data, p=lp_ord)
            lp_lss = Torch_Lp_Loss(Xis=self.lmbda, p=lp_ord)
            loss = torch.sum(dataL + alpha*pdeL + gamma*lp_lss)

            # loss.backward()
            if loss.isnan():
                    print(f"Note that dataLs={dataL} colloLs={pdeL} and loss={loss}")
                    optim.zero_grad(set_to_none=True)
            if loss.requires_grad and torch.isfinite(loss):
                loss.backward()
            # elif loss.isnan().sum():
            #     optim.zero_grad(set_to_none=True)
            return loss
        
        tot_loss = []
        data_loss = []
        pde_loss = []
        lp_loss = []
        tst_loss = []
        tst_data_loss = []
        tst_pde_loss = []

        if mode.lower()=="pre":
            nzs = [ list(range(self.lmbda.data.size(0))) for _ in range(n_trgts)]
            print("Beginning LBFGS Pre-Training Optimization Now")
            tracker = LossTracker(mode="decreasing", chng_mode="rel", patience=20, eps=1e-4, track_mode="network", min_iters=min_epochs)
            # tracker = LossTracker(mode="decreasing", chng_mode="abs", patience=20, eps=1e-6, track_mode="network", min_iters=min_epochs)

        elif mode.lower()=="ado":
            print("Beginning LBFGS ADO-Training Optimization Now")
            tracker = LossTracker(mode="decreasing", chng_mode="rel", patience=20, eps=1e-4, track_mode="network", min_iters=min_epochs)
            # tracker = LossTracker(mode="decreasing", chng_mode="abs", patience=20, eps=1e-6, track_mode="network", min_iters=min_epochs)

        elif mode.lower()=="post":
            print("Beginning LBFGS Post-Training Optimization Now")
            tracker = LossTracker(mode="increasing", chng_mode="rel", patience=20, eps=1e-4, track_mode="lp", min_iters=min_epochs)
            # tracker = LossTracker(mode="increasing", chng_mode="abs", patience=20, eps=1e-6, track_mode="lp", min_iters=min_epochs)

        print(f"            | Data  Type |  Data Loss  |  Pde Loss   | Lp Loss")
        print("-"*75)

        for i in range(max_epochs):
            self.net.train(mode=True)

            optim.step(closure)
            # if schdlr_freq and i>0 and i%schdlr_freq==0:
            #     schdlr.step()
                # schdlr.step(metrics=loss)

            self.net.eval()
            colpnts.grad = None

            trn_dataL = torch.mean((self.net(train_inputs) - train_targets)**2, dim=0)
            trn_col_preds = self.net(colpnts)
            trn_t_prtls = Nth_temporal_prtls(values=trn_col_preds, pts=colpnts, orders=self.tmprl_ords)
            trn_lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts)
            trn_pdeL = torch.zeros(size=(n_trgts,), device=self.device, dtype=self.data_type)
            for j in range(n_trgts): 
                trn_pdeL[j] = torch.mean((trn_t_prtls[:,j:j+1] - trn_lib_evals[:, nzs[j]] @ self.lmbda[nzs[j],j:j+1])**2, dim=0)
            # trn_lp_lss = Torch_Lp_Loss(Xis=self.lmbda.data, p=lp_ord)
            trn_lp_lss = Torch_Lp_Loss(Xis=self.lmbda.data, p=lp_ord)
            trn_loss = torch.sum(trn_dataL + alpha*trn_pdeL + gamma*trn_lp_lss)

            tot_loss.append(trn_loss.detach())
            data_loss.append(trn_dataL.detach())
            pde_loss.append(trn_pdeL.detach())
            lp_loss.append(trn_lp_lss.detach())
            
            
            tst_dataL = torch.mean((self.net(test_inputs) - test_targets)**2, dim=0).detach_()
            tst_col_preds = self.net(test_inputs)
            tst_t_prtls = Nth_temporal_prtls(values=tst_col_preds, pts=test_inputs, orders=self.tmprl_ords).detach_()
            tst_lib_evals = self.lib_func.Calc(network=self.net, inpts=test_inputs)
            tst_pdeL = torch.mean((tst_t_prtls - tst_lib_evals@self.lmbda)**2, dim=0,).detach_()
            ts_loss = torch.sum(tst_dataL + alpha*tst_pdeL).detach_()
            tst_loss.append( ts_loss.detach() )
            tst_data_loss.append( tst_dataL.detach() )
            tst_pde_loss.append( tst_pdeL.detach() )

            # if i%outputFreq==0 and i>0:
            if i%outputFreq==0:
                print(f"Epoch {i:5d} | Train Data | {trn_dataL.sum().item():.5e} | {trn_pdeL.sum().item():.5e} | {trn_lp_lss.sum().item():.4e}")
                print(f"            | Test Data  | {tst_dataL.sum().item():.5e} | {tst_pdeL.sum().item():.5e} | ")
                # print(f"            | Test Data  | {tst_dataL.sum().item():.5e} | {tst_pdeL.sum().item():.5e} | {lp_lss.sum().item():.4e}")


            if tracker(data_loss=tst_dataL.detach(), eq_loss=tst_pdeL.detach(), lp_loss=Torch_Lp_Loss(self.lmbda.data, p=1.0), net=self.net, lib_coefs=self.lmbda):
                break

        if max_epochs>0:
            # Can only do this check if the above loop was run at least once - possible
            # for it to not have been run if epochs=0
            if (i+1)==max_epochs:
                if os.path.isfile(tracker.wghts_bs_file):
                    tracker._load_weights_bais(network=self.net, eq_coefs=self.lmbda)
                    os.remove(path=tracker.wghts_bs_file)

        optim.zero_grad(set_to_none=True)
        del train_inputs, train_targets, test_inputs, test_targets, colpnts

        if threshold:
            self.lmbda.data = torch.where(torch.abs(self.lmbda.data)<= 5e-4, torch.tensor(0.0, device=self.device, dtype=self.data_type), self.lmbda.data)
            self.thresheld = True

        self.trn_batch_size = 1
        cmpltd_epochs = len(tst_loss)
        if cmpltd_epochs>0:
            TrnLosses = torch.vstack(tot_loss).cpu().numpy()
            TstLosses = torch.vstack(tst_loss).cpu().numpy()
            TrnDataLosses = torch.vstack(data_loss).cpu().numpy()
            TstDataLosses = torch.vstack(tst_data_loss).cpu().numpy()
            TrnEqLosses = torch.vstack(pde_loss).cpu().numpy()
            TstEqLosses = torch.vstack(tst_pde_loss).cpu().numpy()
            LpLosses = torch.vstack(lp_loss).cpu().numpy()
        else:
            TrnLosses = np.zeros(shape=(1, n_trgts))
            TstLosses = np.zeros(shape=(1, n_trgts))
            TrnDataLosses = np.zeros(shape=(1, n_trgts))
            TstDataLosses = np.zeros(shape=(1, n_trgts))
            TrnEqLosses = np.zeros(shape=(1, n_trgts))
            TstEqLosses = np.zeros(shape=(1, n_trgts))
            LpLosses = np.zeros(shape=(1, n_trgts))
        TrnLambdas[1] = self.lmbda.data.cpu().numpy()
        TrnFvus[1] = self.FVU_Calc(lib_ceofs=self.lmbda.data).cpu().numpy()

        if mode.lower()=="pre":
            self._LBFGsPreTrnEpochs = cmpltd_epochs
            self._LBFGsPreTrnLrnRt = lrn_rt
            self._LBFGsPreTrnAlpha = alpha
            self._LBFGsPreTrnGamma = gamma
            self.LbfgsPreTrnLoss = TrnLosses
            self.LbfgsPreTstLoss = TstLosses
            self.LbfgsPreTrnDataLoss = TrnDataLosses
            self.LbfgsPreTstDataLoss = TstDataLosses
            self.LbfgsPreTrnEqLoss = TrnEqLosses
            self.LbfgsPreTstEqLoss = TstEqLosses
            self.LbfgsPreTrnLambdas = LpLosses
            self.LbfgsPreTrnLambdas = TrnLambdas
            self.LbfgsPreTrnFvus = TrnFvus

        elif mode.lower()=="post":
            self._PstTrnLBFGsEpochs = cmpltd_epochs
            self._PstTrnLBFGsLrnRt = lrn_rt
            self._PstTrnLBFGsAlpha = alpha
            self.LbfgsPstTrnLoss = TrnLosses
            self.LbfgsPstTstLoss = TstLosses
            self.LbfgsPstTrnDataLoss = TrnDataLosses
            self.LbfgsPstTstDataLoss = TstDataLosses
            self.LbfgsPstTrnEqLoss = TrnEqLosses
            self.LbfgsPstTstEqLoss = TstEqLosses
            self.LbfgsPstLpLosses = LpLosses
            self.LbfgsPstTrnLambdas = TrnLambdas
            self.LbfgsPstTrnFvus = TrnFvus


        # elif mode=="ado":
        #     # do some stuff here or in the ADO method?

        return (TrnLosses, TstLosses, 
                TrnDataLosses, TstDataLosses, 
                TrnEqLosses, TstEqLosses, 
                LpLosses)

    def OptimTraining(self, method:str="adam", mode:str="pre",
                      alpha:float=0.5, gamma:float=0.25, 
                      min_epochs:int=None, max_epochs:int=1000,
                      lrn_rt:float=0.001, lp_ord:float=1.0, 
                      outputFreq:int=100, threshold:bool=False,
                      optim_input_dict:dict={})->Tuple[np.ndarray]:
        """
        
        """
        if not isinstance(method, str):
            raise TypeError(f"The \'method\' input argument needs to be a string argument.")
        if method.lower() not in ["adam", "lbfgs"]:
            raise ValueError(f"The \'method\' input argument can only be the \'adam\' or \'lbfgs\' ")
        if mode.lower() not in ["pre", "post", "ado"]:
            raise ValueError(f"The \'mode\' input argument can only be the \'pre\', \'post\', or \'ado\'")
        if outputFreq is None or outputFreq<= 0:
            outputFreq = max_epochs//10

        raise NotImplementedError()
            
        TrnLambdas = np.empty(shape=(2, *self.lmbda.data.shape), dtype=float)
        TrnFvus = np.empty(shape=(2, self.lmbda.data.shape[1]), dtype=float)
        TrnLambdas[0] = self.lmbda.data.cpu().numpy()
        TrnFvus[0] = self.FVU_Calc(lib_ceofs=self.lmbda.data).cpu().numpy()

        n_trgts = self.lmbda.size(-1)
        nzs = [ [i.item() for i in self.lmbda.data[:,j].nonzero()] for j in range(n_trgts)]

        if method.lower()=="adam":
            mthd_name = "Adam"
            optim = torch.optim.Adam(self.net.parameters(), lr=lrn_rt, **optim_input_dict)
        else:
            mthd_name = "LBFGS"
            optim = torch.optim.LBFGS(params=self.net.parameters(), lr=lrn_rt, **optim_input_dict)
        optim.param_groups[0]['params'].append(self.lmbda)
        # schdlr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.9, last_epoch=-1, ) 
        # schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10,
        #                                         threshold=0.001, threshold_mode='rel', cooldown=1, min_lr=0, 
        #                                             eps=1e-08, )

        train_inputs = torch.from_numpy(self.data_dict['Train_Inputs']).to(device=self.device, dtype=self.data_type)
        train_targets = torch.from_numpy(self.data_dict['Train_Targets']).to(device=self.device, dtype=self.data_type)
        test_inputs = torch.from_numpy(self.data_dict['Test_Inputs']).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        test_targets = torch.from_numpy(self.data_dict['Test_Targets']).to(device=self.device, dtype=self.data_type)
        np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        self.nDpnts = train_inputs.size(0)

        def closure():
            colpnts.grad = None
            optim.zero_grad(set_to_none=True)

            dataL = torch.mean((self.net(train_inputs) - train_targets)**2, dim=0)
            col_preds = self.net(colpnts)
            t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords)
            lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts)
            pdeL = torch.zeros(size=(n_trgts,), device=self.device, dtype=self.data_type)
            for j in range(n_trgts): 
                pdeL[j] = torch.mean((t_prtls[:,j:j+1] - lib_evals[:, nzs[j]] @ self.lmbda[nzs[j],j:j+1])**2, dim=0)
            # lp_lss = Torch_Lp_Loss(Xis=self.lmbda.data, p=lp_ord)
            lp_lss = Torch_Lp_Loss(Xis=self.lmbda, p=lp_ord)
            loss = torch.sum(dataL + alpha*pdeL + gamma*lp_lss)

            # loss.backward()
            if loss.isnan():
                    print(f"Note that dataLs={dataL} colloLs={pdeL} and loss={loss}")
                    optim.zero_grad(set_to_none=True)
            if loss.requires_grad and torch.isfinite(loss):
                loss.backward()
            return loss
        
        tot_loss = []
        data_loss = []
        pde_loss = []
        lp_loss = []
        tst_loss = []
        tst_data_loss = []
        tst_pde_loss = []

        if mode.lower()=="pre":
            nzs = [ list(range(self.lmbda.data.size(0))) for _ in range(n_trgts)]
            print(f"Beginning {mthd_name} Pre-Training Optimization Now")
            if mthd_name=="Adam":
                tracker = LossTracker(mode="decreasing", chng_mode="rel", patience=150, eps=1e-4, track_mode="network", min_iters=min_epochs)
                # tracker = LossTracker(mode="decreasing", chng_mode="abs", patience=150, eps=1e-6, track_mode="network", min_iters=min_epochs)
            else:
                tracker = LossTracker(mode="decreasing", chng_mode="rel", patience=20, eps=1e-4, track_mode="network", min_iters=min_epochs)
                # tracker = LossTracker(mode="decreasing", chng_mode="abs", patience=20, eps=1e-6, track_mode="network", min_iters=min_epochs)
        elif mode.lower()=="ado":
            print(f"Beginning {mthd_name} ADO-Training Optimization Now")
            if mthd_name=="Adam":
                tracker = LossTracker(mode="decreasing", chng_mode="rel", patience=100, eps=1e-4, track_mode="network", min_iters=min_epochs)
                # tracker = LossTracker(mode="decreasing", chng_mode="abs", patience=100, eps=1e-6, track_mode="network", min_iters=min_epochs)
            else:
                tracker = LossTracker(mode="decreasing", chng_mode="rel", patience=20, eps=1e-4, track_mode="network", min_iters=min_epochs)
                # tracker = LossTracker(mode="decreasing", chng_mode="abs", patience=20, eps=1e-6, track_mode="network", min_iters=min_epochs)
        elif mode.lower()=="post":
            print(f"Beginning {mthd_name} Post-Training Optimization Now")
            if mthd_name=="Adam":
                tracker = LossTracker(mode="increasing", chng_mode="rel", patience=200, eps=1e-4, track_mode="lp", min_iters=min_epochs)
                # tracker = LossTracker(mode="increasing", chng_mode="abs", patience=200, eps=1e-6, track_mode="lp", min_iters=min_epochs)
            else:
                tracker = LossTracker(mode="increasing", chng_mode="rel", patience=20, eps=1e-4, track_mode="lp", min_iters=min_epochs)
                # tracker = LossTracker(mode="increasing", chng_mode="abs", patience=20, eps=1e-6, track_mode="lp", min_iters=min_epochs)

        print(f"            | Data  Type |  Data Loss  |  Pde Loss   | Lp Loss")
        print("-"*75)

        for i in range(max_epochs):
            self.net.train(mode=True)

            optim.step(closure)
            # if schdlr_freq and i>0 and i%schdlr_freq==0:
            #     schdlr.step()
                # schdlr.step(metrics=loss)

            self.net.eval()
            colpnts.grad = None

            trn_dataL = torch.mean((self.net(train_inputs) - train_targets)**2, dim=0)
            trn_col_preds = self.net(colpnts)
            trn_t_prtls = Nth_temporal_prtls(values=trn_col_preds, pts=colpnts, orders=self.tmprl_ords)
            trn_lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts)
            trn_pdeL = torch.zeros(size=(n_trgts,), device=self.device, dtype=self.data_type)
            for j in range(n_trgts): 
                trn_pdeL[j] = torch.mean((trn_t_prtls[:,j:j+1] - trn_lib_evals[:, nzs[j]] @ self.lmbda[nzs[j],j:j+1])**2, dim=0)
            trn_# lp_lss = Torch_Lp_Loss(Xis=self.lmbda.data, p=lp_ord)
            lp_lss = Torch_Lp_Loss(Xis=self.lmbda, p=lp_ord)
            trn_loss = torch.sum(trn_dataL + alpha*trn_pdeL + gamma*trn_lp_lss)

            tot_loss.append(trn_loss.detach())
            data_loss.append(trn_dataL.detach())
            pde_loss.append(trn_pdeL.detach())
            lp_loss.append(trn_lp_lss.detach())
            
            
            tst_dataL = torch.mean((self.net(test_inputs) - test_targets)**2, dim=0).detach_()
            tst_col_preds = self.net(test_inputs)
            tst_t_prtls = Nth_temporal_prtls(values=tst_col_preds, pts=test_inputs, orders=self.tmprl_ords).detach_()
            tst_lib_evals = self.lib_func.Calc(network=self.net, inpts=test_inputs)
            tst_pdeL = torch.mean((tst_t_prtls - tst_lib_evals@self.lmbda)**2, dim=0,).detach_()
            ts_loss = torch.sum(tst_dataL + alpha*tst_pdeL).detach_()
            tst_loss.append( ts_loss.detach() )
            tst_data_loss.append( tst_dataL.detach() )
            tst_pde_loss.append( tst_pdeL.detach() )

            # if i%outputFreq==0 and i>0:
            if i%outputFreq==0:
                print(f"Epoch {i:5d} | Train Data | {trn_dataL.sum().item():.5e} | {trn_pdeL.sum().item():.5e} | {trn_lp_lss.sum().item():.4e}")
                print(f"            | Test Data  | {tst_dataL.sum().item():.5e} | {tst_pdeL.sum().item():.5e} | ")

            if tracker(data_loss=tst_dataL.detach(), eq_loss=tst_pdeL.detach(), lp_loss=Torch_Lp_Loss(self.lmbda.data, p=1.0), net=self.net, lib_coefs=self.lmbda):
                break

        if max_epochs>0:
            # Can only do this check if the above loop was run at least once - possible
            # for it to not have been run if epochs=0
            if (i+1)==max_epochs:
                if os.path.isfile(tracker.wghts_bs_file):
                    tracker._load_weights_bais(network=self.net, eq_coefs=self.lmbda)
                    os.remove(path=tracker.wghts_bs_file)

        optim.zero_grad(set_to_none=True)
        del train_inputs, train_targets, test_inputs, test_targets, colpnts

        # Now save the loss and shit to the class variables. 
        cmpltd_epochs = len(tst_loss)
        self.trn_batch_size = 1
        if cmpltd_epochs>0:
            TrnLosses = torch.vstack(tot_loss).cpu().numpy()
            TstLosses = torch.vstack(tst_loss).cpu().numpy()
            TrnDataLosses = torch.vstack(data_loss).cpu().numpy()
            TstDataLosses = torch.vstack(tst_data_loss).cpu().numpy()
            TrnEqLosses = torch.vstack(pde_loss).cpu().numpy()
            TstEqLosses = torch.vstack(tst_pde_loss).cpu().numpy()
            LpLosses = torch.vstack(lp_loss).cpu().numpy()
        else:
            TrnLosses = np.zeros(shape=(1, n_trgts))
            TstLosses = np.zeros(shape=(1, n_trgts))
            TrnDataLosses = np.zeros(shape=(1, n_trgts))
            TstDataLosses = np.zeros(shape=(1, n_trgts))
            TrnEqLosses = np.zeros(shape=(1, n_trgts))
            TstEqLosses = np.zeros(shape=(1, n_trgts))
            LpLosses = np.zeros(shape=(1, n_trgts))

        if mode=="pre" and mthd_name=="Adam":
            self.AdamsPreTrnLambdas = TrnLambdas
            self.AdamsPreTrnFvus = TrnFvus
            self._AdamsPreTrnEpochs = cmpltd_epochs
            self._AdamsPreTrnLrnRt = lrn_rt
            self._AdamsPreTrnAlpha = alpha
            self._AdmasPreTrnGamma = gamma
            self.AdamsPreTrnLoss = TrnLosses
            self.AdamsPreTstLoss = TstLosses
            self.AdamsPreTrnDataLoss = TrnDataLosses
            self.AdamsPreTstDataLoss = TstDataLosses
            self.AdamsPreTrnEqLoss = TrnEqLosses
            self.AdamsPreTstEqLoss = TstEqLosses
            self.AdamsPreLpLosses = LpLosses

        elif mode=="post" and mthd_name=="Adam":
            print('Finished Adams Post Training')
            self.AdamsPstTrnLambdas = TrnLambdas
            self.AdamsPstTrnFvus = TrnFvus
            self._PstTrnAdamsEpochs = cmpltd_epochs
            self._PstTrnAdamsLrnRt = lrn_rt
            self._PstTrnAdamsAlpha = alpha
            self.AdamsPstTrnLoss = TrnLosses
            self.AdamsPstTstLoss = TstLosses
            self.AdamsPstTrnDataLoss = TrnDataLosses
            self.AdamsPstTstDataLoss = TstDataLosses
            self.AdamsPstTrnEqLoss = TrnEqLosses
            self.AdamsPstTstEqLoss = TstEqLosses
            self.AdamsPstLpLosses = LpLosses

        elif mode.lower()=="pre" and mthd_name=="LBFGS":
            self._LBFGsPreTrnEpochs = cmpltd_epochs
            self._LBFGsPreTrnLrnRt = lrn_rt
            self._LBFGsPreTrnAlpha = alpha
            self._LBFGsPreTrnGamma = gamma
            self.LbfgsPreTrnLoss = TrnLosses
            self.LbfgsPreTstLoss = TstLosses
            self.LbfgsPreTrnDataLoss = TrnDataLosses
            self.LbfgsPreTstDataLoss = TstDataLosses
            self.LbfgsPreTrnEqLoss = TrnEqLosses
            self.LbfgsPreTstEqLoss = TstEqLosses
            self.LbfgsPreTrnLambdas = LpLosses
            self.LbfgsPreTrnLambdas = TrnLambdas
            self.LbfgsPreTrnFvus = TrnFvus

        elif mode.lower()=="post" and mthd_name=="LBFGS":
            self._PstTrnLBFGsEpochs = cmpltd_epochs
            self._PstTrnLBFGsLrnRt = lrn_rt
            self._PstTrnLBFGsAlpha = alpha
            self.LbfgsPstTrnLoss = TrnLosses
            self.LbfgsPstTstLoss = TstLosses
            self.LbfgsPstTrnDataLoss = TrnDataLosses
            self.LbfgsPstTstDataLoss = TstDataLosses
            self.LbfgsPstTrnEqLoss = TrnEqLosses
            self.LbfgsPstTstEqLoss = TstEqLosses
            self.LbfgsPstLpLosses = LpLosses
            self.LbfgsPstTrnLambdas = TrnLambdas
            self.LbfgsPstTrnFvus = TrnFvus

        # elif mode=="ado":
        #     # do some stuff here or in the ADO method?

        return (TrnLosses, TstLosses, 
                TrnDataLosses, TstDataLosses, 
                TrnEqLosses, TstEqLosses, 
                LpLosses)

    def ADO_Training(self, iters:int, optim_alpha_grwth_methd:str="poly",
                LBFGS:bool=False, lbfgs_epochs:int=150,
                Early_Term:bool=False,
                 **kwargs):
        if not isinstance(optim_alpha_grwth_methd, str):
            raise TypeError(f"\'optim_alpha_grwth_methd\' needs to be a string")
        if optim_alpha_grwth_methd not in ["poly", "exp", "log"]:
            raise ValueError(f"\'optim_alpha_grwth_methd\' can only be poly, exp, or log")

        alph_ins = torch.linspace(0, 1.0, iters+1, device=self.device)[1:]
        pre_trn_alpha = torch.tensor(self._AdamsPreTrnAlpha, device=self.device, dtype=self.data_type)
        if optim_alpha_grwth_methd=="exp":
            alpha_func = lambda x: pre_trn_alpha * torch.exp(-1*x * torch.log(pre_trn_alpha))
        elif optim_alpha_grwth_methd=="log":
            alpha_func = lambda x: (1 - pre_trn_alpha) * torch.log( pre_trn_alpha /( (1-pre_trn_alpha)*x + pre_trn_alpha ) ) / torch.log(pre_trn_alpha) + pre_trn_alpha
        else:
            if "p" in kwargs.keys():
                p = np.abs(kwargs['p']).item() if kwargs['p']!=0 else 1.0
            else:
                p = 1.0
            alpha_func = lambda x: (1 - pre_trn_alpha)*(x ** p) + pre_trn_alpha
        
        train_alphas = alpha_func(alph_ins)
        if "train_alphas" in kwargs.keys():
            train_alphas = kwargs["train_alphas"]

        ADO_iters = iters
        # ADO_alphas = train_alphas.cpu().numpy()

        train_inputs = torch.from_numpy(self.data_dict['Train_Inputs']).to(device=self.device, dtype=self.data_type)
        # train_targets = torch.from_numpy(self.data_dict['Train_Targets']).to(device=self.device, dtype=self.data_type)
        test_inputs = torch.from_numpy(self.data_dict['Test_Inputs']).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        # test_targets = torch.from_numpy(self.data_dict['Test_Targets']).to(device=self.device, dtype=self.data_type)
        self.nDpnts = train_inputs.size(0)

        # store all the adams optim losses over the epochs and the ado iterations
        if not LBFGS:
            lbfgs_epochs = 0
        n_trgts = self.lmbda.data.size(-1)

        adams_epchs = []
        tot_trn_loss = []
        tot_tst_loss = []
        data_loss = []
        tst_data_loss = []
        pde_loss = []
        tst_pde_loss = []
        lp_loss = []
        used_trn_alphas = []

        iter_lmbdas = []
        iter_fvus = []

        iter_lmbdas.append(self.lmbda.data ) 
        iter_fvus.append(self.FVU_Calc(lib_ceofs=self.lmbda.data) ) 

        early_rfe_term = None
        if "Early_RFE_Term"in kwargs.keys():
            early_rfe_term = kwargs["Early_RFE_Term"]
            # erly_trm_lmbda = torch.zeros(size=(iters, *self.lmbda.data.shape), device=self.device, dtype=self.data_type)
            erly_trm_lmbda = []
            # erly_fvu = torch.zeros(size=(iters, n_trgts), device=self.device, dtype=self.data_type)
            erly_fvu = []
            rfe = RFE(alpha=torch.zeros(size=(1,),device=self.device, dtype=self.data_type), normalize=True, annealing_factor=1,)

        
        for k in range(ADO_iters):
            iter_lmbdas.append(torch.zeros(size= self.lmbda.data.size(), device=self.device, dtype=self.data_type))
            nzs = [ [i.item() for i in self.lmbda.data[:,j].nonzero()] for j in range(n_trgts)]
            
            np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
            colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
            # if not colpnts.requires_grad:
            #     colpnts.requires_grad_(True)

            col_preds = self.net(colpnts)
            t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords).detach()
            lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts).detach()

            for i in range(n_trgts):
                # iter_lmbdas[k+1, nzs[i], i:i+1] = self.sprs_slvr.solve(A=lib_evals[:,nzs[i]], b=t_prtls[:,i:i+1])
                iter_lmbdas[-1][nzs[i], i:i+1] = self.sprs_slvr.solve(A=lib_evals[:,nzs[i]], b=t_prtls[:,i:i+1])

            self.sprs_slvr.cmplted_ADO_iter+=1
            self.lmbda.data = iter_lmbdas[k+1]
            # iter_fvus[k+1] = self.FVU_Calc(lib_ceofs=self.lmbda.data)
            iter_fvus.append(self.FVU_Calc(lib_ceofs=self.lmbda.data))

            print(f"lmbda at ADO iter {k} is ...")
            # print(self.lmbda.data) 
            self.Learned_EQ(output=True, sup_zeros=True,)

            if early_rfe_term:
                erly_trm_lmbda.append(torch.zeros(size= self.lmbda.data.size(), device=self.device, dtype=self.data_type))
                for i in range(n_trgts):
                    # erly_trm_lmbda[k, nzs[i], i:i+1] = rfe.solve(A=lib_evals[:,nzs[i]], b=t_prtls[:,i:i+1])
                    erly_trm_lmbda[-1][nzs[i], i:i+1] = rfe.solve(A=lib_evals[:,nzs[i]], b=t_prtls[:,i:i+1])
                # erly_fvu[k] = self.FVU_Calc(lib_ceofs=erly_trm_lmbda[k])
                erly_fvu.append( self.FVU_Calc(lib_ceofs=erly_trm_lmbda[k]) )

            if Early_Term:
                # Terminate this optim step if the selection of the library terms is the same as the last iteration.
                if nzs == [ [i.item() for i in self.lmbda.data[:,j].nonzero()] for j in range(n_trgts)]:
                    print(f"Early Terminating This Optimization step as desired since the selected library \n" 
                          f"terms have not changed from the last iteration ({k-1}) to this one ({k}).")
                    break

            adams_tple = self.AdamsOptimTraining(mode="ado", alpha=train_alphas[k].item(), gamma=0.0, 
                                    max_epochs=self._AdamsPreTrnEpochs if self._AdamsPreTrnEpochs is not None else 1000, 
                                    lrn_rt=self._AdamsPreTrnLrnRt, lp_ord=1.0,)
            tot_trn_loss.append(adams_tple[0])      # TrnLosses
            tot_tst_loss.append(adams_tple[1])      # TstLosses
            data_loss.append(adams_tple[2])         # TrnDataLosses
            tst_data_loss.append(adams_tple[3])     # TstDataLosses
            pde_loss.append(adams_tple[4])          # TrnEqLosses
            tst_pde_loss.append(adams_tple[5])      # TstEqLosses
            lp_loss.append(adams_tple[6])           # LpLosses
            adams_epchs.append(len(adams_tple[6]))
            used_trn_alphas.append(train_alphas[k].item())

            # if Early_Term:
            #     # Terminate this optim step if the selection of the library terms is the same as the last iteration.
            #     if nzs == [ [i.item() for i in self.lmbda.data[:,j].nonzero()] for j in range(n_trgts)]:
            #         print(f"Early Terminating This Optimization step as desired since the selected library \n" 
            #               f"terms have not changed from the last iteration ({k-1}) to this one ({k}).")
            #         break

        ##################################################################################################################################################
        if ADO_iters==0:
            k=0
        self._ADO_iters = k+1
        
        nzs = [ [i.item() for i in self.lmbda.data[:,j].nonzero()] for j in range(n_trgts)]
        iter_lmbdas.append(torch.zeros(size= self.lmbda.data.size(), device=self.device, dtype=self.data_type))

        model_losses = []
        model_complxtys = []
        model_scores = []

        # get the evaluation of the library functions over the collocation points for later. 
        np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        col_preds = self.net(colpnts)
        t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords).detach()
        lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts).detach()

        tst_col_preds = self.net(test_inputs)
        tst_t_prtls = Nth_temporal_prtls(values=tst_col_preds, pts=test_inputs, orders=self.tmprl_ords).detach_()
        tst_lib_evals = self.lib_func.Calc(network=self.net, inpts=test_inputs).detach()
        # need the complexity of each library term for choosing the best equation bellow. 
        lib_complexities = self.lib_func.Get_Lib_Complexities()
        n_lib_trms = self.lmbda.shape[0]
        # need to store which equations were considered throughout the optimization steps and here. 
        eqs_ids = [[np.arange(n_lib_trms)] for _ in range(n_trgts)]
        pre_ids = [np.arange(n_lib_trms) for _ in range(n_trgts)]
        eq_complexities = [[lib_complexities.sum().item() + n_lib_trms - 1] for _ in range(n_trgts)]
        slctd_eqs_indices = []

        regress_alphas = self.sprs_slvr.alpha

        for q in range(n_trgts):
            # recall each of the n targeted equation might have a different number of functions selected after optimization step. 
            n_act_funcs = len(nzs[q])
            # since the library should be greated culled after the K optim step, just do a greedy sparse regression to 
            # consider remaining models until the Null/Empty model. These are the extra lambdas.
            extra_lmbdas = Specified_Optim_Regres(A=lib_evals, b=t_prtls, activ_ids=nzs[q], alpha=regress_alphas).cpu().numpy()
            # last/remaining models
            lst_models = np.reshape(extra_lmbdas, shape=(n_act_funcs*n_lib_trms + n_lib_trms,), order="F").reshape((n_act_funcs+1,n_lib_trms,1), order="C")
            # all models considered throughout tryining/optim. step. 
            # ado_models = np.concat((iter_lmbdas[:-1, :,q:q+1].detach().cpu().numpy(), lst_models), axis=0)
            ado_models = np.concat(([lmb[:,q:q+1].detach().cpu().numpy() for lmb in iter_lmbdas[:-1]], lst_models), axis=0)
            # get the complexity of each distinct model as it is possible to have some combination of the lib. funcs repeated 
            # from one optim step to the next.
            for lm in ado_models:
                idx = np.nonzero(lm[:,0])[0]    
                if idx.size==pre_ids[q].size:
                    continue
                pre_ids[q] = idx
                eqs_ids[q].append(idx)
                eq_complexities[q].append(lib_complexities[idx].sum().item() + idx.size - 1)
            # number of distinct library func. combinations. 
            n_eqs = len(eqs_ids[q])
            eq_losses = torch.empty(size=(n_eqs,), device=self.device, dtype=self.data_type)
            for j in range(n_eqs):
                # Now get the equation coefficients for the each model and the loss on the test library. 
                # eq_coefs = torch.linalg.lstsq(lib_evals[:,eqs_ids[q][j]],t_prtls[:,q:q+1])[0]
                eq_coefs = solve_svd(lib_evals[:,eqs_ids[q][j]],t_prtls[:,q:q+1], alphas=regress_alphas)
                eq_losses[j] = torch.mean( (tst_lib_evals[:,eqs_ids[q][j]]@eq_coefs -tst_t_prtls[:,q:q+1] ) **2 )
            
            eq_lsses = eq_losses.cpu().numpy()
            model_complexities = np.asarray(eq_complexities[q])
            # flippig the order of the matrix valyes since want the complexity to increase as we go down across (-->) matrix
            # so model loss will then decrease as we get more and more complex equation (reading complexity vs loss curver right left. )
            flppd_eq_losses = np.flipud(eq_lsses)
            model_losses.append(flppd_eq_losses)
            flppd_model_cmplx = np.flipud(model_complexities)
            model_complxtys.append(flppd_model_cmplx)
            # if this is not done, the complexity of the null/empty model is -1 : so make it 0 as no model had no complexity. 
            # flppd_model_cmplx[0] = np.int64(0)
            scores = -np.log(flppd_eq_losses[1:]/flppd_eq_losses[:-1]) / (flppd_model_cmplx[1:] - flppd_model_cmplx[:-1])
            model_scores.append(scores)
            max_scr_idx = scores.argmax()+1
            slctd_eqs_indices.append(eqs_ids[q][len(eqs_ids[q]) - max_scr_idx-1])

        del pre_ids
        
        for i in range(n_trgts):
            # iter_lmbdas[-1, nzs[i], i:i+1] = self.sprs_slvr.solve(A=lib_evals[:,nzs[i]], b=t_prtls[:,i:i+1])
            iter_lmbdas[-1][slctd_eqs_indices[i], i:i+1] = solve_svd(lib_evals[:,slctd_eqs_indices[i]],t_prtls[:,i:i+1], alphas=regress_alphas)

        ##################################################################################################################################################


        self.lmbda.data = iter_lmbdas[-1]
        # iter_fvus[-1] = self.FVU_Calc(lib_ceofs=self.lmbda.data)
        iter_fvus.append( self.FVU_Calc(lib_ceofs=self.lmbda.data) )
        
        if early_rfe_term:
            self.earl_lmbds = torch.stack(erly_trm_lmbda, dim=0).detach().cpu().numpy()
            # self.earl_fvus = erly_fvu.detach().cpu().numpy()
            self.earl_fvus = torch.vstack(erly_fvu).detach().cpu().numpy()
        self._ADO_lambdas = torch.stack(iter_lmbdas, dim=0).detach().cpu().numpy()
        # self._ADO_FVUs = iter_fvus.cpu().numpy()
        self._ADO_FVUs = torch.vstack(iter_fvus).cpu().numpy()
        
        # self._ADO_alphas = train_alphas
        self._ADO_alphas = np.array(used_trn_alphas)

        self._ADO_epchs = np.array(adams_epchs) + lbfgs_epochs

        self.sprs_crs_val_folds = self.sprs_slvr.Kfolds

        self._ADOTrnLosses = np.vstack(tot_trn_loss)
        self._ADOTstLosses = np.vstack(tot_tst_loss)
        self._ADOtrnDataLs = np.vstack(data_loss)
        self._ADOtstDataLs = np.vstack(tst_data_loss)
        self._ADOtrnColloLs = np.vstack(pde_loss)
        self._ADOtstColloLs = np.vstack(tst_pde_loss)
        self._AdoLpLosses= np.vstack(lp_loss)
        
        self._model_losses = model_losses
        self._model_complxtys = model_complxtys
        self._model_scores = model_scores
    
    def WriteResults(self, data_set_name:str, file_name:str='Results', precision:int=5, true_eq=None, errors=None, 
                     act_func=None, **kwargs)->None:
        """
        Method that appends to a file in the current working direct titled as file_name.txt the results of the model at the time of
        rutorch.nning this function. So for this method to work properly it is best to have all the model values set prior to rutorch.nning. 
        Input arguments are as follows:
            * lrnd_eq (str) - The learned equation written as a string argument. At this current moment expect this arg.
                        to just basically be given as u_t + model.Learned_EQ(). In any case what ever this string argument
                        is it will be written to the file as the learned equations
            * file_name (str) - name of the file that the results will be written/appended to. NOTE: Do not attatch a file
                        extension as the results will always be written to a .txt file for simplicity with the name as 
                        file_name.txt. Attatching a file extension will have unknown behavior at this current moment of
                        writing (03/11/2022)
            * precision (int) - The precision with wich the lambda values throughout all the learning steps will be 
                        printed in the txt file. The values are writen in exponential form and so this defines the
                        number of decimal places shown. 
            * true equation: An optional input argument. If the true PDE equation is know by the user then passing that
                        equations written as a string to this argument will result in the results file containing this 
                        true equation. If given it must be a string 
            * errors: An optional input argument. If the user knows what the true equation is and has a way of 
                        determining the error between that equation and what was learned as a float then the written
                        results will have this included on its own line. Error must be given as a float.
        TODO -  (1) Rework the learned equation method of the class to create a class variable that contains the learned equation
                or possibly have this already created and the method just updates it when it is run so that it does not have to be an
                argument to this function and can just be called using self.lrnd_eq or something like that. 
                (2) Handle the case the the ADO-RFE was run with 0 epochs or (inclusive) 0 ADO_iters
        """
        # First some like input arguments checking:
        if not isinstance(file_name, str):
            print('ERROR - The user passed value for the file_name input argument is not a string but is a {} '.format(type(file_name)))
            print('Results will instead be appened to the following file in the Current working directy - Results.txt')
            file_name = 'Results'
        if not (isinstance(true_eq, str)  or true_eq is None):
            print('ERROR - The user passed value for the true_eq input argument is not a string nor None but is a {} '.format(type(true_eq)))
            print('To handle will not print anything about true results')
            true_eq = None
        if not (isinstance(errors, list)  or errors is None):
            print('ERROR - The user passed input for the errors input argument is not a list (of errors) nor None but is a {} '.format(type(errors)))
            print('To handle will not print anything about true results')
            errors = None
        if errors==None: errors=['~~~~', '~~~~~']

        dvc = self.device

        library_names = self.lib_func.get_library_names(net_out_func_names=self.ntwk_out_names)
        n_lib = len(library_names)
        n_trgts = self.lmbda.data.size(1)
        lrnd_eq = self.Learned_EQ(output=False,)
        # get someway to identify at the time that this function was run what training methods have been completed. 
        # NOTE: MAYBE A BETTER IDEA TO HAVE A PRIVATE CLASS LIST IS ONLY UPDATED WHEN A TRAINING METHOD IS RUN TO IDENTIFY WHICH TRAINING METHODS WERE USED AND THE ORDER
        # as I am working under the assumption that the things have been run in the poper order of pretrainings, ADO then post trainings
        trn_ids = np.array([0 if itm is None else 1 for itm in [self.AdamsPreTrnLambdas, self.LbfgsPreTrnLambdas, self._ADO_lambdas, self.AdamsPstTrnLambdas, self.LbfgsPstTrnLambdas]])
        if trn_ids.sum()==0:
            raise RuntimeError("Training has not be done in any form for this learning model")
        plc_hldr = np.empty(shape=self.lmbda.shape)
        FVU_plc_hldr = np.empty(shape=(n_trgts,),)
        headings = ['Lib.Terms', 'Init.']
        if trn_ids[0] == 1: # Adams pretraining was done 
            # print('Adams Pre-Training was done to get these results')
            headings.append('Adams PrTrn')
            plc_hldr = np.concatenate((plc_hldr, self.AdamsPreTrnLambdas.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
            FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.AdamsPreTrnFvus.flatten()), axis=0)
        if trn_ids[1] == 1: # LBFGS pretraining was done 
            # print('LBFGS Pre-Training was done to get these results')
            headings.append('LFBGS PrTrn')
            if trn_ids[:1].sum()==0: # IF this is true than this traing method/step was the first one done.
                plc_hldr = np.concatenate((plc_hldr, self.LbfgsPreTrnLambdas.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.LbfgsPreTrnFvus.flatten()), axis=0)
            else: 
                plc_hldr = np.concatenate((plc_hldr, self.LbfgsPreTrnLambdas[1]), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.LbfgsPreTrnFvus[1]), axis=0)
        if trn_ids[2] == 1: # ADO was done 
            # print('ADO Training was done to get these results')
            for i in range(self._ADO_iters+1):
                headings.append('ADO '+str(i))
            # if self._ADO_iters==0: self._ADO_iters=None
            if trn_ids[:2].sum()==0: # IF this is true than this traing method/step was the first one done.
                plc_hldr = np.concatenate((plc_hldr, self._ADO_lambdas.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self._ADO_FVUs.flatten()), axis=0)
            else: 
                plc_hldr = np.concatenate((plc_hldr, self._ADO_lambdas[1:].reshape((-1,1),order='C').reshape((n_lib,self._ADO_iters+1),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self._ADO_FVUs[1:].flatten()), axis=0)
        if trn_ids[3] == 1: # Adams post training was done
            # print('Adams Post Training was done to get these results')
            headings.append('Adams PostTrn')
            if trn_ids[:3].sum()==0: # IF this is true than this traing method/step was the first one done.
                plc_hldr = np.concatenate((plc_hldr, self.AdamsPstTrnLambdas.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.AdamsPstTrnFvus.flatten()), axis=0)
            else: 
                plc_hldr = np.concatenate((plc_hldr, self.AdamsPstTrnLambdas[1]), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.AdamsPstTrnFvus[1]), axis=0)
        if trn_ids[4] == 1: # Adams post training was done 
            # print('LBFGS Post Training was done to get these results')
            headings.append('LBFGS PostTrn')
            if trn_ids[:4].sum()==0: # IF this is true than this traing method/step was the first one done.
                plc_hldr = np.concatenate((plc_hldr, self.LbfgsPstTrnLambdas.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.LbfgsPstTrnFvus.flatten()), axis=0)
            else: 
                plc_hldr = np.concatenate((plc_hldr, self.LbfgsPstTrnLambdas[1]), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.LbfgsPstTrnFvus[1]), axis=0)

        # Now place holder will hold all the lambas parameter values/vectors throughout all the training 
        # step 
        tbl_vals = np.concatenate((plc_hldr[:,n_trgts:], FVU_plc_hldr[n_trgts:][np.newaxis,:]), axis=0)
        
        # precision = 5
        lst = list(kwargs.keys())
        if 'jobID' in kwargs.keys(): jobID = kwargs['jobID']
        else: jobID = None
        if 'jobVer'in kwargs.keys(): jobVer = kwargs['jobVer']
        else: jobVer = None
        if 'node' in kwargs.keys(): node = kwargs['node']
        else: node = None
        if 'run_time'in kwargs.keys(): run_time = kwargs['run_time']
        else: run_time = 0
        if 'subsample_prcntg' in kwargs.keys(): subsample_prcntg = kwargs['subsample_prcntg']
        else: subsample_prcntg = None
        if 'noisePrcntg' in kwargs.keys(): noisePrcntg = kwargs['noisePrcntg']
        else: noisePrcntg = None
        if 'NpSeed' in kwargs.keys(): NpSeed = kwargs['NpSeed']
        else: NpSeed = None
        if 'Ns' in kwargs.keys(): Ns = kwargs['Ns']
        else: Ns = None

        # All this does is to print out in the data file the evolution of the candidate library coefficients
        # through out the whole process of learning. 
        longest = 0     # what is the longest library function name
        library_names.append('FVU values')
        for term in library_names:
            if longest<len(term):
                longest=len(term)
        if longest < len(' Adams PostTrn '):
            longest = len(' Adams PostTrn ')
        if longest%2==1: longest+=1
        # Now make the rows that form the sort of coefficient evolution table
        space = ' '
        lines = ''
        temp = '%3.{}e'.format(precision)
        n_tbl_cls = tbl_vals.shape[1]
        n_trn_stps = int(n_tbl_cls / n_trgts)
        for k in range(n_lib+1):
            lines+=library_names[k]
            lines += (longest-len(library_names[k]) )*space
            lines += " |"
            for j in range(n_trn_stps):
                for l in range(n_trgts):
                    col = n_trgts*j+l
                    if (l+1)==n_trgts:
                        if tbl_vals[k,col].item()>= 0:
                            lines = lines +  " "+ temp % (tbl_vals[k,col].item()) + " | "
                        else:
                            lines = lines +  temp % (tbl_vals[k,col].item()) + " | "
                    else:    
                        if tbl_vals[k,col].item()>= 0:
                            lines = lines +  " "+ temp % (tbl_vals[k,col].item()) + " , "
                        else:
                            lines = lines +  temp % (tbl_vals[k,col].item()) + " , "
            # for j in range(n_tbl_cls):
            #     if tbl_vals[k,j].item()>= 0:
            #         lines = lines +  " "+ temp % (tbl_vals[k,j].item()) + " | "
            #     else:
            #         lines = lines +  temp % (tbl_vals[k,j].item()) + " | "
            lines+='\n'
        n = lines.find('\n')
        # Now make the header or the top row of the table that contains the columns names
        hline=''        # horizontal line of ------
        header = ''     # the header and or top row containing the column names
        breaks=[0]
        starts = []
        for k, i in enumerate(lines[:n]):
            if i=='|':
                breaks.append(k)
        for k in range(0, len(breaks)-1): 
            mid = int((breaks[k]+breaks[k+1])/2)
            starts.append(mid-int(len(headings[k])/2))
        i = 0
        j = 0
        k = 1
        while i<n:
            if i==starts[j]:
                header+=headings[j]
                hline+=len(headings[j])*'-'
                i+=len(headings[j])
                j+=1
                if j==len(headings):
                    j=0
            elif i==breaks[k]:
                header+='|'
                hline+='-'
                k+=1
                i+=1
                if k==len(breaks):
                    k=1
            else:
                header+=space
                hline+='-'
                i+=1
        # final = header+'\n'+hline+'\n'+lines+'\n'+ '~'*150+'\n'
        # final = header+'\n'+hline+'\n'+lines+'\n'
        idx = lines.find('FVU values')
        final = header+'\n'+hline+'\n'+lines[:idx]+hline+'\n'+lines[idx:]+'\n'
        if self.earl_lmbds is not None:
            final += 'Early termination learned equations (ie. EQs learned if ADO was terminated earlier with original RFE)\n'
            for i, lmbda in enumerate(self.earl_lmbds):
                final += f"ADO iter. {i}:\n"
                eqs = self.Learned_EQ(coefs=lmbda, output=False,).split('\n')
                for j, eqnt in enumerate(eqs):
                    # final += f"(FVU = {self.earl_fvus[i, j].round(8)}) " + eqnt + '\n'
                    final +=  "(FVU = " + temp % (self.earl_fvus[i, j].item()) + ") " + eqnt + '\n'
                
        final +='~'*150+'\n'
        # Write the results to the txt data file
        with open(file_name+'Results.txt', 'a', encoding='UTF-8', errors='replace') as file:
            file.write('Results and Hyperparamter Values for '+data_set_name+' data set using RFE - Job Num = {}, Ver. = {} ran on {} \n'.format(jobID, jobVer, node))                                                                                  # line 1
            file.write('Network Nonlinear activation function  = {} \n'.format(act_func))                                                                                                                                                               # line 2 
            file.write('Device that the results were obtained on - {}\n'.format(dvc))                                                                                                                                                                   # line 3
            file.write('Total wall-clock run time  = {} seconds = {} minutes\n'.format(run_time, (run_time)/60))                                                                                                                                        # line 4
            file.write('Various Hyperparater values and trainining settings used to get these results:\n')                                                                                                                                              # line 5
            file.write('Num of training data points = {}, data sampling percentage {}%, Num of collocation points = {}, noise percentage {}%\n'.format(self.nDpnts, subsample_prcntg*100, self.N_col_pnts, noisePrcntg))                                # line 6
            file.write('Training Batch size = {}, Num training spatial points = {}, Num testing spatial points = {}, Numpy RNG seed/entropy value = {}\n'.format(self.trn_batch_size, Ns[0], Ns[1], NpSeed))                                            # line 7
            file.write('Adams Pretraining Stuff - epochs = {}, init. learning rate = {}, alpha loss value = {}, beta loss value = {}\n'.format(self._AdamsPreTrnEpochs, self._AdamsPreTrnLrnRt, self._AdamsPreTrnAlpha, self._AdmasPreTrnGamma))        # line 8
            file.write('LBFGS Pretraining Stuff - epochs = {}, init. learning rate = {}, alpha loss value = {}, beta loss value = {}\n'.format(self._LBFGsPreTrnEpochs, self._LBFGsPreTrnLrnRt, self._LBFGsPreTrnAlpha, self._LBFGsPreTrnGamma))        # line 9
            file.write('ADO training stuff - iterations = {}, sparse slvr X val. folds = {}, training epochs = {}, Adams alpha hyper parameter values = {}\n'.format(self._ADO_iters, self.sprs_crs_val_folds, self._ADO_epchs, self._ADO_alphas))      # line 10
            file.write('Post ADO Adams training Stuff - epochs = {}, init. learning rate = {}, alpha loss value = {}\n'.format(self._PstTrnAdamsEpochs, self._PstTrnAdamsLrnRt, self._PstTrnAdamsAlpha))                                                # line 11
            file.write('Post ADO LBFGS training Stuff - epochs = {}, init. learning rate = {}, alpha loss value = {}\n'.format(self._PstTrnLBFGsEpochs, self._PstTrnLBFGsLrnRt, self._PstTrnLBFGsAlpha))                                                # line 12
            file.write('LEARNED Equation is - \n '+lrnd_eq+'\n')                                                                                                                                                                                        # line 13 & 14
            if not (true_eq is None):
                file.write('TRUE Equation is - \n '+true_eq+'\n')                                                                                                                                                                                       # line 15 & 16
            else:
                file.write('TRUE Equation is - \n  \n')                                                                                                                                                                                                 # line 15 & 16
            file.write('Individual Coefficient (Relative) Errors:\n {}\n'.format(errors[1]))                                                                                                                                                            # lines 17 and 18 
            file.write('Mean Coefficient (Relative) Error:\n {}\n'.format(errors[0]))                                                                                                                                                                   # lines 19 and 20 
            file.write('Evolution of candidate library lambda/coefficient values throughout learning\n')                                                                                                                                                # line 21
            file.write(final)                                                                                                                                                                                                                           # line 22 (head), 23(-----) and as 
                                                                                                                                                                                                                                                        # 24-[24+(num. lib terms)-1 + 2]
        return None
     
    def TrainingLossPlots(self, dataset_name:str, file_name:str, plot_title:str='PreTrain and ADO Training Losses', 
                          font_size:float=10)->None:
        """
            Create a plot displaying the model's training losses through both pre-training and ADO-RFE training. The losses are
            plottd on a single figure as one single continuous line over all the training epochs such that losses through the 
            pretraining are first drawn in blue then the losses incurred over the ADO steps in differing colors one after the next.
            The x-axis label is super epochs since it is all the pretraining epochs may differ from the ADO training epochs but 
            the total number of epoch that that the model trained under is the summation of the number of pretrained epoch and 
            the ADO-RFE epochs. The figure contain the plot of the training losses will be saved as a png file to a folder titled
            LossFigures/dataset_name where dataset_name the current value of model.dataset_name (see set_Dataset_Name() method).
            Should these folder not be created or there is a problem changing to them either, the folders will be created or the 
            figure/plot will be saved tothe last folder/directory that we were in before trying to change had some error. 
            The input arguements are as follows:
                * file_name (str) - The name of the file under which the plot of the training losses is saved. It should not 
                            include .png or anything like it. 
                * plot_title (str) - Title of the plot
                * font_size (float) - fontsize of the plot title's text
                * show_fig (bool) - Whether or not to display the figure prior to saving it to disk
        """
        # NOTE - just some like input argument cheching. Eventually will need to do something better and stronger than this
        # in the future prior to release
        if not isinstance(file_name, str):
            print('ERROR!!! - The file_name argument needs to be a string object not a {} object'.format(type(file_name)))
            print('Will handle this by using the file_name = ErrorNamedPlot.png')
            file_name = 'ErrorNamedPlot'
        if not isinstance(plot_title, str):
            print('ERROR!!! - The plot_title argument needs to be a string object not a {} object'.format(type(plot_title)))
            print('Will handle this by using the plot_title = Prtrn and ADO Training Losses')
            plot_title = 'Prtrn and ADO Training Losses'
        if not isinstance(font_size, (float, int)):
            print('ERROR!!! - The font_size argument needs to be a float/int object not a {} object'.format(type(font_size)))
            print('Will handle this by using the font_size = 10')
            font_size = 10
        
        # if not isinstance(show_fig, bool):
        #     print('ERROR!!! - The show_fig argument needs to be a string object not a {} object'.format(type(show_fig)))
        #     print('Will handle this by using the show_fig = False')
        #     show_fig = False

        sv_dir = os.path.join("LossFigures", dataset_name)
        os.makedirs(name=sv_dir, exist_ok=True)
        
        n_trgts = self.lmbda.data.shape[1]       
        
        # Combine the losses from the pretraining and ado-training epochs for Adams
        if isinstance(self.AdamsPreTrnLoss,np.ndarray):
            adams_pre_trn_epchs = self.AdamsPreTrnLoss.shape[0]
            AdamsPreTrnLoss = self.AdamsPreTrnLoss
            AdamsPreTstLoss = self.AdamsPreTstLoss
            AdamsPreTrnDataLoss = self.AdamsPreTrnDataLoss
            AdamsPreTstDataLoss = self.AdamsPreTstDataLoss
            AdamsPreTrnColloLoss = self.AdamsPreTrnEqLoss
            AdamsPreTstColloLoss = self.AdamsPreTstEqLoss
            AdamsPreLpLosses = self.AdamsPreLpLosses
        else:
            adams_pre_trn_epchs = 0
            AdamsPreTrnLoss = np.zeros((0,n_trgts))
            AdamsPreTstLoss = np.zeros((0,n_trgts))
            AdamsPreTrnDataLoss = np.zeros((0,n_trgts))
            AdamsPreTstDataLoss = np.zeros((0,n_trgts))
            AdamsPreTrnColloLoss = np.zeros((0,n_trgts))
            AdamsPreTstColloLoss = np.zeros((0,n_trgts))
            AdamsPreLpLosses = np.zeros((0,n_trgts))
           
        if isinstance(self.LbfgsPreTrnLoss,np.ndarray):
            LBFGS_pre_trn_epchs = self.LbfgsPreTrnLoss.shape[0]
            LBFGsPreTrnLoss = self.LbfgsPreTrnLoss
            LBFGsPreTstLoss = self.LbfgsPreTstLoss
            LBFGsPreTrnDataLoss = self.LbfgsPreTrnDataLoss
            LBFGsPreTstDataLoss = self.LbfgsPreTstDataLoss
            LBFGsPreTrnColloLoss = self.LbfgsPreTrnEqLoss
            LBFGsPreTstColloLoss = self.LbfgsPreTstEqLoss
            LbfgsPreLpLosses = self.LbfgsPreLpLosses
        else:
            LBFGS_pre_trn_epchs = 0
            LBFGsPreTrnLoss = np.zeros((0,n_trgts))
            LBFGsPreTstLoss = np.zeros((0,n_trgts))
            LBFGsPreTrnDataLoss = np.zeros((0,n_trgts))
            LBFGsPreTstDataLoss = np.zeros((0,n_trgts))
            LBFGsPreTrnColloLoss = np.zeros((0,n_trgts))
            LBFGsPreTstColloLoss = np.zeros((0,n_trgts))
            LbfgsPreLpLosses = np.zeros((0,n_trgts))

        if isinstance(self._ADOTrnLosses, np.ndarray):
            ADO_epchs = self._ADO_epchs.tolist()
            ADO_iters = self._ADO_epchs.shape[-1]
            ADOTrnlosses = self._ADOTrnLosses
            ADOTstlosses = self._ADOTstLosses
            ADOtrnDataLs = self._ADOtrnDataLs
            ADOtstDataLs = self._ADOtstDataLs
            ADOtrnColloLs = self._ADOtrnColloLs
            ADOtstColloLs = self._ADOtstColloLs
            AdoLpLosses = self._AdoLpLosses
        else:
            ADO_iters, ADO_epchs = 0, [0]
            ADOTrnlosses = np.zeros((0,n_trgts))
            ADOTstlosses = np.zeros((0,n_trgts))
            ADOtrnDataLs = np.zeros((0,n_trgts))
            ADOtstDataLs = np.zeros((0,n_trgts))
            ADOtrnColloLs = np.zeros((0,n_trgts))
            ADOtstColloLs = np.zeros((0,n_trgts))
            AdoLpLosses = np.zeros((0,n_trgts))

        if isinstance(self.AdamsPstTrnLoss,np.ndarray):
            adams_post_trn_epchs = self.AdamsPstTrnLoss.shape[0]
            AdamsPstTrnLoss = self.AdamsPstTrnLoss
            AdamsPstTstLoss = self.AdamsPstTstLoss
            AdamsPstTrnDataLoss = self.AdamsPstTrnDataLoss
            AdamsPstTstDataLoss = self.AdamsPstTstDataLoss
            AdamsPstTrnColloLoss = self.AdamsPstTrnEqLoss
            AdamsPstTstColloLoss = self.AdamsPstTstEqLoss
            AdamsPstLpLosses = self.AdamsPstLpLosses
        else:
            adams_post_trn_epchs = 0
            AdamsPstTrnLoss = np.zeros((0,n_trgts))
            AdamsPstTstLoss = np.zeros((0,n_trgts))
            AdamsPstTrnDataLoss = np.zeros((0,n_trgts))
            AdamsPstTstDataLoss = np.zeros((0,n_trgts))
            AdamsPstTrnColloLoss = np.zeros((0,n_trgts))
            AdamsPstTstColloLoss = np.zeros((0,n_trgts))
            AdamsPstLpLosses = np.zeros((0,n_trgts))
            
        if isinstance(self.LbfgsPstTrnLoss,np.ndarray):
            LBFGS_post_trn_epchs = self.LbfgsPstTrnLoss.shape[0]
            LBFGsPstTrnLoss = self.LbfgsPstTrnLoss
            LBFGsPstTstLoss = self.LbfgsPstTstLoss
            LBFGsPstTrnDataLoss = self.LbfgsPstTrnDataLoss
            LBFGsPstTstDataLoss = self.LbfgsPstTstDataLoss
            LBFGsPstTrnColloLoss = self.LbfgsPstTrnEqLoss
            LBFGsPstTstColloLoss = self.LbfgsPstTstEqLoss
            LbfgsPstLpLosses = self.LbfgsPstLpLosses
        else:
            LBFGS_post_trn_epchs = 0
            LBFGsPstTrnLoss = np.zeros((0,n_trgts))
            LBFGsPstTstLoss = np.zeros((0,n_trgts))
            LBFGsPstTrnDataLoss = np.zeros((0,n_trgts))
            LBFGsPstTstDataLoss = np.zeros((0,n_trgts))
            LBFGsPstTrnColloLoss = np.zeros((0,n_trgts))
            LBFGsPstTstColloLoss = np.zeros((0,n_trgts))
            LbfgsPstLpLosses = np.zeros((0,n_trgts))

        # Create the array to contian all the training losses
        Cmbd_trn_losses = np.vstack((AdamsPreTrnLoss, LBFGsPreTrnLoss, ADOTrnlosses, AdamsPstTrnLoss, LBFGsPstTrnLoss))
        Cmbd_tst_losses = np.vstack((AdamsPreTstLoss, LBFGsPreTstLoss, ADOTstlosses, AdamsPstTstLoss, LBFGsPstTstLoss))
        cmbdTrn_data_losses = np.vstack((AdamsPreTrnDataLoss, LBFGsPreTrnDataLoss, ADOtrnDataLs, AdamsPstTrnDataLoss, LBFGsPstTrnDataLoss))
        cmbdTst_data_losses = np.vstack((AdamsPreTstDataLoss, LBFGsPreTstDataLoss, ADOtstDataLs, AdamsPstTstDataLoss, LBFGsPstTstDataLoss))
        cmbdTrn_collo_losses = np.vstack((AdamsPreTrnColloLoss, LBFGsPreTrnColloLoss, ADOtrnColloLs, AdamsPstTrnColloLoss, LBFGsPstTrnColloLoss))
        cmbdTst_collo_losses = np.vstack((AdamsPreTstColloLoss, LBFGsPreTstColloLoss, ADOtstColloLs, AdamsPstTstColloLoss, LBFGsPstTstColloLoss))
        cmbd_lp_losses = np.vstack((AdamsPreLpLosses, LbfgsPreLpLosses, AdoLpLosses, AdamsPstLpLosses, LbfgsPstLpLosses))

        optim_trn_epochs = np.array([adams_pre_trn_epchs, LBFGS_pre_trn_epchs] + ADO_epchs + [adams_post_trn_epchs, LBFGS_post_trn_epchs])
        plt_labels = ["PreTrn Adams", "PreTrn LBFGS"] + [f"ADO {k}" for k in range(ADO_iters)] + ["PstTrn Adams", "PstTrn LBFGS"]

        if Cmbd_trn_losses.size==0:
            print('NO plot can be made as there are no losses. You have yet to run the pretraining nor ADO training IDOIT!!!!')
            return None
        
        n_eqs = self.lmbda.data.shape[-1]
        for k in range(n_eqs):
            cmbnd_fig = model_loss_plotter(type_loss="total",
                           losses_trn=Cmbd_trn_losses, losses_tst=Cmbd_tst_losses, 
                           optim_trn_epochs=optim_trn_epochs, plt_lbls=plt_labels, 
                           plt_tle=plot_title, fnt_sz=font_size)
            cmbnd_fig.savefig(fname=os.path.join(sv_dir, file_name+f"Eq{k+1}CombinedLosses.png"), format='png')

            dataLs_fig = model_loss_plotter(type_loss="data",
                           losses_trn=cmbdTrn_data_losses, losses_tst=cmbdTst_data_losses, 
                           optim_trn_epochs=optim_trn_epochs, plt_lbls=plt_labels, 
                           plt_tle=plot_title, fnt_sz=font_size)
            dataLs_fig.savefig(fname=os.path.join(sv_dir,file_name+f"Eq{k+1}DataLosses.png"), format='png')

            EqLs_fig = model_loss_plotter(type_loss="eq",
                           losses_trn=cmbdTrn_collo_losses, losses_tst=cmbdTst_collo_losses, 
                           optim_trn_epochs=optim_trn_epochs, plt_lbls=plt_labels, 
                           plt_tle=plot_title, fnt_sz=font_size)
            EqLs_fig.savefig(fname=os.path.join(sv_dir,file_name+f"Eq{k+1}DiffEqLosses.png"), format='png')

            LpLs_fig = model_loss_plotter(type_loss="lp",
                           losses_trn=cmbd_lp_losses, losses_tst=None, 
                           optim_trn_epochs=optim_trn_epochs, plt_lbls=plt_labels, 
                           plt_tle=plot_title, fnt_sz=font_size)
            LpLs_fig.savefig(fname=os.path.join(sv_dir,file_name+f"Eq{k+1}LpLosses.png"), format='png')

        return None

    def FVU_Plot(self, dataset_name:str, file_name:str, save_fig:bool=True)->None:
        """
        
        """
        library_names = self.lib_func.get_library_names(net_out_func_names=self.ntwk_out_names)
        n_lib = len(library_names)
        n_trgts = self.lmbda.data.size(1)
        # get someway to identify at the time that this function was run what training methods have been completed. 
        # NOTE: MAYBE A BETTER IDEA TO HAVE A PRIVATE CLASS LIST IS ONLY UPDATED WHEN A TRAINING METHOD IS RUN TO IDENTIFY WHICH TRAINING METHODS WERE USED AND THE ORDER
        # as I am working under the assumption that the things have been run in the poper order of pretrainings, ADO then post trainings
        trn_ids = np.array([0 if itm is None else 1 for itm in [self.AdamsPreTrnLambdas, self.LbfgsPreTrnLambdas, self._ADO_lambdas, self.AdamsPstTrnLambdas, self.LbfgsPstTrnLambdas]])
        if trn_ids.sum()==0:
            raise RuntimeError("Training has not be done in any form for this learning model")
        lmb_vals_plc_hldr = np.empty(shape=self.lmbda.data.shape)
        FVU_plc_hldr = np.empty(shape=(n_trgts,),)
        headings = ['Init.']
        if trn_ids[0] == 1: # Adams pretraining was done 
            # print('Adams Pre-Training was done to get these results')
            headings.append('Adams PrTrn')
            lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self.AdamsPreTrnLambdas.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
            FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.AdamsPreTrnFvus.flatten()), axis=0)
        if trn_ids[1] == 1: # LBFGS pretraining was done 
            # print('LBFGS Pre-Training was done to get these results')
            headings.append('LFBGS PrTrn')
            if trn_ids[:1].sum()==0: # IF this is true than this traing method/step was the first one done.
                lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self.LbfgsPreTrnLambdas.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.LbfgsPreTrnFvus.flatten()), axis=0)
            else: 
                lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self.LbfgsPreTrnLambdas[1]), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.LbfgsPreTrnFvus[1]), axis=0)
        if trn_ids[2] == 1: # ADO was done 
            # print('ADO Training was done to get these results')

            for i in range(self._ADO_iters+1):
                headings.append('ADO '+str(i))
            # if self._ADO_iters==0: self._ADO_iters=None
            if trn_ids[:2].sum()==0: # IF this is true than this traing method/step was the first one done.
                lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self._ADO_lambdas.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self._ADO_FVUs.flatten()), axis=0)
            else: 
                lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self._ADO_lambdas[1:].reshape((-1,1),order='C').reshape((n_lib,self._ADO_iters+1),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self._ADO_FVUs[1:].flatten()), axis=0)
        if trn_ids[3] == 1: # Adams post training was done
            # print('Adams Post Training was done to get these results')
            headings.append('Adams PostTrn')
            if trn_ids[:3].sum()==0: # IF this is true than this traing method/step was the first one done.
                lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self.AdamsPstTrnLambdas.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.AdamsPstTrnFvus.flatten()), axis=0)
            else: 
                lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self.AdamsPstTrnLambdas[1]), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.AdamsPstTrnFvus[1]), axis=0)
        if trn_ids[4] == 1: # Adams post training was done 
            # print('LBFGS Post Training was done to get these results')
            headings.append('LBFGS PostTrn')
            if trn_ids[:4].sum()==0: # IF this is true than this traing method/step was the first one done.
                lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self.LbfgsPstTrnLambdas.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.LbfgsPstTrnFvus.flatten()), axis=0)
            else: 
                lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self.LbfgsPstTrnLambdas[1]), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.LbfgsPstTrnFvus[1]), axis=0)

        lmb_vals = lmb_vals_plc_hldr[:,n_trgts:]
        FVU_vals = FVU_plc_hldr[n_trgts:]

        num_trn_stps = len(headings)
        eqs = [self.Learned_EQ(coefs=lmb_vals[:, n_trgts*i:n_trgts*(i+1)], output=False, sup_zeros=True, dec_rnd=5, prnt_sig_dif=3) for i in range(0, num_trn_stps)]
        idx = headings.index("ADO 0")

        # if FVU_vals[k]<=FVU_vals[k-1] and FVU_vals[k]<=FVU_vals[k+1]:
        #     verticalalignment = 'top'
        #     horizontalalignment = 'center'
        # elif FVU_vals[k]>=FVU_vals[k-1] and FVU_vals[k]>=FVU_vals[k+1]:
        #     verticalalignment = 'bottom'
        #     horizontalalignment = 'center'
        # elif FVU_vals[k-1]>=FVU_vals[k]>=FVU_vals[k+1]:
        #     verticalalignment = 'bottom'
        #     horizontalalignment = 'left'
        # elif FVU_vals[k-1]<=FVU_vals[k]<=FVU_vals[k+1]:
        #     verticalalignment = 'top'
        #     horizontalalignment = 'left'

        fig, axis = plt.subplots(nrows=1, ncols=1, )
        x_vals = np.arange(1, len(headings)+1)
        fig.set_size_inches(w=8, h=8)
        axis.semilogy(x_vals, FVU_vals, '.-', markersize=10)
        # axis.plot(np.arange(len(headings))+1, FVU_vals)
        axis.set_xlim(0, len(headings)+1)
        axis.set_xticks(ticks=x_vals, labels=headings, rotation='vertical',)
        axis.set_title('FVU Evolution')
        axis.set_ylabel('FVU')
        # align_dict = {'verticalalignment':'bottom', 'horizontalalignment':'left'}
        # axis.text(x=x_vals[idx+1], y=FVU_vals[idx+1], s=eqs[idx+1], verticalalignment='bottom', horizontalalignment='left')

        sv_dir = os.path.join("FvuEquationPlots", dataset_name)
        os.makedirs(name=sv_dir, exist_ok=True)
        
        if save_fig:
            
            fig.savefig(fname=os.path.join(sv_dir, file_name+'.png'), format='png')
           
    def Loss_Score_Complexity_Plot(self, dataset_name:str, save_dir_name:str, save_figs:bool=True)->None:
        """
        
        """
        sv_dir = os.path.join("ComplexityVersusPlots", dataset_name, save_dir_name)
        os.makedirs(name=sv_dir, exist_ok=True)

        for k in range(self.lmbda.data.shape[1]):
            sv_name = os.path.join(sv_dir, f"LearnedEq{k+1}Plot.png")
            fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), layout="constrained")
            axis.semilogy(self._model_complxtys[k], self._model_losses[k], "r.-", label="losses")
            axis.semilogy(self._model_complxtys[k][1:], self._model_scores[k], "bx-", label="score")
            axis.semilogy(self._model_complxtys[k][1:], -1*np.log(self._model_losses[k][1:] / self._model_losses[k][:-1]), "go-", label=r"$\log\left(\frac{l_{i}}{l_{i-1}}\right)$")
            axis.set_xlabel("complexitity value")
            axis.set_xlim(-2, self._model_complxtys[k].max()+1)
            axis.set_ylabel("")
            axis.legend()

            axis.set_title(f"Learned EQ. Num. {k+1} - Models Losses and Scores vs Complexity")
            if save_figs:
                fig.savefig(fname=sv_name, format='png')

        return None


class EqLearner1D(EqLearner):

    def __init__(self, 
            net, 
            Lmbda, 
            lib_func, 
            sprs_slvr, 
            data_dict, 
            tmprl_ords, 
            col_pnts_smplr = Rand_Col_Sampler(), 
            N_col_pnts = 10000, 
            ntwk_out_names:List[str]=None,
            device = torch.device('cpu'), 
            data_type = None):
        super().__init__(net, Lmbda, lib_func, sprs_slvr, data_dict, tmprl_ords, col_pnts_smplr, N_col_pnts, ntwk_out_names, device, data_type)

    def AnimatedPlot(self, dataset_name:str, spdx:float, pts:Union[np.ndarray,torch.Tensor], values:Union[np.ndarray,torch.Tensor], 
                     ani_title:str='UntiltedAnimatedPlot', fig_title:str='', state:str=''):
        """
        Function that creates an animated plot of the data that has been given in the pts
        and values tensors/arrays (values vs pts) along side the function that was learned.
        The values of the function that were learned are determined by passing the pts 
        through the torch.nn. The animation plot is saved as a .gif file.
        Input arguments are the following:
            * pts - A K by 2 tensor/numpy array that contains the spatiotemporal (x,t) points 
                    where the values in the values array have been determined at. 
                    The order of the points should be like the following:
                      (x_1,t_1), (x_2,t_1),...,(x_N,t_1), (x_1,t_2),...(x_N,t_M)
                      where x_j < x_i for j<i and t_k < t_l for k < l
                    Ideally this function would be ran by passing X_trn or
                    X_tst tensor/array found in the LearningMatData1D DataSet classes.
            * values - A tensor/numpy array that contains the evaluations of the
                    spatiotemporal points (x,t) points under some function 
                    which ideally is the one that is trying to be learned.  
                    the values array have been determined at. 
                    The ordering of the values should be like the following:
                      u(x_1,t_1), u(x_2,t_1),...,u(x_N,t_1), u(x_1,t_2),...u(x_N,t_M)
                      where x_j < x_i for j<i and t_k < t_l for k < l and u is 
                      the function to the learned
                    Ideally this function would be ran by passing u_trn or
                    u_tst tensor/array found in the LearningMatData1D DataSet classes.
            * ani_title - The title of the saved animation .gif file. 
            * fig_title - The figure title of the animated plot. Just needs to be a string
                    argument. If no titled is wanted just pass it the value of ''
            * state - Animated plots title will be state + Learned Equation/Model at t = 
                    and so this argument can be any string that you would like to replace
                    the state word or not thing at all (i.e state = '')
        """
            
        def learned_animation(i, ax, pts:np.ndarray, exact:np.ndarray, maxs:list, mins:list,t:np.array, dgt:int, trn_state:str):
            k = int(i*spdx)
            n_t = t.shape[0]
            n_x = int(pts.shape[0]/n_t)
            x = pts[k*n_x:(k+1)*n_x,0:1]
            # x.sort()
            lrn = self.net(torch.from_numpy(pts[k*n_x:(k+1)*n_x,:]).to(device=self.device, dtype=self.data_type, non_blocking=True)).cpu().detach().numpy()
            x_min, y_min = mins
            x_max, y_max = maxs
            ax.clear()
            ax.plot(x, exact[k*n_x:(k+1)*n_x:, :], color='blue', marker='o', linestyle='solid', linewidth=1, markersize=2, label='exact')
            ax.plot(x, lrn, color='red', marker='*', linestyle='solid', linewidth=1, markersize=2, label='lrnd')
            ax.set_xlabel('x')
            ax.set_xlim(left=x_min, right=x_max)
            ax.set_ylabel('u(x,t)')
            ax.set_ylim(bottom=y_min, top=y_max)
            ax.set_title(trn_state+' Learned Equation/Model at t = {}'.format(t[k].round(dgt)))
            ax.legend()

        # Something to make sure that we have the prediction and the exact values on the cpu and in numpy arrays for ploting
        if isinstance(values, torch.Tensor):vals = values.cpu().detach().numpy()
        elif isinstance(values, np.ndarray):vals = np.copy(values)
        else: raise TypeError(f"The values function argument needs to be either a torch.Tensor or numpy.ndarray object not a {type(values).__name__} object as was given!")
        # N by 2 array/tensors these need to be. 
        if isinstance(pts, torch.Tensor):X = pts.cpu().detach().numpy()
        elif isinstance(pts, np.ndarray): X = np.copy(pts)
        else:raise TypeError(f"The pts function argument needs to be either a torch.Tensor or numpy.ndarray object not a {type(pts).__name__} object as was given!")
        
        sv_dir = os.path.join('AnimatedPlotsFigs', dataset_name)
        os.makedirs(name=sv_dir, exist_ok=True)
    
        t = np.unique(X[:, 1])
        t.sort()
        rnd = int(np.log10(np.min(t[1:]-t[:-1]))) + 3
        y_max = np.ceil(vals.max())
        y_min = np.floor(vals.min())
        x_max, x_min = np.ceil(pts[:, 0].max()), np.floor(pts[:,0].min())
        mins = [x_min, y_min]
        maxs = [x_max, y_max]
        
        plot_fig, ax = plt.subplots()
        plot_fig.set_size_inches(w=10, h=8)
            
        ani_plot = animation.FuncAnimation(
            plot_fig, learned_animation, fargs=(ax, X, vals, maxs, mins, t, rnd, state), save_count=50, 
            frames=int(t.shape[0]/spdx), interval=100, repeat=True, blit=False)
        plot_fig.suptitle(fig_title,fontsize=10)
        writer = animation.FFMpegWriter(fps=30, codec='mpeg4', metadata=dict(artist='Me'), bitrate=-1)
        # ani_plot.save(ani_title+'.gif', writer=writer)
        ani_plot.save(os.path.join(sv_dir, ani_title+'.gif'), writer=writer)
        
        return None

    def ContourLikeComparisonPlot(self, dataset_name:str, pts:Union[np.ndarray,torch.Tensor], values:Union[np.ndarray,torch.Tensor], 
                                  fig_title:str='', state:str='', show_fig:bool=False, 
                                  save_fig:bool=True, save_title:str='',
                                  **kwargs)->None:
        """
        DESCRIPTIVE TEXT GOES HERE EVENTUALLY DESCRIBING THE FUNC ARGUMENTS as well as input parameter checking
        The values are expected to be N by 
        """
        
        
        # Something to make sure that we have the prediction and the exact values on the cpu and in numpy arrays for ploting
        if isinstance(values, torch.Tensor):vals = values.cpu().detach().numpy()
        elif isinstance(values, np.ndarray):vals = np.copy(values)
        else: raise TypeError(f"The values function argument needs to be either a torch.Tensor or numpy.ndarray object not a {type(values).__name__} object as was given!")
        # N by 2 array/tensors these need to be. 
        if isinstance(pts, torch.Tensor):X = pts.cpu().detach().numpy()
        elif isinstance(pts, np.ndarray): X = np.copy(pts)
        else:raise TypeError(f"The pts function argument needs to be either a torch.Tensor or numpy.ndarray object not a {type(pts).__name__} object as was given!")
        
        sv_dir = os.path.join('LearnedEquationPlots', dataset_name)
        os.makedirs(name=sv_dir, exist_ok=True)
        
        ## if needed, check the kwargs keys - if end up using additional keyword argument
        #  
        if 'dif' in kwargs.keys() and isinstance(kwargs['dif'], bool):
            dif = kwargs['dif']
        else:
             dif = False
        if 'exact' in kwargs.keys() and isinstance(kwargs['exact'], bool):
            exact = kwargs['exact']
        else:
             exact = False
        if 'learned' in kwargs.keys() and isinstance(kwargs['learned'], bool):
            learned = kwargs['learned']
        else:
            learned = False

        t = np.unique(X[:, 1])
        t.sort()
        n_t = t.shape[0]
        n_x = int(X.shape[0]/n_t)
        preds = np.empty_like(vals)
        # This is done just in case number of points is so larger that they catorch.nnot all be placed on the device at the same time
        # do to memory constrains 
        for i in range(n_t):
            preds[i*n_x:(i+1)*n_x,:] = self.net(torch.from_numpy(X[i*n_x:(i+1)*n_x,:]).to(device=self.device, dtype=self.data_type, non_blocking=True)).cpu().detach().numpy()
        preds = preds.reshape((n_t, n_x, ))
        vals = vals.reshape((n_t, n_x, ))
        diff = np.absolute(preds - vals)
        T = X[:,1].reshape((n_t, n_x, ))
        XX = X[:,0].reshape((n_t, n_x, ))

        # Now create the plots/figures and save them. 
        # First the figure that has all of the other plots within them.
        fig = plt.figure(figsize=(16,10), layout="constrained")
        mosaic = """AB;CC"""
        axs =  fig.subplot_mosaic(mosaic) # axs here is a dictionary with keywords being A, B and C see matplotlib mosaic for details
        c_map = 'jet'

        if vals.min()<0 and vals.max()<=0: norm = Normalize(vmin=vals.min(), vmax=0.0)
        elif vals.min()>=0 and vals.max()>0: norm = Normalize(vmin=0.0, vmax=vals.max())
        else: norm = TwoSlopeNorm(0.0, vals.min(), vals.max())
        c = axs["A"].imshow(vals, cmap=c_map, norm=norm,
                      extent=[XX.min(), XX.max(), T.min(), T.max()],
                      interpolation='bilinear', origin='lower', aspect='auto')
        axs["A"].set_title('True/Actual Function')
        axs["A"].set_xlabel('x')
        axs["A"].set_ylabel('t')
        fig.colorbar(c, ax=axs["A"], location='right', orientation='vertical',)
        # NOTE: that this is a problem when there are nan values in preds or any of the matrices as nan is always < & and > 0.0
        if preds.min()<0 and preds.max()<=0: norm = Normalize(vmin=preds.min(), vmax=0.0)
        elif preds.min()>=0 and preds.max()>0: norm = Normalize(vmin=0.0, vmax=preds.max())
        else: norm = TwoSlopeNorm(0.0, preds.min(), preds.max())    # This block is entered when preds has a nan value also does not through and error
        c = axs["B"].imshow(preds, cmap=c_map, norm=norm,
                      extent=[XX.min(), XX.max(), T.min(), T.max()],
                      interpolation='bilinear', origin='lower', aspect='auto')
        axs["B"].set_title('Pinns Learned Equation/Model '+state)
        axs["B"].set_xlabel('x')
        axs["B"].set_ylabel('t')
        fig.colorbar(c, ax=axs["B"], location='right', orientation='vertical',)

        if preds.min()<0 and preds.max()<=0: norm = Normalize(vmin=preds.min(), vmax=0.0)
        elif preds.min()>=0 and preds.max()>0: norm = Normalize(vmin=0.0, vmax=preds.max())
        else: norm = TwoSlopeNorm(0.0, preds.min(), preds.max())
        c = axs["C"].imshow(diff, cmap=c_map, norm=norm,
                      extent=[XX.min(), XX.max(), T.min(), T.max()],
                      interpolation='bilinear', origin='lower', aspect='auto')
        axs["C"].set_title('Absolute Difference')
        axs["C"].set_xlabel('x')
        axs["C"].set_ylabel('t')
        fig.colorbar(c, ax=axs["C"], location='right', orientation='vertical',)

        fig.suptitle(fig_title)
        if show_fig: 
            plt.show()
        if save_fig: 
            fig.savefig(fname=os.path.join(sv_dir, save_title+'.png'), format='png')

        # Now create and save the figure of just the actual/exact data given the keyword argument value. 
        if exact:
            fig, axis = plt.subplots(nrows=1, ncols=1, layout="constrained", figsize=(8,5))
            if vals.min()<0 and vals.max()<=0: norm = Normalize(vmin=vals.min(), vmax=0.0)
            elif vals.min()>=0 and vals.max()>0: norm = Normalize(vmin=0.0, vmax=vals.max())
            else: norm = TwoSlopeNorm(0.0, vals.min(), vals.max())
            c = axis.imshow(vals, cmap=c_map, norm=norm,
                          extent=[XX.min(), XX.max(), T.min(), T.max()],
                          interpolation='bilinear', origin='lower', aspect='auto')
            axis.set_title('True/Actual Function')
            axis.set_xlabel('x')
            axis.set_ylabel('t')
            fig.colorbar(c, ax=axis, location='right', orientation='vertical',)
            if save_fig: 
                fig.savefig(fname=os.path.join(sv_dir, save_title+'FunctionData.png'), format='png')
        # Now create and save the figure of just the difference between the true/exact data and the learned 
        # function/solution/equation using given dif keyword argument value. 
        if dif:
            fig, axis = plt.subplots(nrows=1, ncols=1, layout="constrained", figsize=(8,5))
            if preds.min()<0 and preds.max()<=0: norm = Normalize(vmin=preds.min(), vmax=0.0)
            elif preds.min()>=0 and preds.max()>0: norm = Normalize(vmin=0.0, vmax=preds.max())
            else: norm = TwoSlopeNorm(0.0, preds.min(), preds.max())
            c = axis.imshow(diff, cmap=c_map, norm=norm,
                          extent=[XX.min(), XX.max(), T.min(), T.max()],
                          interpolation='bilinear', origin='lower', aspect='auto')
            axis.set_title('Absolute Difference')
            axis.set_xlabel('x')
            axis.set_ylabel('t')
            fig.colorbar(c, ax=axis, location='right', orientation='vertical',)
            if save_fig: 
                fig.savefig(fname=os.path.join(sv_dir, save_title+'DifInExactandLrned.png'), format='png')
        # Now create and save the figure of just the learned equation/solution/function ussing the given 
        # learned keyword argument value. 
        if learned:
            fig, axis = plt.subplots(nrows=1, ncols=1, layout="constrained", figsize=(8,5))
            if preds.min()<0 and preds.max()<=0: norm = Normalize(vmin=preds.min(), vmax=0.0)
            elif preds.min()>=0 and preds.max()>0: norm = Normalize(vmin=0.0, vmax=preds.max())
            else: norm = TwoSlopeNorm(0.0, preds.min(), preds.max())    # This block is entered when preds has a nan value also does not through and error
            c = axis.imshow(preds, cmap=c_map, norm=norm,
                          extent=[XX.min(), XX.max(), T.min(), T.max()],
                          interpolation='bilinear', origin='lower', aspect='auto')
            axis.set_title('Pinns Learned Equation/Model '+state)
            axis.set_xlabel('x')
            axis.set_ylabel('t')
            fig.colorbar(c, ax=axis, location='right', orientation='vertical',)
            if save_fig: 
                fig.savefig(fname=os.path.join(sv_dir, save_title+'LearnedFunc.png'), format='png')
        
        return None


class OldEqLearner():
    """
    
    """
    def __init__(self,
        net:torch.nn.Module,
        Lmbda:torch.Tensor,
        lib_func:BaseFuncLib,
        sprs_slvr:SparseRegressAlg,
        data_dict:dict,
        tmprl_ords:List[int],
        col_pnts_smplr:Rand_Col_Sampler=Rand_Col_Sampler(),
        N_col_pnts:int=10000,
        ntwk_out_names:List[str]=None,
        device:torch.device=torch.device('cpu'),
        data_type:torch.dtype=None
    ):
        
        self.net = net
        self.lmbda = Lmbda
        self.lib_func = lib_func
        self.sprs_slvr = sprs_slvr
        self.data_dict = data_dict
        self.tmprl_ords = tmprl_ords
        self.col_pnts_smplr = col_pnts_smplr
        self.N_col_pnts = N_col_pnts
        self.ntwk_out_names = ntwk_out_names
        self.device = device
        self.data_type = data_type

        dtime = datetime.datetime.now()
        self.sv_fname = f"LearnerCreatedOnY{dtime.year}M{dtime.month}D{dtime.day}at{dtime.hour}Hr{dtime.minute}Min{dtime.second}Sec"
        
        self._AdamsPreTrnEpochs = None
        self._AdamsPreTrnLrnRt = None
        self._AdamsPreTrnAlpha = None
        self._AdmasPreTrnGamma = None
        self.AdamsPreTrnLambdas = None
        self.AdamsPreTrnFvus = None
        
        self._LBFGsPreTrnEpochs = None
        self._LBFGsPreTrnLrnRt = None
        self._LBFGsPreTrnAlpha = None
        self._LBFGsPreTrnGamma = None
        self.LbfgsPreTrnLambdas = None
        self.LbfgsPreTrnFvus = None
        
        self._ADO_iters = None
        self._ADO_epchs = None
        self._ADO_alphas = None
        self._ADO_lambdas = None
        self._ADO_FVUs = None
        self.earl_lmbds = None
        self.earl_fvus = None
        
        self._PstTrnAdamsEpochs = None
        self._PstTrnAdamsLrnRt = None
        self._PstTrnAdamsAlpha = None
        self.AdamsPstTrnLambdas = None
        self.AdamsPstTrnFvus = None
        
        self._PstTrnLBFGsEpochs = None
        self._PstTrnLBFGsLrnRt = None
        self._PstTrnLBFGsAlpha = None
        self.LbfgsPstTrnLambdas = None
        self.LbfgsPstTrnFvus = None

        self._AdamsPreTrnEpochs = None
        self._AdamsPreTrnLrnRt = None
        self._AdamsPreTrnAlpha = None
        self._AdmasPreTrnGamma = None
        self.trn_batch_size = None
        self.AdamsPreTrnLoss = None
        self.AdamsPreTstLoss = None
        self.AdamsPreTrnDataLoss = None
        self.AdamsPreTstDataLoss = None
        self.AdamsPreTrnEqLoss = None
        self.AdamsPreTstEqLoss = None
        self.AdamsPreTrnLambdas = None
        self.AdamsPreTrnFvus = None
        self._LBFGsPreTrnEpochs = None
        self._LBFGsPreTrnLrnRt = None
        self._LBFGsPreTrnAlpha = None
        self._LBFGsPreTrnGamma = None
        self.LbfgsPreTrnLoss = None
        self.LbfgsPreTstLoss = None
        self.LbfgsPreTrnDataLoss = None
        self.LbfgsPreTstDataLoss = None
        self.LbfgsPreTrnEqLoss = None
        self.LbfgsPreTstEqLoss = None
        self.LbfgsPreTrnLambdas = None
        self.LbfgsPreTrnFvus = None
        self.earl_lmbds = None
        self.earl_fvus = None
        self._ADO_lambdas = None
        self._ADO_FVUs = None
        self._ADO_iters = None
        self._ADO_epchs = None
        self._ADO_alphas = None
        self._PstTrnAdamsEpochs = None
        self._PstTrnAdamsLrnRt = None
        self._PstTrnAdamsAlpha = None
        self.AdamsPstTrnLoss = None
        self.AdamsPstTstLoss = None
        self.AdamsPstTrnDataLoss = None
        self.AdamsPstTstDataLoss = None
        self.AdamsPstTrnEqLoss = None
        self.AdamsPstTstEqLoss = None
        self.AdamsPstTrnLambdas = None
        self.AdamsPstTrnFvus = None
        self._PstTrnLBFGsEpochs = None
        self._PstTrnLBFGsLrnRt = None
        self._PstTrnLBFGsAlpha = None
        self.LbfgsPstTrnLoss = None
        self.LbfgsPstTstLoss = None
        self.LbfgsPstTrnDataLoss = None
        self.LbfgsPstTstDataLoss = None
        self.LbfgsPstTrnEqLoss = None
        self.LbfgsPstTstEqLoss = None
        self.LbfgsPstTrnLambdas = None
        self.LbfgsPstTrnFvus = None

    def Learned_EQ(self, coefs:torch.Tensor=None, output:bool=False, sup_zeros:bool=True, dec_rnd:int=8, prnt_sig_dif:int=None)->str:
        """
        Returned what the current learned equations it. This is done by using current values in the Lambda
        matrix/vector/tensor and the current library terms in the following way - 
            Lambda(0)libraryfuncs(0) + Lambda(1)libraryfuncs(1) + .... + Lambda(end)libraryfuncs(end)
        So the number of library terms needs to be the  same as the number of elements in Lambda. Input 
        arguments/parameters are the following -
            EqLHS: the Left Hand Side of the equation if you want it included in the returned Diff Eq.
                Optional but if included needs to be a string.
            output: Boolean on whether or not (False) to print the learned equation to standard out 
            sup_zeros: (Suppress Zeros) Whether or not to suppress the terms in the library that have a 
                0 coefficient that is multiplying them. If true any library function that would be 
                multiplied by a zero is not printed
            dec_rnd: (decimal round) the number of decimal places to round up to when printing the 
                current learned equation.
        return - The learned equation as a string
        TODO - Possibly changed the returned equation to have the Left Hand Side of the pde since right 
            now it only has the RHS returned but in the training I have it explicitly set to have the
            LHS as the first temporal partial so probably merits to having this function return u_t = ....
            Also maybe have this update a class variable that contains the current learned equation (03/11/2022)
        """

        library_names = self.lib_func.get_library_names(net_out_func_names=self.ntwk_out_names)
        n = len(library_names)
        if coefs is None:
            coefs = torch.round(torch.clone(self.lmbda.data.detach()), decimals=dec_rnd)

        if n != coefs.shape[0] or n==0:
            print('ERROR! - The number of Library function names is not the same as the number of Lambda elements')
            print('or the Library of function names is empty - Returning None')
            return None
        if not isinstance(dec_rnd, int):
            print('Given decimal rounding number is not a int. Will default to using 8 decimal places to round')
            dec_rnd=8
        if not isinstance(prnt_sig_dif, int):
            prnt_sig_dif=8
        # Now that the little amount of argument checking has been done lets print out current learned equation
        N_outs = coefs.shape[-1]
        if self.ntwk_out_names:
            EqLHs = self.ntwk_out_names
        else:
            EqLHs = ["O"+str(j+1) for j in range(N_outs)]
        
        lrnd_eq = ""
        plus = ' + '
        num_frmtr = "%3.{}e".format(prnt_sig_dif)
        for k in range(N_outs):
            lrnd_eq = lrnd_eq + EqLHs[k] + "_"+self.tmprl_ords[k]*"t" +" = "
            
            for i in range(n):
                # num = str(np.around(coefs[i,k].item(), decimals=dec_rnd))
                num = num_frmtr % (np.around(coefs[i,k].item(), decimals=dec_rnd))
                if coefs[i,0].item() == 0 and sup_zeros:
                    continue
                lrnd_eq = lrnd_eq + num +'*'+library_names[i]
                if i != n-1:
                    lrnd_eq = lrnd_eq + plus
            if lrnd_eq[len(lrnd_eq)-2] == '+':
                lrnd_eq = lrnd_eq[0:len(lrnd_eq)-2] # has an extra space at the end of the sting
                # lrnd_eq = lrnd_eq[len(lrnd_eq)-3] # does not have a space at the end of the string
            if k!=N_outs-1:
                lrnd_eq = lrnd_eq + "\n"
        
        if output:
            print('The current Learned Equation is - ')
            print(lrnd_eq)
        return lrnd_eq

    def FVU_Calc(self, lib_ceofs:torch.Tensor):
        """
            Calculate the Fraction of Variance Unexplained (FVU) for the 
            linearly regressioned learned equation using the coefficient 
            values offound in the lmbda parameter at any moment. Data used 
            to find the FVU is the test/validation data set. 
        """
        # Set Up stuff
        pnts = torch.from_numpy(self.data_dict['Test_Inputs']).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        trgts = Nth_temporal_prtls(values=self.net(pnts), pts=pnts, orders=self.tmprl_ords).detach()
        trg_means = trgts.mean(dim=0)
        lib_evals = self.lib_func.Calc(network=self.net, inpts=pnts).detach()
        estimates = lib_evals @ lib_ceofs
        # Calculate the Sum of Squared pred errors (SSerr) and Total Sum of Squares (SStot)
        SSerr = torch.sum( (trgts - estimates)**2, dim=0)
        SStot = torch.sum( (trgts - trg_means)**2, dim=0)
        fvu = SSerr / SStot
        return fvu

    def Save_Model(self, data_set:str, fname:str=None)->None:
        """
        
        """
        og_dir = os.getcwd()
        try:
            os.mkdir('Saved_Models')
        except FileExistsError:
            print('Saved_Models'+' Directory already exists so did not create it')
        try: 
            os.chdir('Saved_Models')
        except (OSError, FileNotFoundError,PermissionError,NotADirectoryError):
            print('Could not change directory to {}. Animated Learned Plots will be writen to mp4 file in CWD={}'.format('Saved_Models', os.getcwd()))
        try:
            os.mkdir(data_set)
        except FileExistsError:
            print('{} Directory already exists so did not create it'.format(data_set))
        try: 
            os.chdir(data_set)
        except (OSError, FileNotFoundError,PermissionError,NotADirectoryError):
            print('Could not change directory to {}. Animated Learned Plots will be writen to mp4 file in CWD={}'.format(data_set, os.getcwd()))
        chckPntDic = {
            'net_state_dict': self.net.state_dict(),  # First the model's parameters then all the class variables/parameters should be the exact same as the in init method.
            'lambda': self.lmbda.data.cpu().numpy(),  # Adams Pretrain variables
            '_AdamsPreTrnEpochs':self._AdamsPreTrnEpochs,
            '_AdamsPreTrnLrnRt':self._AdamsPreTrnLrnRt,
            '_AdamsPreTrnAlpha':self._AdamsPreTrnAlpha,
            '_AdmasPreTrnGamma':self._AdmasPreTrnGamma,
            'trn_batch_size':self.trn_batch_size,
            'AdamsPreTrnLoss':self.AdamsPreTrnLoss,
            'AdamsPreTstLoss':self.AdamsPreTstLoss,
            'AdamsPreTrnDataLoss':self.AdamsPreTrnDataLoss,
            'AdamsPreTstDataLoss':self.AdamsPreTstDataLoss,
            'AdamsPreTrnEqLoss':self.AdamsPreTrnEqLoss,
            'AdamsPreTstEqLoss':self.AdamsPreTstEqLoss,
            'AdamsPreTrnLambdas':self.AdamsPreTrnLambdas,
            'AdamsPreTrnFvus':self.AdamsPreTrnFvus,     # LBFGs Pretrain variables
            '_LBFGsPreTrnEpochs':self._LBFGsPreTrnEpochs,
            '_LBFGsPreTrnLrnRt':self._LBFGsPreTrnLrnRt,
            '_LBFGsPreTrnAlpha':self._LBFGsPreTrnAlpha,
            '_LBFGsPreTrnGamma':self._LBFGsPreTrnGamma,
            'LbfgsPreTrnLoss':self.LbfgsPreTrnLoss,
            'LbfgsPreTstLoss':self.LbfgsPreTstLoss,
            'LbfgsPreTrnDataLoss':self.LbfgsPreTrnDataLoss,
            'LbfgsPreTstDataLoss':self.LbfgsPreTstDataLoss,
            'LbfgsPreTrnEqLoss':self.LbfgsPreTrnEqLoss,
            'LbfgsPreTstEqLoss':self.LbfgsPreTstEqLoss,
            'LbfgsPreTrnLambdas':self.LbfgsPreTrnLambdas,
            'LbfgsPreTrnFvus':self.LbfgsPreTrnFvus,
            'N_col_pnts': self.N_col_pnts,          # Now the stuff for ADO-Like Alg
            'earl_lmbds':self.earl_lmbds,
            'earl_fvus':self.earl_fvus,
            '_ADO_lambdas':self._ADO_lambdas,
            '_ADO_FVUs':self._ADO_FVUs,
            '_ADO_iters':self._ADO_iters,
            '_ADO_epchs':self._ADO_epchs,
            '_ADO_alphas':self._ADO_alphas,          # Now the post training stuff
            '_PstTrnAdamsEpochs':self._PstTrnAdamsEpochs,
            '_PstTrnAdamsLrnRt':self._PstTrnAdamsLrnRt,
            '_PstTrnAdamsAlpha':self._PstTrnAdamsAlpha,
            'AdamsPstTrnLoss':self.AdamsPstTrnLoss,
            'AdamsPstTstLoss':self.AdamsPstTstLoss,
            'AdamsPstTrnDataLoss':self.AdamsPstTrnDataLoss,
            'AdamsPstTstDataLoss':self.AdamsPstTstDataLoss,
            'AdamsPstTrnEqLoss':self.AdamsPstTrnEqLoss,
            'AdamsPstTstEqLoss':self.AdamsPstTstEqLoss,
            'AdamsPstTrnLambdas':self.AdamsPstTrnLambdas,
            'AdamsPstTrnFvus':self.AdamsPstTrnFvus,      # LBFGs Post training stuff
            '_PstTrnLBFGsEpochs':self._PstTrnLBFGsEpochs,
            '_PstTrnLBFGsLrnRt':self._PstTrnLBFGsLrnRt,
            '_PstTrnLBFGsAlpha':self._PstTrnLBFGsAlpha,
            'LbfgsPstTrnLoss':self.LbfgsPstTrnLoss,
            'LbfgsPstTstLoss':self.LbfgsPstTstLoss,
            'LbfgsPstTrnDataLoss':self.LbfgsPstTrnDataLoss,
            'LbfgsPstTstDataLoss':self.LbfgsPstTstDataLoss,
            'LbfgsPstTrnEqLoss':self.LbfgsPstTrnEqLoss,
            'LbfgsPstTstEqLoss':self.LbfgsPstTstEqLoss,
            'LbfgsPstTrnLambdas':self.LbfgsPstTrnLambdas,
            'LbfgsPstTrnFvus':self.LbfgsPstTrnFvus,
        }
        if fname is None:
            fname = self.sv_fname

        path = fname+'.tar'
        torch.save(chckPntDic, path)
        try:
            os.chdir(og_dir)
        except (OSError, FileNotFoundError, PermissionError, NotADirectoryError):
            print(f"Could not change back to the original working directory/folder after changing to save the model checkpoint.")

    def Load_Model(self,folder_loc:str, fname:str)->None:
        """
        
        """
        path = folder_loc +'/'+fname+'.tar'
        state = torch.load(path,)

        self.net.load_state_dict(state['net_state_dict'])
        self.lmbda = torch.from_numpy(state['lambda']).to(device=self.device, dtype=self.data_type).requires_grad_(True)

        self._AdamsPreTrnEpochs = state['_AdamsPreTrnEpochs']
        self._AdamsPreTrnLrnRt = state['_AdamsPreTrnLrnRt']
        self._AdamsPreTrnAlpha = state['_AdamsPreTrnAlpha']
        self._AdmasPreTrnGamma = state['_AdmasPreTrnGamma']
        self.trn_batch_size = state['trn_batch_size']
        self.AdamsPreTrnLoss = state['AdamsPreTrnLoss']
        self.AdamsPreTstLoss = state['AdamsPreTstLoss']
        self.AdamsPreTrnDataLoss = state['AdamsPreTrnDataLoss']
        self.AdamsPreTstDataLoss = state['AdamsPreTstDataLoss']
        self.AdamsPreTrnEqLoss = state['AdamsPreTrnEqLoss']
        self.AdamsPreTstEqLoss = state['AdamsPreTstEqLoss']
        self.AdamsPreTrnLambdas = state['AdamsPreTrnLambdas']
        self.AdamsPreTrnFvus = state['AdamsPreTrnFvus']
        self._LBFGsPreTrnEpochs = state['_LBFGsPreTrnEpochs']
        self._LBFGsPreTrnLrnRt = state['_LBFGsPreTrnLrnRt']
        self._LBFGsPreTrnAlpha = state['_LBFGsPreTrnAlpha']
        self._LBFGsPreTrnGamma = state['_LBFGsPreTrnGamma']
        self.LbfgsPreTrnLoss = state['LbfgsPreTrnLoss']
        self.LbfgsPreTstLoss = state['LbfgsPreTstLoss']
        self.LbfgsPreTrnDataLoss = state['LbfgsPreTrnDataLoss']
        self.LbfgsPreTstDataLoss = state['LbfgsPreTstDataLoss']
        self.LbfgsPreTrnEqLoss = state['LbfgsPreTrnEqLoss']
        self.LbfgsPreTstEqLoss = state['LbfgsPreTstEqLoss']
        self.LbfgsPreTrnLambdas = state['LbfgsPreTrnLambdas']
        self.LbfgsPreTrnFvus = state['LbfgsPreTrnFvus']
        self.N_col_pnts = state['N_col_pnts']
        self.earl_lmbds = state['earl_lmbds']
        self.earl_fvus = state['earl_fvus']
        self._ADO_lambdas = state['_ADO_lambdas']
        self._ADO_FVUs = state['_ADO_FVUs']
        self._ADO_iters = state['_ADO_iters']
        self._ADO_epchs = state['_ADO_epchs']
        self._ADO_alphas = state['_ADO_alphas']
        self._PstTrnAdamsEpochs = state['_PstTrnAdamsEpochs']
        self._PstTrnAdamsLrnRt = state['_PstTrnAdamsLrnRt']
        self._PstTrnAdamsAlpha = state['_PstTrnAdamsAlpha']
        self.AdamsPstTrnLoss = state['AdamsPstTrnLoss']
        self.AdamsPstTstLoss = state['AdamsPstTstLoss']
        self.AdamsPstTrnDataLoss = state['AdamsPstTrnDataLoss']
        self.AdamsPstTstDataLoss = state['AdamsPstTstDataLoss']
        self.AdamsPstTrnEqLoss = state['AdamsPstTrnEqLoss']
        self.AdamsPstTstEqLoss = state['AdamsPstTstEqLoss']
        self.AdamsPstTrnLambdas = state['AdamsPstTrnLambdas']
        self.AdamsPstTrnFvus = state['AdamsPstTrnFvus']
        self._PstTrnLBFGsEpochs = state['_PstTrnLBFGsEpochs']
        self._PstTrnLBFGsLrnRt = state['_PstTrnLBFGsLrnRt']
        self._PstTrnLBFGsAlpha = state['_PstTrnLBFGsAlpha']
        self.LbfgsPstTrnLoss = state['LbfgsPstTrnLoss']
        self.LbfgsPstTstLoss = state['LbfgsPstTstLoss']
        self.LbfgsPstTrnDataLoss = state['LbfgsPstTrnDataLoss']
        self.LbfgsPstTstDataLoss = state['LbfgsPstTstDataLoss']
        self.LbfgsPstTrnEqLoss = state['LbfgsPstTrnEqLoss']
        self.LbfgsPstTstEqLoss = state['LbfgsPstTstEqLoss']
        self.LbfgsPstTrnLambdas = state['LbfgsPstTrnLambdas']
        self.LbfgsPstTrnFvus = state['LbfgsPstTrnFvus']

    def Adams_Pretraining(self, alpha:float=0.5, gamma:float=0.25, epochs:int=1000,
        lrn_rt:float=0.001, lp_ord:float=1.0, Save_File_Name:str=None, tstFreq:int=100, schdlr_freq:int=0,
        betas:Tuple=(0.9, 0.99), eps:float=1e-8, wght_dcy:float=0, amsgrad:bool=False):

        if Save_File_Name==None or (not isinstance(Save_File_Name, str)):
            msg = ("The Save_File_Name argument was None or was not a string object the file name  \n"
                   "where the model will periodically be saved to is based off the day and time the \n" 
                   "class object was created. ")
            warnings.warn(message=msg, stacklevel=1)
            Save_File_Name = self.sv_fname
        Save_File_Name += ".pt"

        self.AdamsPreTrnLambdas = np.empty(shape=(2, *self.lmbda.data.shape), dtype=float)
        self.AdamsPreTrnFvus = np.empty(shape=(2, self.lmbda.data.shape[1]), dtype=float)
        self.AdamsPreTrnLambdas[0] = self.lmbda.data.cpu().numpy()
        self.AdamsPreTrnFvus[0] = self.FVU_Calc(lib_ceofs=self.lmbda.data).cpu().numpy()

        optim = torch.optim.Adam(self.net.parameters(), lr=lrn_rt, betas=betas, eps=eps, weight_decay=wght_dcy, amsgrad=amsgrad)
        optim.param_groups[0]['params'].append(self.lmbda)
        schdlr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.9, last_epoch=-1, ) 
        # schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10,
        #                                         threshold=0.001, threshold_mode='rel', cooldown=1, min_lr=0, 
        #                                             eps=1e-08, )

        train_inputs = torch.from_numpy(self.data_dict['Train_Inputs']).to(device=self.device, dtype=self.data_type)
        train_targets = torch.from_numpy(self.data_dict['Train_Targets']).to(device=self.device, dtype=self.data_type)
        test_inputs = torch.from_numpy(self.data_dict['Test_Inputs']).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        test_targets = torch.from_numpy(self.data_dict['Test_Targets']).to(device=self.device, dtype=self.data_type)
        self.nDpnts = train_inputs.size(0)
        b_sizes = train_inputs.size(0)
        
        # n_trn_points = train_inputs.size(0)
        # n_dims = train_inputs.size(1)
        # N_btchs = int(np.ceil(n_trn_points/b_sizes))
        # n_col_pnts = int(np.around(self.N_col_pnts/n_trn_points) * n_trn_points)
        # if n_col_pnts==0:
        #     n_col_pnts = n_trn_points
        # k = int(n_col_pnts / n_trn_points)
        
        
        # np_colpnts = self.col_pnts_smplr.sample(n_pnts=n_col_pnts)
        # if k==1:
        #     colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type)
        # else:
        #     colpnts = torch.from_numpy(np_colpnts.reshape((n_trn_points, k, n_dims))).to(device=self.device, dtype=self.data_type)

        # dset = TensorDataset (train_inputs, train_targets, colpnts)
        # loader = DataLoader(dataset=dset, batch_size=b_sizes,shuffle=True, pin_memory=False)
        
        np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type)

        tot_loss = torch.zeros(size=(epochs, 1), device=self.device,)
        data_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)
        pde_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)
        lp_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)

        tst_loss = torch.zeros(size=(epochs, 1), device=self.device,)
        tst_data_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)
        tst_pde_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)

        # lmbda_lst = []

        for i in range(epochs):
            self.net.train(True)
            # lmbda_lst.append(torch.clone(self.lmbda.data.detach()))

            # colpnts = (ten_ub - ten_lb)*torch.rand((N_col,2), device=dvc, dtype=torch.float32, requires_grad=True) + ten_lb
            colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
                
            optim.zero_grad(set_to_none=True)

            dataL = torch.mean((self.net(train_inputs) - train_targets)**2, dim=0)
            
            col_preds = self.net(colpnts)
            t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords)
            lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts)

            pdeL = torch.mean((t_prtls - lib_evals@self.lmbda)**2, dim=0,)
            # loss = torch.sum(dataL + alpha*pdeL + gamma*torch.linalg.vector_norm(self.lmbda, ord=1, dim=0))
            lp_lss = Torch_Lp_Loss(self.lmbda, p=lp_ord)
            loss = torch.sum(dataL + alpha*pdeL + gamma*lp_lss)
            loss.backward()
            optim.step()

            tot_loss[i] = loss.detach()
            data_loss[i] = dataL.detach()
            pde_loss[i] = pdeL.detach()
            lp_loss[i] = lp_lss.detach()

            if schdlr_freq and i>0 and i%schdlr_freq==0:
                schdlr.step()
                # schdlr.step(metrics=loss)
            self.net.eval()
            if i%tstFreq==0 or i+1==epochs:
                tst_dataL = torch.mean((self.net(test_inputs) - test_targets)**2, dim=0).detach_()
                tst_col_preds = self.net(test_inputs)
                tst_t_prtls = Nth_temporal_prtls(values=tst_col_preds, pts=test_inputs, orders=self.tmprl_ords).detach_()
                tst_lib_evals = self.lib_func.Calc(network=self.net, inpts=test_inputs)
                tst_pdeL = torch.mean((tst_t_prtls - tst_lib_evals@self.lmbda)**2, dim=0,).detach_()
                # ts_loss = torch.sum(tst_dataL + alpha*tst_pdeL + gamma*torch.linalg.vector_norm(self.lmbda, ord=1, dim=0))
                ts_loss = torch.sum(tst_dataL + alpha*tst_pdeL + gamma*Torch_Lp_Loss(self.lmbda, p=lp_ord))
                tst_loss[i] = ts_loss.detach()
                tst_data_loss[i] = tst_dataL.detach()
                tst_pde_loss[i] = tst_pdeL.detach()

        optim.zero_grad(set_to_none=True)
        print('Finished Adams Pretraining')
        del train_inputs, train_targets, test_inputs, test_targets, colpnts
        self._AdamsPreTrnEpochs = epochs
        self._AdamsPreTrnLrnRt = lrn_rt
        self._AdamsPreTrnAlpha = alpha
        self._AdmasPreTrnGamma = gamma
        self.trn_batch_size = b_sizes
        self.AdamsPreTrnLoss = tot_loss.cpu().numpy()
        self.AdamsPreTstLoss = tst_loss.cpu().numpy()
        self.AdamsPreTrnDataLoss = data_loss.cpu().numpy()
        self.AdamsPreTstDataLoss = tst_data_loss.cpu().numpy()
        self.AdamsPreTrnEqLoss = pde_loss.cpu().numpy()
        self.AdamsPreTstEqLoss = tst_pde_loss.cpu().numpy()
        self.AdamsPreLpLosses = lp_loss.cpu().numpy()
        self.AdamsPreTrnLambdas[1] = self.lmbda.data.cpu().numpy()
        # self.AdamsPreTrnIterLambdas = torch.stack(lmbda_lst, dim=0).cpu().numpy()
        self.AdamsPreTrnFvus[1] = self.FVU_Calc(lib_ceofs=self.lmbda.data).cpu().numpy()

    def Batched_Adams_Pretraining(self, alpha:float=0.5, gamma:float=0.25, epochs:int=1000,
        lrn_rt:float=0.001, b_sizes:int=1, lp_ord:float=1.0, Save_File_Name:str=None, tstFreq:int=100, schdlr_freq:int=0,
        betas:Tuple=(0.9, 0.99), eps:float=1e-8, wght_dcy:float=0, amsgrad:bool=False):

        if Save_File_Name==None or (not isinstance(Save_File_Name, str)):
            msg = ("The Save_File_Name argument was None or was not a string object the file name  \n"
                   "where the model will periodically be saved to is based off the day and time the \n" 
                   "class object was created. ")
            warnings.warn(message=msg, stacklevel=1)
            Save_File_Name = self.sv_fname
        Save_File_Name += ".pt"

        self.AdamsPreTrnLambdas = np.empty(shape=(2, *self.lmbda.data.shape), dtype=float)
        self.AdamsPreTrnFvus = np.empty(shape=(2, self.lmbda.data.shape[1]), dtype=float)
        self.AdamsPreTrnLambdas[0] = self.lmbda.data.cpu().numpy()
        self.AdamsPreTrnFvus[0] = self.FVU_Calc(lib_ceofs=self.lmbda.data).cpu().numpy()

        optim = torch.optim.Adam(self.net.parameters(), lr=lrn_rt, betas=betas, eps=eps, weight_decay=wght_dcy, amsgrad=amsgrad)
        optim.param_groups[0]['params'].append(self.lmbda)
        schdlr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.9, last_epoch=-1, ) 
        # schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10,
        #                                         threshold=0.001, threshold_mode='rel', cooldown=1, min_lr=0, 
        #                                             eps=1e-08, )

        train_inputs = torch.from_numpy(self.data_dict['Train_Inputs']).to(device=self.device, dtype=self.data_type)
        train_targets = torch.from_numpy(self.data_dict['Train_Targets']).to(device=self.device, dtype=self.data_type)
        test_inputs = torch.from_numpy(self.data_dict['Test_Inputs']).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        test_targets = torch.from_numpy(self.data_dict['Test_Targets']).to(device=self.device, dtype=self.data_type)
        self.nDpnts = train_inputs.size(0)
        
        n_trn_points = train_inputs.size(0)
        n_dims = train_inputs.size(1)
        N_btchs = int(np.ceil(n_trn_points/b_sizes))
        n_col_pnts = int(np.around(self.N_col_pnts/n_trn_points) * n_trn_points)
        if n_col_pnts==0:
            n_col_pnts = n_trn_points
        k = int(n_col_pnts / n_trn_points)
        
        
        np_colpnts = self.col_pnts_smplr.sample(n_pnts=n_col_pnts)
        if k==1:
            colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type)
        else:
            colpnts = torch.from_numpy(np_colpnts.reshape((n_trn_points, k, n_dims))).to(device=self.device, dtype=self.data_type)

        dset = TensorDataset (train_inputs, train_targets, colpnts)
        loader = DataLoader(dataset=dset, batch_size=b_sizes,shuffle=True, pin_memory=False)

        tot_loss = torch.zeros(size=(epochs, 1), device=self.device,)
        data_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)
        pde_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)
        lp_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)

        tst_loss = torch.zeros(size=(epochs, 1), device=self.device,)
        tst_data_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)
        tst_pde_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)


        for i in range(epochs):
            self.net.train(True)
            
            for _, (tnn_ins, trn_trgts, cols) in enumerate(loader):
                eq_pnts = cols.view((-1, n_dims)).requires_grad_(True)
                optim.zero_grad(set_to_none=True)

                dataL = torch.mean((self.net(tnn_ins) - trn_trgts)**2, dim=0)
                col_preds = self.net(eq_pnts)
                t_prtls = Nth_temporal_prtls(values=col_preds, pts=eq_pnts, orders=self.tmprl_ords)
                lib_evals = self.lib_func.Calc(network=self.net, inpts=eq_pnts)
                pdeL = torch.mean((t_prtls - lib_evals@self.lmbda)**2, dim=0,)
                lp_lss = Torch_Lp_Loss(self.lmbda, p=lp_ord)
                # loss = torch.sum(dataL + alpha*pdeL + gamma*torch.linalg.vector_norm(self.lmbda, ord=1, dim=0))
                loss = torch.sum(dataL + alpha*pdeL + gamma*lp_lss)
                loss.backward()

                tot_loss[i].add_( loss.detach() )
                data_loss[i].add_( dataL.detach() )
                pde_loss[i].add_( pdeL.detach() )
                lp_loss[i].add_( lp_lss.detach() )

                optim.step()
                del dataL, col_preds, t_prtls, lib_evals, pdeL, loss
   
            tot_loss[i].divide_(N_btchs)
            data_loss[i].divide_(N_btchs)
            pde_loss[i].divide_(N_btchs)
            lp_loss[i].divide_(N_btchs)

            if schdlr_freq and i>0 and i%schdlr_freq==0:
                schdlr.step()
                # schdlr.step(metrics=loss)
            self.net.eval()
            if i%tstFreq==0 or i+1==epochs:
                tst_dataL = torch.mean((self.net(test_inputs) - test_targets)**2, dim=0).detach_()
                tst_col_preds = self.net(test_inputs) 
                tst_t_prtls = Nth_temporal_prtls(values=tst_col_preds, pts=test_inputs, orders=self.tmprl_ords).detach_()
                tst_lib_evals = self.lib_func.Calc(network=self.net, inpts=test_inputs)
                tst_pdeL = torch.mean((tst_t_prtls - tst_lib_evals@self.lmbda)**2, dim=0,).detach_()
                # ts_loss = torch.sum(tst_dataL + alpha*tst_pdeL + gamma*torch.linalg.vector_norm(self.lmbda, ord=1, dim=0))
                ts_loss = torch.sum(tst_dataL + alpha*tst_pdeL + gamma*Torch_Lp_Loss(self.lmbda, p=lp_ord))
                tst_loss[i] = ts_loss.detach()
                tst_data_loss[i] = tst_dataL.detach()
                tst_pde_loss[i] = tst_pdeL.detach()

        optim.zero_grad(set_to_none=True)
        print('Finished Adams Pretraining')
        del train_inputs, train_targets, test_inputs, test_targets, colpnts
        self._AdamsPreTrnEpochs = epochs
        self._AdamsPreTrnLrnRt = lrn_rt
        self._AdamsPreTrnAlpha = alpha
        self._AdmasPreTrnGamma = gamma
        self.trn_batch_size = b_sizes
        self.AdamsPreTrnLoss = tot_loss.cpu().numpy()
        self.AdamsPreTstLoss = tst_loss.cpu().numpy()
        self.AdamsPreTrnDataLoss = data_loss.cpu().numpy()
        self.AdamsPreTstDataLoss = tst_data_loss.cpu().numpy()
        self.AdamsPreTrnEqLoss = pde_loss.cpu().numpy()
        self.AdamsPreTstEqLoss = tst_pde_loss.cpu().numpy()
        self.AdamsPreLpLosses = lp_loss.cpu().numpy()
        self.AdamsPreTrnLambdas[1] = self.lmbda.data.cpu().numpy()
        self.AdamsPreTrnFvus[1] = self.FVU_Calc(lib_ceofs=self.lmbda.data).cpu().numpy()

    def Lbfgs_Pretraining(self, alpha:float=0.5, gamma:float=0.25, epochs:int=1000,
        lrn_rt:float=0.001, Save_File_Name:str=None, tstFreq:int=100, schdlr_freq:int=0,
        max_it:int=20, max_evl:int=None, grad_tol:float=1e-07, tol_change:float=1e-09,
            history_size:int=100, line_srch_fn:str=None):
    
        if Save_File_Name==None or (not isinstance(Save_File_Name, str)):
            msg = ("The Save_File_Name argument was None or was not a string object the file name  \n"
                   "where the model will periodically be saved to is based off the day and time the \n" 
                   "class object was created. ")
            warnings.warn(message=msg, stacklevel=1)
            Save_File_Name = self.sv_fname
        Save_File_Name += ".pt"

        self.LbfgsPreTrnLambdas = np.empty(shape=(2, *self.lmbda.data.shape), dtype=float)
        self.LbfgsPreTrnFvus = np.empty(shape=(2, self.lmbda.data.shape[1]), dtype=float)
        self.LbfgsPreTrnLambdas[0] = self.lmbda.data.cpu().numpy()
        self.LbfgsPreTrnFvus[0] = self.FVU_Calc(lib_ceofs=self.lmbda.data).cpu().numpy()

        optim = torch.optim.LBFGS(params=self.net.parameters(), lr=lrn_rt, max_iter=max_it, max_eval=max_evl,
            tolerance_grad=grad_tol, tolerance_change=tol_change, history_size=history_size, line_search_fn=line_srch_fn)
        optim.param_groups[0]['params'].append(self.lmbda)
        schdlr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.9, last_epoch=-1, ) 
        # schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10,
        #                                         threshold=0.001, threshold_mode='rel', cooldown=1, min_lr=0, 
        #                                             eps=1e-08, )

        train_inputs = torch.from_numpy(self.data_dict['Train_Inputs']).to(device=self.device, dtype=self.data_type)
        train_targets = torch.from_numpy(self.data_dict['Train_Targets']).to(device=self.device, dtype=self.data_type)
        test_inputs = torch.from_numpy(self.data_dict['Test_Inputs']).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        test_targets = torch.from_numpy(self.data_dict['Test_Targets']).to(device=self.device, dtype=self.data_type)
        self.nDpnts = train_inputs.size(0)

        np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        # if not colpnts.requires_grad:
        #     colpnts.requires_grad_(True)

        tot_loss = torch.zeros(size=(epochs, 1), device=self.device,)
        data_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)
        pde_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)

        tst_loss = torch.zeros(size=(epochs, 1), device=self.device,)
        tst_data_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)
        tst_pde_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)

        def closure():
            optim.zero_grad(set_to_none=True)

            dataL = torch.mean((self.net(train_inputs) - train_targets)**2, dim=0)
            col_preds = self.net(colpnts)
            t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords)
            lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts)
            pdeL = torch.mean((t_prtls - lib_evals@self.lmbda)**2, dim=0,)
            loss = torch.sum(dataL + alpha*pdeL + gamma*torch.linalg.vector_norm(self.lmbda, ord=1, dim=0))

            # loss.backward()
            if loss.isnan().sum():
                    print(f"Note that dataLs={dataL} colloLs={pdeL} and loss={loss}")
                    optim.zero_grad(set_to_none=True)
            if loss.requires_grad and torch.isfinite(loss):
                loss.backward()
            # elif loss.isnan().sum():
            #     optim.zero_grad(set_to_none=True)
            return loss


        for i in range(epochs):

            self.net.train(mode=True)

            optim.step(closure)
            if schdlr_freq and i>0 and i%schdlr_freq==0:
                schdlr.step()
                # schdlr.step(metrics=loss)

            self.net.eval()
            trn_dataL = torch.mean((self.net(train_inputs) - train_targets)**2, dim=0).detach()
            trn_col_preds = self.net(colpnts)
            trn_t_prtls = Nth_temporal_prtls(values=trn_col_preds, pts=colpnts, orders=self.tmprl_ords).detach()
            trn_lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts)
            trn_pdeL = torch.mean((trn_t_prtls - trn_lib_evals@self.lmbda)**2, dim=0,).detach()
            trn_loss = torch.sum(trn_dataL + alpha*trn_pdeL + gamma*torch.linalg.vector_norm(self.lmbda, ord=1, dim=0)).detach()

            tot_loss[i] = trn_loss.detach()
            data_loss[i] = trn_dataL.detach()
            pde_loss[i] = trn_pdeL.detach()

            if i%tstFreq==0 or i+1==epochs:
                tst_dataL = torch.mean((self.net(test_inputs) - test_targets)**2, dim=0).detach()
                tst_col_preds = self.net(test_inputs)
                tst_t_prtls = Nth_temporal_prtls(values=tst_col_preds, pts=test_inputs, orders=self.tmprl_ords).detach()
                tst_lib_evals = self.lib_func.Calc(network=self.net, inpts=test_inputs)
                tst_pdeL = torch.mean((tst_t_prtls - tst_lib_evals@self.lmbda)**2, dim=0,).detach()
                ts_loss = torch.sum(tst_dataL + alpha*tst_pdeL + gamma*torch.linalg.vector_norm(self.lmbda, ord=1, dim=0)).detach()
                tst_loss[i] = ts_loss.detach()
                tst_data_loss[i] = tst_dataL.detach()
                tst_pde_loss[i] = tst_pdeL.detach()

        optim.zero_grad(set_to_none=True)
        print('Finished LBFGS Pretraining')
        del train_inputs, train_targets, test_inputs, test_targets, colpnts
        self._LBFGsPreTrnEpochs = epochs
        self._LBFGsPreTrnLrnRt = lrn_rt
        self._LBFGsPreTrnAlpha = alpha
        self._LBFGsPreTrnGamma = gamma
        self.LbfgsPreTrnLoss = tot_loss.cpu().numpy()
        self.LbfgsPreTstLoss = tst_loss.cpu().numpy()
        self.LbfgsPreTrnDataLoss = data_loss.cpu().numpy()
        self.LbfgsPreTstDataLoss = tst_data_loss.cpu().numpy()
        self.LbfgsPreTrnEqLoss = pde_loss.cpu().numpy()
        self.LbfgsPreTstEqLoss = tst_pde_loss.cpu().numpy()
        self.LbfgsPreTrnLambdas[1] = self.lmbda.data.cpu().numpy()
        self.LbfgsPreTrnFvus[1] = self.FVU_Calc(lib_ceofs=self.lmbda.data).cpu().numpy()

    def ADO_Training(self, iters:int, 
                LBFGS:bool=False, adms_epochs:int=250, lbfgs_epochs:int=150,
                schdlr:bool=False, Save_File_Name:str='Untitled.pt', chckFreqnt:int=100,
                train_alphas:np.ndarray=np.linspace(start=0.001, stop=0.5, num=10), **kwargs):

        if Save_File_Name==None or (not isinstance(Save_File_Name, str)):
            msg = ("The Save_File_Name argument was None or was not a string object the file name  \n"
                   "where the model will periodically be saved to is based off the day and time the \n" 
                   "class object was created. ")
            warnings.warn(message=msg, stacklevel=1)
            Save_File_Name = self.sv_fname
        Save_File_Name += ".pt"

        if not isinstance(chckFreqnt, int):
            msg = (f"The chckFreqnt input argument is not of the integer type and so \n"
                   "we will just set the frequency at which to check point the model \n"
                   "during training is every tenth of the epochs that the opcimization\n"
                   "has been set to run for")
            warnings.warn(message=msg, stacklevel=1)
            sv_frq = int(adms_epochs/10) if adms_epochs>=10 else 1
        else:
            sv_frq = chckFreqnt
        # Save_File_Path = "SavedModels/" + Save_File_Name

        ADO_iters = iters
        # ADO_alphas = train_alphas.cpu().numpy()

        tstFreq=10
        train_inputs = torch.from_numpy(self.data_dict['Train_Inputs']).to(device=self.device, dtype=self.data_type)
        train_targets = torch.from_numpy(self.data_dict['Train_Targets']).to(device=self.device, dtype=self.data_type)
        test_inputs = torch.from_numpy(self.data_dict['Test_Inputs']).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        test_targets = torch.from_numpy(self.data_dict['Test_Targets']).to(device=self.device, dtype=self.data_type)
        self.nDpnts = train_inputs.size(0)

        # n_inpts = Train_Inputs.shape[0]
        # tst_num_btchs=1

        # z_m = network.module.Lambda.data.shape[0]
        # z_n = Kfolds
        # # Do the ADO-esque optimization of the lambda function
        # all_lambdas = torch.zeros(size=(z_m, 1+iters+1), device=dvc, dtype=torch.float32)
        # # btchd_lmbdas = torch.zeros(size=(iters+1, z_m, z_n,), device=dvc)
        # btchd_lmbdas = torch.zeros(size=(iters+1, z_n, z_m), device=dvc, dtype=torch.float32)
        # slctd_lmbd_ids = torch.zeros(size=(iters+1,), dtype=int, device=dvc)
        # all_lambdas[:,0:1] = torch.clone(network.module.Lambda.data.detach())
        # btchd_lmbdas_lst = [torch.empty(size=(iters+1,z_n, z_m), dtype=torch.float32, device=dvc) for _ in range(4)]

        # store all the adams optim losses over the epochs and the ado iterations
        if not LBFGS:
            lbfgs_epochs = 0
        n_trgts = self.lmbda.data.size(-1)
        data_loss = torch.zeros(size=(iters, adms_epochs+lbfgs_epochs, n_trgts), device=self.device)
        pde_loss = torch.zeros(size=(iters, adms_epochs+lbfgs_epochs, n_trgts), device=self.device)
        tst_data_loss = torch.zeros(size=(iters, adms_epochs+lbfgs_epochs, n_trgts), device=self.device)
        tst_pde_loss = torch.zeros(size=(iters, adms_epochs+lbfgs_epochs, n_trgts), device=self.device)


        FnshdAdoTrn = False

        iter_lmbdas = torch.zeros(size=(1+iters+1, *self.lmbda.data.shape), device=self.device, dtype=self.data_type)
        iter_fvus = torch.zeros(size=(1+iters+1, n_trgts), device=self.device, dtype=self.data_type)
        iter_lmbdas[0] = self.lmbda.data
        iter_fvus[0] = self.FVU_Calc(lib_ceofs=self.lmbda.data)

        early_term = None
        if "Early_Term"in kwargs.keys():
            early_term = kwargs["Early_Term"]
            erly_trm_lmbda = torch.zeros(size=(iters, *self.lmbda.data.shape), device=self.device, dtype=self.data_type)
            erly_fvu = torch.zeros(size=(iters, n_trgts), device=self.device, dtype=self.data_type)
            rfe = RFE(alpha=torch.zeros(size=(1,),device=self.device, dtype=self.data_type), normalize=True, annealing_factor=1,)

        # tstFreq=20
        schdlr_freq=None

        for k in range(ADO_iters):

            nzs = [ [i.item() for i in self.lmbda.data[:,j].nonzero()] for j in range(n_trgts)]
            
            np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
            colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
            # if not colpnts.requires_grad:
            #     colpnts.requires_grad_(True)

            col_preds = self.net(colpnts)
            t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords).detach()
            lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts).detach()

            for i in range(n_trgts):
                iter_lmbdas[k+1, nzs[i], i:i+1] = self.sprs_slvr.solve(A=lib_evals[:,nzs[i]], b=t_prtls[:,i:i+1])
            self.sprs_slvr.cmplted_ADO_iter+=1
            self.lmbda.data = iter_lmbdas[k+1]
            iter_fvus[k+1] = self.FVU_Calc(lib_ceofs=self.lmbda.data)

            print(f"lmbda at ADO iter {k} is ...")
            print(self.lmbda.data)

            if early_term:
                for i in range(n_trgts):
                    erly_trm_lmbda[k, nzs[i], i:i+1] = rfe.solve(A=lib_evals[:,nzs[i]], b=t_prtls[:,i:i+1])
                erly_fvu[k] = self.FVU_Calc(lib_ceofs=erly_trm_lmbda[k])

            optim = torch.optim.Adam(self.net.parameters(), lr=0.001,)
            schdlr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.95, last_epoch=-1, ) 
            
            # np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
            for i in range(adms_epochs):
            #     data_btch_inds = partitionIndices(size=train_inputs.size(0), Nprts=n_btchs, rngSeed=None)
            #     pde_btch_inds = partitionIndices(size=self.N_col_pnts, Nprts=n_btchs, rngSeed=None)
            #     # colpnts = (ten_ub - ten_lb)*torch.rand((N_col,2), device=dvc, dtype=torch.float32, requires_grad=True) + ten_lb
            #     colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type)

                # for j in range(n_btchs):
                
                #     optim.zero_grad(set_to_none=True)
                #     self.net.train(True)

                #     dataL = torch.mean((self.net(train_inputs[data_btch_inds[j]]) - train_targets[data_btch_inds[j]])**2, dim=0)
                #     eq_pnts = colpnts[pde_btch_inds[j]].requires_grad_(True)
                #     col_preds = self.net(eq_pnts)
                #     t_prtls = Nth_temporal_prtls(values=col_preds, pts=eq_pnts, orders=self.tmprl_ords)
                #     lib_evals = self.lib_func.Calc(network=self.net, inpts=eq_pnts)
                #     pdeL = torch.mean((t_prtls - lib_evals@self.lmbda)**2, dim=0,)
                #     loss = torch.sum(dataL +  train_alphas[k]*pdeL)
                #     loss.backward()

                #     # tot_loss[i].add_( loss.detach() )
                #     data_loss[k, j].add_( dataL.detach() )
                #     pde_loss[k, j].add_( pdeL.detach() )

                #     optim.step()
                
                # data_loss[k,i].divide_(n_btchs)
                # pde_loss[k,i].divide_(n_btchs)

                colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
                optim.zero_grad(set_to_none=True)
                self.net.train(True)

                dataL = torch.mean((self.net(train_inputs) - train_targets)**2, dim=0)
                col_preds = self.net(colpnts)
                t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords)
                lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts)
                pdeL = torch.mean((t_prtls - lib_evals@self.lmbda)**2, dim=0,)
                loss = torch.sum(dataL + train_alphas[k]*pdeL )
                loss.backward()

                # tot_loss[k, i] = loss.detach()
                data_loss[k, i] = dataL.detach()
                pde_loss[k, i] = pdeL.detach()

                optim.step()
                
                if schdlr_freq and i>0 and i%schdlr_freq==0:
                    schdlr.step()
                    # schdlr.step(metrics=loss)
                self.net.eval()
                if i%tstFreq==0 or i+1==adms_epochs:
                    tst_dataL = torch.mean((self.net(test_inputs) - test_targets)**2, dim=0).detach_()
                    tst_col_preds = self.net(test_inputs)
                    tst_t_prtls = Nth_temporal_prtls(values=tst_col_preds, pts=test_inputs, orders=self.tmprl_ords).detach_()
                    tst_lib_evals = self.lib_func.Calc(network=self.net, inpts=test_inputs)
                    tst_pdeL = torch.mean((tst_t_prtls - tst_lib_evals@self.lmbda)**2, dim=0,).detach_()
                    # ts_loss = torch.sum(tst_dataL + train_alphas[k]*tst_pdeL )
                    # tst_loss[k, i//tstFreq] = ts_loss.detach()
                    tst_data_loss[k, i] = tst_dataL.detach()
                    tst_pde_loss[k, i] = tst_pdeL.detach()

        # nzs = [ [i for i in range(self.lmbda.data.size(0))] for j in range(n_trgts)]
        nzs = [ [i.item() for i in self.lmbda.data[:,j].nonzero()] for j in range(n_trgts)]
        # nzs = [ [i.item() for i in iter_lmbdas[1,:,j].nonzero()] for j in range(n_trgts)]

        # # colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        # np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        # colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        # if not colpnts.requires_grad:
        #     colpnts.requires_grad_(True)

        # col_preds = self.net(colpnts)
        # t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords).detach()
        # lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts).detach()

        # for i in range(n_trgts):
        #     iter_lmbdas[-1, nzs[i], i:i+1] = self.sprs_slvr.solve(A=lib_evals[:,nzs[i]], b=t_prtls[:,i:i+1])

        ##################################################################################################################################################

        model_losses = []
        model_complxtys = []
        model_scores = []

        # get the evaluation of the library functions over the collocation points for later. 
        np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        col_preds = self.net(colpnts)
        t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords).detach()
        lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts).detach()

        tst_col_preds = self.net(test_inputs)
        tst_t_prtls = Nth_temporal_prtls(values=tst_col_preds, pts=test_inputs, orders=self.tmprl_ords).detach_()
        tst_lib_evals = self.lib_func.Calc(network=self.net, inpts=test_inputs).detach()
        # need the complexity of each library term for choosing the best equation bellow. 
        lib_complexities = self.lib_func.Get_Lib_Complexities()
        n_lib_trms = self.lmbda.shape[0]
        # need to store which equations were considered throughout the optimization steps and here. 
        eqs_ids = [[np.arange(n_lib_trms)] for _ in range(n_trgts)]
        pre_ids = [np.arange(n_lib_trms) for _ in range(n_trgts)]
        eq_complexities = [[lib_complexities.sum().item() + n_lib_trms - 1] for _ in range(n_trgts)]
        slctd_eqs_indices = []

        regress_alphas = self.sprs_slvr.alpha

        for q in range(n_trgts):
            # recall each of the n targeted equation might have a different number of functions selected after optimization step. 
            n_act_funcs = len(nzs[q])
            # since the library should be greated culled after the K optim step, just do a greedy sparse regression to 
            # consider remaining models until the Null/Empty model. These are the extra lambdas.
            extra_lmbdas = Specified_Optim_Regres(A=lib_evals, b=t_prtls, activ_ids=nzs[q], alpha=regress_alphas).cpu().numpy()
            # last/remaining models
            lst_models = np.reshape(extra_lmbdas, shape=(n_act_funcs*n_lib_trms + n_lib_trms,), order="F").reshape((n_act_funcs+1,n_lib_trms,1), order="C")
            # all models considered throughout tryining/optim. step. 
            ado_models = np.concat((iter_lmbdas[:-1, :,q:q+1].detach().cpu().numpy(), lst_models), axis=0)
            # get the complexity of each distinct model as it is possible to have some combination of the lib. funcs repeated 
            # from one optim step to the next.
            for lm in ado_models:
                idx = np.nonzero(lm[:,0])[0]    
                if idx.size==pre_ids[q].size:
                    continue
                pre_ids[q] = idx
                eqs_ids[q].append(idx)
                eq_complexities[q].append(lib_complexities[idx].sum().item() + idx.size - 1)
            # number of distinct library func. combinations. 
            n_eqs = len(eqs_ids[q])
            eq_losses = torch.empty(size=(n_eqs,), device=self.device, dtype=self.data_type)
            for j in range(n_eqs):
                # Now get the equation coefficients for the each model and the loss on the test library. 
                # eq_coefs = torch.linalg.lstsq(lib_evals[:,eqs_ids[q][j]],t_prtls[:,q:q+1])[0]
                eq_coefs = solve_svd(lib_evals[:,eqs_ids[q][j]],t_prtls[:,q:q+1], alphas=regress_alphas)
                eq_losses[j] = torch.mean( (tst_lib_evals[:,eqs_ids[q][j]]@eq_coefs -tst_t_prtls[:,q:q+1] ) **2 )
            
            eq_lsses = eq_losses.cpu().numpy()
            model_complexities = np.asarray(eq_complexities[0])
            # flippig the order of the matrix valyes since want the complexity to increase as we go down across (-->) matrix
            # so model loss will then decrease as we get more and more complex equation (reading complexity vs loss curver right left. )
            flppd_eq_losses = np.flipud(eq_lsses)
            model_losses.append(flppd_eq_losses)
            flppd_model_cmplx = np.flipud(model_complexities)
            model_complxtys.append(flppd_model_cmplx)
            # if this is not done, the complexity of the null/empty model is -1 : so make it 0 as no model had no complexity. 
            # flppd_model_cmplx[0] = np.int64(0)
            scores = -np.log(flppd_eq_losses[1:]/flppd_eq_losses[:-1]) / (flppd_model_cmplx[1:] - flppd_model_cmplx[:-1])
            model_scores.append(scores)
            max_scr_idx = scores.argmax()+1
            slctd_eqs_indices.append(eqs_ids[q][len(eqs_ids[0]) - max_scr_idx-1])

        del pre_ids
        
        for i in range(n_trgts):
            # iter_lmbdas[-1, nzs[i], i:i+1] = self.sprs_slvr.solve(A=lib_evals[:,nzs[i]], b=t_prtls[:,i:i+1])
            iter_lmbdas[-1, slctd_eqs_indices[i], i:i+1] = solve_svd(lib_evals[:,slctd_eqs_indices[i]],t_prtls[:,i:i+1], alphas=regress_alphas)

        ##################################################################################################################################################


        self.lmbda.data = iter_lmbdas[-1]
        iter_fvus[-1] = self.FVU_Calc(lib_ceofs=self.lmbda.data)
        if early_term:
            self.earl_lmbds = erly_trm_lmbda.detach().cpu().numpy()
            self.earl_fvus = erly_fvu.detach().cpu().numpy()
        self._ADO_lambdas = iter_lmbdas.detach().cpu().numpy()
        self._ADO_FVUs = iter_fvus.cpu().numpy()
        self._ADO_iters = iters
        self._ADO_epchs = adms_epochs + lbfgs_epochs
        self._ADO_alphas = train_alphas
        self._ADOtrnDataLs = data_loss.detach().cpu().numpy()
        self._ADOtstDataLs = tst_data_loss.detach().cpu().numpy()
        self._ADOtrnColloLs = pde_loss.detach().cpu().numpy()
        self._ADOtstColloLs = tst_pde_loss.detach().cpu().numpy()
        tple = (train_alphas.shape[0],1,1)
        
        if isinstance(train_alphas, np.ndarray):
            self._ADOTrnLosses = self._ADOtrnDataLs + train_alphas.reshape(tple) * self._ADOtrnColloLs
            self._ADOTstLosses = self._ADOtstDataLs + train_alphas.reshape(tple) * self._ADOtstColloLs
        elif isinstance(train_alphas, torch.Tensor):
            self._ADOTrnLosses = self._ADOtrnDataLs + train_alphas.detach().cpu().numpy().reshape(tple) * self._ADOtrnColloLs
            self._ADOTstLosses = self._ADOtstDataLs + train_alphas.detach().cpu().numpy().reshape(tple) * self._ADOtstColloLs
        self._model_losses = model_losses
        self._model_complxtys = model_complxtys
        self._model_scores = model_scores

    def Batched_ADO_Training(self, iters:int, 
                LBFGS:bool=False, adms_epochs:int=250, lbfgs_epochs:int=150, b_sizes:int=1,
                schdlr:bool=False, Save_File_Name:str='Untitled.pt', chckFreqnt:int=100,
                train_alphas:np.ndarray=np.linspace(start=0.001, stop=0.5, num=10), **kwargs):

        if Save_File_Name==None or (not isinstance(Save_File_Name, str)):
            msg = ("The Save_File_Name argument was None or was not a string object the file name  \n"
                   "where the model will periodically be saved to is based off the day and time the \n" 
                   "class object was created. ")
            warnings.warn(message=msg, stacklevel=1)
            Save_File_Name = self.sv_fname
        Save_File_Name += ".pt"

        if not isinstance(chckFreqnt, int):
            msg = (f"The chckFreqnt input argument is not of the integer type and so \n"
                   "we will just set the frequency at which to check point the model \n"
                   "during training is every tenth of the epochs that the opcimization\n"
                   "has been set to run for")
            warnings.warn(message=msg, stacklevel=1)
            sv_frq = int(adms_epochs/10) if adms_epochs>=10 else 1
        else:
            sv_frq = chckFreqnt
        # Save_File_Path = "SavedModels/" + Save_File_Name

        ADO_iters = iters
        # ADO_alphas = train_alphas.cpu().numpy()

        

        tstFreq=10

        # train_inputs = torch.from_numpy(self.data_dict['Train_Inputs']).to(device=self.device, dtype=self.data_type)
        # train_targets = torch.from_numpy(self.data_dict['Train_Targets']).to(device=self.device, dtype=self.data_type)
        # test_inputs = torch.from_numpy(self.data_dict['Test_Inputs']).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        # test_targets = torch.from_numpy(self.data_dict['Test_Targets']).to(device=self.device, dtype=self.data_type)

        train_inputs = torch.from_numpy(self.data_dict['Train_Inputs']).to(device=self.device, dtype=self.data_type)
        train_targets = torch.from_numpy(self.data_dict['Train_Targets']).to(device=self.device, dtype=self.data_type)
        test_inputs = torch.from_numpy(self.data_dict['Test_Inputs']).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        test_targets = torch.from_numpy(self.data_dict['Test_Targets']).to(device=self.device, dtype=self.data_type)
        self.nDpnts = train_inputs.size(0)
        
        n_trn_points = train_inputs.size(0)
        n_dims = train_inputs.size(1)
        N_btchs = int(np.ceil(n_trn_points/b_sizes))
        n_col_pnts = int(np.around(self.N_col_pnts/n_trn_points) * n_trn_points)
        if n_col_pnts==0:
            n_col_pnts = n_trn_points
        q = int(n_col_pnts / n_trn_points)
        
        
        # np_colpnts = self.col_pnts_smplr.sample(n_pnts=n_col_pnts)


        # n_inpts = Train_Inputs.shape[0]
        # tst_num_btchs=1

        # z_m = network.module.Lambda.data.shape[0]
        # z_n = Kfolds
        # # Do the ADO-esque optimization of the lambda function
        # all_lambdas = torch.zeros(size=(z_m, 1+iters+1), device=dvc, dtype=torch.float32)
        # # btchd_lmbdas = torch.zeros(size=(iters+1, z_m, z_n,), device=dvc)
        # btchd_lmbdas = torch.zeros(size=(iters+1, z_n, z_m), device=dvc, dtype=torch.float32)
        # slctd_lmbd_ids = torch.zeros(size=(iters+1,), dtype=int, device=dvc)
        # all_lambdas[:,0:1] = torch.clone(network.module.Lambda.data.detach())
        # btchd_lmbdas_lst = [torch.empty(size=(iters+1,z_n, z_m), dtype=torch.float32, device=dvc) for _ in range(4)]

        # store all the adams optim losses over the epochs and the ado iterations
        if not LBFGS:
            lbfgs_epochs = 0
        n_trgts = self.lmbda.data.size(-1)
        data_loss = torch.zeros(size=(iters, adms_epochs+lbfgs_epochs, n_trgts), device=self.device)
        pde_loss = torch.zeros(size=(iters, adms_epochs+lbfgs_epochs, n_trgts), device=self.device)
        tst_data_loss = torch.zeros(size=(iters, adms_epochs+lbfgs_epochs, n_trgts), device=self.device)
        tst_pde_loss = torch.zeros(size=(iters, adms_epochs+lbfgs_epochs, n_trgts), device=self.device)


        FnshdAdoTrn = False

        iter_lmbdas = torch.zeros(size=(1+iters+1, *self.lmbda.data.shape), device=self.device, dtype=self.data_type)
        iter_fvus = torch.zeros(size=(1+iters+1, n_trgts), device=self.device, dtype=self.data_type)
        iter_lmbdas[0] = self.lmbda.data
        iter_fvus[0] = self.FVU_Calc(lib_ceofs=self.lmbda.data)

        early_term = None
        if "Early_Term"in kwargs.keys():
            early_term = kwargs["Early_Term"]
            erly_trm_lmbda = torch.zeros(size=(iters, *self.lmbda.data.shape), device=self.device, dtype=self.data_type)
            erly_fvu = torch.zeros(size=(iters, n_trgts), device=self.device, dtype=self.data_type)
            rfe = RFE(alpha=torch.zeros(size=(1,),device=self.device, dtype=self.data_type), normalize=True, annealing_factor=1,)

        # tstFreq=20
        schdlr_freq=None
        # print(f"self.N_col_pnts = {self.N_col_pnts}")
        # print(f"n_col_pnts = {n_col_pnts}")

        for k in range(ADO_iters):

            nzs = [ [i.item() for i in self.lmbda.data[:,j].nonzero()] for j in range(n_trgts)]
            
            np_colpnts = self.col_pnts_smplr.sample(n_pnts=n_col_pnts)
            colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
            # if not colpnts.requires_grad:
            #     colpnts.requires_grad_(True)

            col_preds = self.net(colpnts)
            t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords).detach()
            lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts).detach()

            for i in range(n_trgts):
                iter_lmbdas[k+1, nzs[i], i:i+1] = self.sprs_slvr.solve(A=lib_evals[:,nzs[i]], b=t_prtls[:,i:i+1])
            self.sprs_slvr.cmplted_ADO_iter+=1
            self.lmbda.data = iter_lmbdas[k+1]
            iter_fvus[k+1] = self.FVU_Calc(lib_ceofs=self.lmbda.data)

            print(f"lmbda at ADO iter {k} is ...")
            print(self.lmbda.data)

            if early_term:
                for i in range(n_trgts):
                    erly_trm_lmbda[k, nzs[i], i:i+1] = rfe.solve(A=lib_evals[:,nzs[i]], b=t_prtls[:,i:i+1])
                erly_fvu[k] = self.FVU_Calc(lib_ceofs=erly_trm_lmbda[k])

            optim = torch.optim.Adam(self.net.parameters(), lr=0.001,)
            schdlr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.95, last_epoch=-1, ) 
            # print(f"np_colpnts.shape = {np_colpnts.shape}")
            if q==1:
                colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type)
            else:
                colpnts = torch.from_numpy(np_colpnts.reshape((n_trn_points, q, n_dims))).to(device=self.device, dtype=self.data_type)

            dset = TensorDataset (train_inputs, train_targets, colpnts)
            loader = DataLoader(dataset=dset, batch_size=b_sizes,shuffle=True, pin_memory=False)
            
            # np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
            for i in range(adms_epochs):
                self.net.train(True)
                for _, (tnn_ins, trn_trgts, cols) in enumerate(loader):
                    eq_pnts = cols.view((-1, n_dims)).requires_grad_(True)
                    optim.zero_grad(set_to_none=True)

                    dataL = torch.mean((self.net(tnn_ins) - trn_trgts)**2, dim=0)
                    col_preds = self.net(eq_pnts)
                    t_prtls = Nth_temporal_prtls(values=col_preds, pts=eq_pnts, orders=self.tmprl_ords)
                    lib_evals = self.lib_func.Calc(network=self.net, inpts=eq_pnts)
                    pdeL = torch.mean((t_prtls - lib_evals@self.lmbda)**2, dim=0,)
                    loss = torch.sum(dataL + train_alphas[k]*pdeL )
                    loss.backward()

                    # tot_loss[k, i].add_( loss.detach() )
                    data_loss[k, i].add_( dataL.detach() )
                    pde_loss[k, i].add_( pdeL.detach() )

                    optim.step()
                    del dataL, col_preds, t_prtls, lib_evals, pdeL, loss
                # tot_loss[k, i].divide_(N_btchs)
                data_loss[k, i].divide_(N_btchs)
                pde_loss[k, i].divide_(N_btchs)

                # colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
                # optim.zero_grad(set_to_none=True)
                # self.net.train(True)

                # dataL = torch.mean((self.net(train_inputs) - train_targets)**2, dim=0)
                # col_preds = self.net(colpnts)
                # t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords)
                # lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts)
                # pdeL = torch.mean((t_prtls - lib_evals@self.lmbda)**2, dim=0,)
                # loss = torch.sum(dataL + train_alphas[k]*pdeL )
                # loss.backward()

                # # tot_loss[k, i] = loss.detach()
                # data_loss[k, i] = dataL.detach()
                # pde_loss[k, i] = pdeL.detach()

                # optim.step()
                
                if schdlr_freq and i>0 and i%schdlr_freq==0:
                    schdlr.step()
                    # schdlr.step(metrics=loss)
                
                if i%tstFreq==0 or i+1==adms_epochs:
                    self.net.eval()
                    tst_dataL = torch.mean((self.net(test_inputs) - test_targets)**2, dim=0).detach_()
                    tst_col_preds = self.net(test_inputs)
                    tst_t_prtls = Nth_temporal_prtls(values=tst_col_preds, pts=test_inputs, orders=self.tmprl_ords).detach_()
                    tst_lib_evals = self.lib_func.Calc(network=self.net, inpts=test_inputs)
                    tst_pdeL = torch.mean((tst_t_prtls - tst_lib_evals@self.lmbda)**2, dim=0,).detach_()
                    ts_loss = torch.sum(tst_dataL + train_alphas[k]*tst_pdeL )
                    tst_data_loss[k, i] = tst_dataL.detach()
                    tst_pde_loss[k, i] = tst_pdeL.detach()
                    
            del dset, loader

        # nzs = [ [i for i in range(self.lmbda.data.size(0))] for j in range(n_trgts)]
        nzs = [ [i.item() for i in self.lmbda.data[:,j].nonzero()] for j in range(n_trgts)]
        # nzs = [ [i.item() for i in iter_lmbdas[1,:,j].nonzero()] for j in range(n_trgts)]
        # colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        # np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        # colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        # if not colpnts.requires_grad:
        #     colpnts.requires_grad_(True)

        # col_preds = self.net(colpnts)
        # t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords).detach()
        # lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts).detach()

        # for i in range(n_trgts):
        #     iter_lmbdas[-1, nzs[i], i:i+1] = self.sprs_slvr.solve(A=lib_evals[:,nzs[i]], b=t_prtls[:,i:i+1])


        ##################################################################################################################################################

        model_losses = []
        model_complxtys = []
        model_scores = []

        # get the evaluation of the library functions over the collocation points for later. 
        np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        col_preds = self.net(colpnts)
        t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords).detach()
        lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts).detach()

        tst_col_preds = self.net(test_inputs)
        tst_t_prtls = Nth_temporal_prtls(values=tst_col_preds, pts=test_inputs, orders=self.tmprl_ords).detach_()
        tst_lib_evals = self.lib_func.Calc(network=self.net, inpts=test_inputs).detach()
        # need the complexity of each library term for choosing the best equation bellow. 
        lib_complexities = self.lib_func.Get_Lib_Complexities()
        n_lib_trms = self.lmbda.shape[0]
        # need to store which equations were considered throughout the optimization steps and here. 
        eqs_ids = [[np.arange(n_lib_trms)] for _ in range(n_trgts)]
        pre_ids = [np.arange(n_lib_trms) for _ in range(n_trgts)]
        eq_complexities = [[lib_complexities.sum().item() + n_lib_trms - 1] for _ in range(n_trgts)]
        slctd_eqs_indices = []

        regress_alphas = self.sprs_slvr.alpha

        for q in range(n_trgts):
            # recall each of the n targeted equation might have a different number of functions selected after optimization step. 
            n_act_funcs = len(nzs[q])
            # since the library should be greated culled after the K optim step, just do a greedy sparse regression to 
            # consider remaining models until the Null/Empty model. These are the extra lambdas.
            extra_lmbdas = Specified_Optim_Regres(A=lib_evals, b=t_prtls, activ_ids=nzs[q], alpha=regress_alphas).cpu().numpy()
            # last/remaining models
            lst_models = np.reshape(extra_lmbdas, shape=(n_act_funcs*n_lib_trms + n_lib_trms,), order="F").reshape((n_act_funcs+1,n_lib_trms,1), order="C")
            # all models considered throughout tryining/optim. step. 
            ado_models = np.concat((iter_lmbdas[:-1, :,q:q+1].detach().cpu().numpy(), lst_models), axis=0)
            # get the complexity of each distinct model as it is possible to have some combination of the lib. funcs repeated 
            # from one optim step to the next.
            for lm in ado_models:
                idx = np.nonzero(lm[:,0])[0]    
                if idx.size==pre_ids[q].size:
                    continue
                pre_ids[q] = idx
                eqs_ids[q].append(idx)
                eq_complexities[q].append(lib_complexities[idx].sum().item() + idx.size - 1)
            # number of distinct library func. combinations. 
            n_eqs = len(eqs_ids[q])
            eq_losses = torch.empty(size=(n_eqs,), device=self.device, dtype=self.data_type)
            for j in range(n_eqs):
                # Now get the equation coefficients for the each model and the loss on the test library. 
                # eq_coefs = torch.linalg.lstsq(lib_evals[:,eqs_ids[q][j]],t_prtls[:,q:q+1])[0]
                eq_coefs = solve_svd(lib_evals[:,eqs_ids[q][j]],t_prtls[:,q:q+1], alphas=regress_alphas)
                eq_losses[j] = torch.mean( (tst_lib_evals[:,eqs_ids[q][j]]@eq_coefs -tst_t_prtls[:,q:q+1] ) **2 )
            
            eq_lsses = eq_losses.cpu().numpy()
            model_complexities = np.asarray(eq_complexities[0])
            # flippig the order of the matrix valyes since want the complexity to increase as we go down across (-->) matrix
            # so model loss will then decrease as we get more and more complex equation (reading complexity vs loss curver right left. )
            flppd_eq_losses = np.flipud(eq_lsses)
            model_losses.append(flppd_eq_losses)
            flppd_model_cmplx = np.flipud(model_complexities)
            model_complxtys.append(flppd_model_cmplx)
            # if this is not done, the complexity of the null/empty model is -1 : so make it 0 as no model had no complexity. 
            # flppd_model_cmplx[0] = np.int64(0)
            scores = -np.log(flppd_eq_losses[1:]/flppd_eq_losses[:-1]) / (flppd_model_cmplx[1:] - flppd_model_cmplx[:-1])
            model_scores.append(scores)
            max_scr_idx = scores.argmax()+1
            slctd_eqs_indices.append(eqs_ids[q][len(eqs_ids[0]) - max_scr_idx-1])

        del pre_ids
        
        for i in range(n_trgts):
            # iter_lmbdas[-1, nzs[i], i:i+1] = self.sprs_slvr.solve(A=lib_evals[:,nzs[i]], b=t_prtls[:,i:i+1])
            iter_lmbdas[-1, slctd_eqs_indices[i], i:i+1] = solve_svd(lib_evals[:,slctd_eqs_indices[i]],t_prtls[:,i:i+1], alphas=regress_alphas)

        ##################################################################################################################################################


        self.lmbda.data = iter_lmbdas[-1]
        iter_fvus[-1] = self.FVU_Calc(lib_ceofs=self.lmbda.data)
        if early_term:
            self.earl_lmbds = erly_trm_lmbda.detach().cpu().numpy()
            self.earl_fvus = erly_fvu.detach().cpu().numpy()
        self._ADO_lambdas = iter_lmbdas.detach().cpu().numpy()
        self._ADO_FVUs = iter_fvus.cpu().numpy()
        self._ADO_iters = iters
        self._ADO_epchs = adms_epochs + lbfgs_epochs
        self._ADO_alphas = train_alphas
        self._ADOtrnDataLs = data_loss.detach().cpu().numpy()
        self._ADOtstDataLs = tst_data_loss.detach().cpu().numpy()
        self._ADOtrnColloLs = pde_loss.detach().cpu().numpy()
        self._ADOtstColloLs = tst_pde_loss.detach().cpu().numpy()
        tple = (train_alphas.shape[0],1,1)
        
        if isinstance(train_alphas, np.ndarray):
            self._ADOTrnLosses = self._ADOtrnDataLs + train_alphas.reshape(tple) * self._ADOtrnColloLs
            self._ADOTstLosses = self._ADOtstDataLs + train_alphas.reshape(tple) * self._ADOtstColloLs
        elif isinstance(train_alphas, torch.Tensor):
            self._ADOTrnLosses = self._ADOtrnDataLs + train_alphas.detach().cpu().numpy().reshape(tple) * self._ADOtrnColloLs
            self._ADOTstLosses = self._ADOtstDataLs + train_alphas.detach().cpu().numpy().reshape(tple) * self._ADOtstColloLs
        self._model_losses = model_losses
        self._model_complxtys = model_complxtys
        self._model_scores = model_scores

    def Adams_PostTraining(self, alpha:float=1.0, epochs:int=1000, lrn_rt:float=0.001, lp_ord:float=1.0,
                Save_File_Name:str=None, tstFreq:int=100, schdlr_freq:int=0,
                betas:Tuple=(0.9, 0.99), eps:float=1e-8, wght_dcy:float=0, amsgrad:bool=False):

        if Save_File_Name==None or (not isinstance(Save_File_Name, str)):
            msg = ("The Save_File_Name argument was None or was not a string object the file name  \n"
                   "where the model will periodically be saved to is based off the day and time the \n" 
                   "class object was created. ")
            warnings.warn(message=msg, stacklevel=1)
            Save_File_Name = self.sv_fname
        Save_File_Name += ".pt"

        self.AdamsPstTrnLambdas = np.empty(shape=(2, *self.lmbda.data.shape), dtype=float)
        self.AdamsPstTrnFvus = np.empty(shape=(2, self.lmbda.data.shape[1]), dtype=float)
        self.AdamsPstTrnLambdas[0] = self.lmbda.data.cpu().numpy()
        self.AdamsPstTrnFvus[0] = self.FVU_Calc(lib_ceofs=self.lmbda.data).cpu().numpy()

        n_trgts = self.lmbda.size(-1)
        nzs = [ [i.item() for i in self.lmbda.data[:,j].nonzero()] for j in range(n_trgts)]

        optim = torch.optim.Adam(self.net.parameters(), lr=lrn_rt, betas=betas, eps=eps, weight_decay=wght_dcy, amsgrad=amsgrad)
        optim.param_groups[0]['params'].append(self.lmbda)
        schdlr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.9, last_epoch=-1, ) 
        # schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10,
        #                                         threshold=0.001, threshold_mode='rel', cooldown=1, min_lr=0, 
        #                                             eps=1e-08, )

        train_inputs = torch.from_numpy(self.data_dict['Train_Inputs']).to(device=self.device, dtype=self.data_type)
        train_targets = torch.from_numpy(self.data_dict['Train_Targets']).to(device=self.device, dtype=self.data_type)
        test_inputs = torch.from_numpy(self.data_dict['Test_Inputs']).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        test_targets = torch.from_numpy(self.data_dict['Test_Targets']).to(device=self.device, dtype=self.data_type)
        np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        self.nDpnts = train_inputs.size(0)
        b_sizes = train_inputs.size(0)
        
        tot_loss = torch.zeros(size=(epochs, 1), device=self.device,)
        data_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)
        pde_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)
        lp_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)

        tst_loss = torch.zeros(size=(epochs, 1), device=self.device,)
        tst_data_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)
        tst_pde_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)

        # lmbda_lst = []

        for i in range(epochs):
            self.net.train(True)
            # lmbda_lst.append(torch.clone(self.lmbda.data.detach()))
            colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
                
            optim.zero_grad(set_to_none=True)
            self.net.train(True)

            dataL = torch.mean((self.net(train_inputs) - train_targets)**2, dim=0)
            col_preds = self.net(colpnts)
            t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords)
            lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts)
            pdeL = torch.zeros(size=(n_trgts,), device=self.device, dtype=self.data_type)
            # lp_lss = Torch_Lp_Loss(Xis=self.lmbda.data, p=lp_ord)
            lp_lss = Torch_Lp_Loss(Xis=self.lmbda, p=lp_ord)
            for j in range(n_trgts): 
                pdeL[j] = torch.mean((t_prtls[:,j:j+1] - lib_evals[:, nzs[j]] @ self.lmbda[nzs[j],j:j+1])**2, dim=0)
            # pdeL = torch.mean((t_prtls - lib_evals@self.lmbda)**2, dim=0,)
            loss = torch.sum(dataL + alpha*pdeL)
            loss.backward()

            tot_loss[i] = loss.detach()
            data_loss[i] = dataL.detach()
            pde_loss[i] = pdeL.detach()
            lp_loss[i] = lp_lss.detach()

            optim.step()
            if schdlr_freq and i>0 and i%schdlr_freq==0:
                schdlr.step()
                # schdlr.step(metrics=loss)
            self.net.eval()
            if i%tstFreq==0 or i+1==epochs:
                tst_dataL = torch.mean((self.net(test_inputs) - test_targets)**2, dim=0).detach_()
                tst_col_preds = self.net(test_inputs)
                tst_t_prtls = Nth_temporal_prtls(values=tst_col_preds, pts=test_inputs, orders=self.tmprl_ords).detach_()
                tst_lib_evals = self.lib_func.Calc(network=self.net, inpts=test_inputs)
                tst_pdeL = torch.mean((tst_t_prtls - tst_lib_evals@self.lmbda)**2, dim=0,).detach_()
                # tst_pdeL = torch.zeros(size=(n_trgts,), device=self.device, dtype=self.data_type)
                # for j in range(n_trgts): tst_pdeL[j] = torch.mean((tst_t_prtls[:,j:j+1] - tst_lib_evals[:, nzs[j]] @ tst_lib_evals[nzs[j],j:j+1]**2), dim=0).detach_()
                ts_loss = torch.sum(tst_dataL + alpha*tst_pdeL).detach_()
                tst_loss[i] = ts_loss.detach()
                tst_data_loss[i] = tst_dataL.detach()
                tst_pde_loss[i] = tst_pdeL.detach()

    

        optim.zero_grad(set_to_none=True)
        print('Finished Adams Post Training')
        del train_inputs, train_targets, test_inputs, test_targets
        self._PstTrnAdamsEpochs = epochs
        self._PstTrnAdamsLrnRt = lrn_rt
        self._PstTrnAdamsAlpha = alpha
        self.AdamsPstTrnLoss = tot_loss.cpu().numpy()
        self.AdamsPstTstLoss = tst_loss.cpu().numpy()
        self.AdamsPstTrnDataLoss = data_loss.cpu().numpy()
        self.AdamsPstTstDataLoss = tst_data_loss.cpu().numpy()
        self.AdamsPstTrnEqLoss = pde_loss.cpu().numpy()
        self.AdamsPstTstEqLoss = tst_pde_loss.cpu().numpy()
        self.AdamsPstLpLoss = lp_loss.cpu().numpy()
        self.AdamsPstTrnLambdas[1] = self.lmbda.data.cpu().numpy()
        # self.AdamsPstTrnIterLambdas = torch.stack(lmbda_lst, dim=0).cpu().numpy()
        self.AdamsPstTrnFvus[1] = self.FVU_Calc(lib_ceofs=self.lmbda.data).cpu().numpy()
    
    def Batched_Adams_PostTraining(self, alpha:float=0.5, epochs:int=1000, lrn_rt:float=0.001,  
                b_sizes:int=1, Save_File_Name:str=None, tstFreq:int=100, schdlr_freq:int=0,
                betas:Tuple=(0.9, 0.99), eps:float=1e-8, wght_dcy:float=0, amsgrad:bool=False):

        if Save_File_Name==None or (not isinstance(Save_File_Name, str)):
            msg = ("The Save_File_Name argument was None or was not a string object the file name  \n"
                   "where the model will periodically be saved to is based off the day and time the \n" 
                   "class object was created. ")
            warnings.warn(message=msg, stacklevel=1)
            Save_File_Name = self.sv_fname
        Save_File_Name += ".pt"

        self.AdamsPstTrnLambdas = np.empty(shape=(2, *self.lmbda.data.shape), dtype=float)
        self.AdamsPstTrnFvus = np.empty(shape=(2, self.lmbda.data.shape[1]), dtype=float)
        self.AdamsPstTrnLambdas[0] = self.lmbda.data.cpu().numpy()
        self.AdamsPstTrnFvus[0] = self.FVU_Calc(lib_ceofs=self.lmbda.data).cpu().numpy()

        n_trgts = self.lmbda.size(-1)
        nzs = [ [i.item() for i in self.lmbda.data[:,j].nonzero()] for j in range(n_trgts)]

        optim = torch.optim.Adam(self.net.parameters(), lr=lrn_rt, betas=betas, eps=eps, weight_decay=wght_dcy, amsgrad=amsgrad)
        optim.param_groups[0]['params'].append(self.lmbda)
        schdlr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.9, last_epoch=-1, ) 
        # schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10,
        #                                         threshold=0.001, threshold_mode='rel', cooldown=1, min_lr=0, 
        #                                             eps=1e-08, )

        train_inputs = torch.from_numpy(self.data_dict['Train_Inputs']).to(device=self.device, dtype=self.data_type)
        train_targets = torch.from_numpy(self.data_dict['Train_Targets']).to(device=self.device, dtype=self.data_type)
        test_inputs = torch.from_numpy(self.data_dict['Test_Inputs']).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        test_targets = torch.from_numpy(self.data_dict['Test_Targets']).to(device=self.device, dtype=self.data_type)
        self.nDpnts = train_inputs.size(0)

        n_trn_points = train_inputs.size(0)
        n_dims = train_inputs.size(1)
        N_btchs = int(np.ceil(n_trn_points/b_sizes))
        n_col_pnts = int(np.around(self.N_col_pnts/n_trn_points) * n_trn_points)
        if n_col_pnts==0:
            n_col_pnts = n_trn_points
        k = int(n_col_pnts / n_trn_points)
        
        
        np_colpnts = self.col_pnts_smplr.sample(n_pnts=n_col_pnts)
        if k==1:
            colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type)
        else:
            colpnts = torch.from_numpy(np_colpnts.reshape((n_trn_points, k, n_dims))).to(device=self.device, dtype=self.data_type)

        dset = TensorDataset (train_inputs, train_targets, colpnts)
        loader = DataLoader(dataset=dset, batch_size=b_sizes,shuffle=True, pin_memory=False)
        
        tot_loss = torch.zeros(size=(epochs, 1), device=self.device,)
        data_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)
        pde_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)

        tst_loss = torch.zeros(size=(epochs, 1), device=self.device,)
        tst_data_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)
        tst_pde_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)


        for i in range(epochs):

            self.net.train(True)
            for _, (tnn_ins, trn_trgts, cols) in enumerate(loader):
                eq_pnts = cols.view((-1, n_dims)).requires_grad_(True)
                optim.zero_grad(set_to_none=True)

                dataL = torch.mean((self.net(tnn_ins) - trn_trgts)**2, dim=0)
                col_preds = self.net(eq_pnts)
                t_prtls = Nth_temporal_prtls(values=col_preds, pts=eq_pnts, orders=self.tmprl_ords)
                lib_evals = self.lib_func.Calc(network=self.net, inpts=eq_pnts)
                pdeL = torch.zeros(size=(n_trgts,), device=self.device, dtype=self.data_type)
                for j in range(n_trgts): 
                    pdeL[j] = torch.mean((t_prtls[:,j:j+1] - lib_evals[:, nzs[j]] @ self.lmbda[nzs[j],j:j+1])**2, dim=0)
                # pdeL = torch.mean((t_prtls - lib_evals@self.lmbda)**2, dim=0,)
                loss = torch.sum(dataL + alpha*pdeL)
                loss.backward()

                tot_loss[i].add_( loss.detach() )
                data_loss[i].add_( dataL.detach() )
                pde_loss[i].add_( pdeL.detach() )

                optim.step()
                del dataL, col_preds, t_prtls, lib_evals, pdeL, loss
            
            tot_loss[i].divide_(N_btchs)
            data_loss[i].divide_(N_btchs)
            pde_loss[i].divide_(N_btchs)

            # colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
                
            # optim.zero_grad(set_to_none=True)
            # self.net.train(True)

            # dataL = torch.mean((self.net(train_inputs) - train_targets)**2, dim=0)
            # col_preds = self.net(colpnts)
            # t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords)
            # lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts)
            # pdeL = torch.zeros(size=(n_trgts,), device=self.device, dtype=self.data_type)
            # for j in range(n_trgts): 
            #     pdeL[j] = torch.mean((t_prtls[:,j:j+1] - lib_evals[:, nzs[j]] @ self.lmbda[nzs[j],j:j+1])**2, dim=0)
            # # pdeL = torch.mean((t_prtls - lib_evals@self.lmbda)**2, dim=0,)
            # loss = torch.sum(dataL + alpha*pdeL)
            # loss.backward()

            # tot_loss[i] = loss.detach()
            # data_loss[i] = dataL.detach()
            # pde_loss[i] = pdeL.detach()
            
            # optim.step()

            if schdlr_freq and i>0 and i%schdlr_freq==0:
                schdlr.step()
                # schdlr.step(metrics=loss)
            self.net.eval()
            if i%tstFreq==0 or i+1==epochs:
                tst_dataL = torch.mean((self.net(test_inputs) - test_targets)**2, dim=0).detach_()
                tst_col_preds = self.net(test_inputs)
                tst_t_prtls = Nth_temporal_prtls(values=tst_col_preds, pts=test_inputs, orders=self.tmprl_ords).detach_()
                tst_lib_evals = self.lib_func.Calc(network=self.net, inpts=test_inputs)
                tst_pdeL = torch.mean((tst_t_prtls - tst_lib_evals@self.lmbda)**2, dim=0,).detach_()
                # tst_pdeL = torch.zeros(size=(n_trgts,), device=self.device, dtype=self.data_type)
                # for j in range(n_trgts): tst_pdeL[j] = torch.mean((tst_t_prtls[:,j:j+1] - tst_lib_evals[:, nzs[j]] @ tst_lib_evals[nzs[j],j:j+1]**2), dim=0).detach_()
                ts_loss = torch.sum(tst_dataL + alpha*tst_pdeL).detach_()
                tst_loss[i] = ts_loss.detach()
                tst_data_loss[i] = tst_dataL.detach()
                tst_pde_loss[i] = tst_pdeL.detach()

    

        optim.zero_grad(set_to_none=True)
        print('Finished Adams Post Training')
        del train_inputs, train_targets, test_inputs, test_targets, colpnts
        self._PstTrnAdamsEpochs = epochs
        self._PstTrnAdamsLrnRt = lrn_rt
        self._PstTrnAdamsAlpha = alpha
        self.AdamsPstTrnLoss = tot_loss.cpu().numpy()
        self.AdamsPstTstLoss = tst_loss.cpu().numpy()
        self.AdamsPstTrnDataLoss = data_loss.cpu().numpy()
        self.AdamsPstTstDataLoss = tst_data_loss.cpu().numpy()
        self.AdamsPstTrnEqLoss = pde_loss.cpu().numpy()
        self.AdamsPstTstEqLoss = tst_pde_loss.cpu().numpy()
        self.AdamsPstTrnLambdas[1] = self.lmbda.data.cpu().numpy()
        self.AdamsPstTrnFvus[1] = self.FVU_Calc(lib_ceofs=self.lmbda.data).cpu().numpy()
    
    def Lbfgs_PostTraining(self, alpha:float=0.5, gamma:float=0.25, epochs:int=1000,
        lrn_rt:float=0.001, Save_File_Name:str=None, tstFreq:int=100, schdlr_freq:int=0,
        max_it:int=20, max_evl:int=None, grad_tol:float=1e-07, tol_change:float=1e-09,
            history_size:int=100, line_srch_fn:str=None):

        if Save_File_Name==None or (not isinstance(Save_File_Name, str)):
            msg = ("The Save_File_Name argument was None or was not a string object the file name  \n"
                   "where the model will periodically be saved to is based off the day and time the \n" 
                   "class object was created. ")
            warnings.warn(message=msg, stacklevel=1)
            Save_File_Name = self.sv_fname
        Save_File_Name += ".pt"

        self.LbfgsPstTrnLambdas = np.empty(shape=(2, *self.lmbda.data.shape), dtype=float)
        self.LbfgsPstTrnFvus = np.empty(shape=(2, self.lmbda.data.shape[1]), dtype=float)
        self.LbfgsPstTrnLambdas[0] = self.lmbda.data.cpu().numpy()
        self.LbfgsPstTrnFvus[0] = self.FVU_Calc(lib_ceofs=self.lmbda.data).cpu().numpy()

        optim = torch.optim.LBFGS(params=self.net.parameters(), lr=lrn_rt, max_iter=max_it, max_eval=max_evl,
            tolerance_grad=grad_tol, tolerance_change=tol_change, history_size=history_size, line_search_fn=line_srch_fn)
        optim.param_groups[0]['params'].append(self.lmbda)
        schdlr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.9, last_epoch=-1, ) 
        # schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10,
        #                                         threshold=0.001, threshold_mode='rel', cooldown=1, min_lr=0, 
        #                                             eps=1e-08, )

        train_inputs = torch.from_numpy(self.data_dict['Train_Inputs']).to(device=self.device, dtype=self.data_type)
        train_targets = torch.from_numpy(self.data_dict['Train_Targets']).to(device=self.device, dtype=self.data_type)
        test_inputs = torch.from_numpy(self.data_dict['Test_Inputs']).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        test_targets = torch.from_numpy(self.data_dict['Test_Targets']).to(device=self.device, dtype=self.data_type)
        self.nDpnts = train_inputs.size(0)
        colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        if not colpnts.requires_grad:
            colpnts.requires_grad_(True)

        tot_loss = torch.zeros(size=(epochs, 1), device=self.device,)
        data_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)
        pde_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)

        tst_loss = torch.zeros(size=(epochs, 1), device=self.device,)
        tst_data_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)
        tst_pde_loss = torch.zeros(size=(epochs, test_targets.size(-1)), device=self.device,)

        def closure():
            optim.zero_grad(set_to_none=True)

            dataL = torch.mean((self.net(train_inputs) - train_targets)**2, dim=0)
            col_preds = self.net(colpnts)
            t_prtls = Nth_temporal_prtls(values=col_preds, pts=colpnts, orders=self.tmprl_ords)
            lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts)
            pdeL = torch.mean((t_prtls - lib_evals@self.lmbda)**2, dim=0,)
            loss = torch.sum(dataL + alpha*pdeL + gamma*torch.linalg.vector_norm(self.lmbda, ord=1, dim=0))

            # loss.backward()
            if loss.isnan().sum():
                    print(f"Note that dataLs={dataL} colloLs={pdeL} and loss={loss}")
                    optim.zero_grad(set_to_none=True)
            if loss.requires_grad and torch.isfinite(loss):
                loss.backward()
            # elif loss.isnan().sum():
            #     optim.zero_grad(set_to_none=True)
            return loss


        for i in range(epochs):

            self.net.train(mode=True)

            optim.step(closure)
            if schdlr_freq and i>0 and i%schdlr_freq==0:
                schdlr.step()
                # schdlr.step(metrics=loss)

            self.net.eval()
            trn_dataL = torch.mean((self.net(train_inputs) - train_targets)**2, dim=0).detach_()
            trn_col_preds = self.net(colpnts)
            trn_t_prtls = Nth_temporal_prtls(values=trn_col_preds, pts=colpnts, orders=self.tmprl_ords).detach_()
            trn_lib_evals = self.lib_func.Calc(network=self.net, inpts=colpnts)
            trn_pdeL = torch.mean((trn_t_prtls - trn_lib_evals@self.lmbda)**2, dim=0,).detach_()
            trn_loss = torch.sum(trn_dataL + alpha*trn_pdeL + gamma*torch.linalg.vector_norm(self.lmbda, ord=1, dim=0)).detach_()

            tot_loss[i] = trn_loss.detach()
            data_loss[i] = trn_dataL.detach()
            pde_loss[i] = trn_pdeL.detach()

            if i%tstFreq==0 or i+1==epochs:
                tst_dataL = torch.mean((self.net(test_inputs) - test_targets)**2, dim=0)
                tst_col_preds = self.net(test_inputs)
                tst_t_prtls = Nth_temporal_prtls(values=tst_col_preds, pts=test_inputs, orders=self.tmprl_ords)
                tst_lib_evals = self.lib_func.Calc(network=self.net, inpts=test_inputs)
                tst_pdeL = torch.mean((tst_t_prtls - tst_lib_evals@self.lmbda)**2, dim=0,)
                loss = torch.sum(tst_dataL + alpha*tst_pdeL)
                tst_loss[i//tstFreq] = loss.detach()
                tst_data_loss[i//tstFreq] = tst_dataL.detach()
                tst_pde_loss[i//tstFreq] = tst_pdeL.detach()

        optim.zero_grad(set_to_none=True)
        print('Finished LBFGS Pretraining')
        del train_inputs, train_targets, test_inputs, test_targets, colpnts
        self._PstTrnLBFGsEpochs = epochs
        self._PstTrnLBFGsLrnRt = lrn_rt
        self._PstTrnLBFGsAlpha = alpha
        self.LbfgsPstTrnLoss = tot_loss.cpu().numpy()
        self.LbfgsPstTstLoss = tst_loss.cpu().numpy()
        self.LbfgsPstTrnDataLoss = data_loss.cpu().numpy()
        self.LbfgsPstTstDataLoss = tst_data_loss.cpu().numpy()
        self.LbfgsPstTrnEqLoss = pde_loss.cpu().numpy()
        self.LbfgsPstTstEqLoss = tst_pde_loss.cpu().numpy()
        self.LbfgsPstTrnLambdas[1] = self.lmbda.data.cpu().numpy()
        self.LbfgsPstTrnFvus[1] = self.FVU_Calc(lib_ceofs=self.lmbda.data).cpu().numpy()
        
    def WriteResults(self, data_set_name:str, file_name:str='Results', precision:int=5, true_eq=None, errors=None, 
                     act_func=None, **kwargs)->None:
        """
        Method that appends to a file in the current working direct titled as file_name.txt the results of the model at the time of
        rutorch.nning this function. So for this method to work properly it is best to have all the model values set prior to rutorch.nning. 
        Input arguments are as follows:
            * lrnd_eq (str) - The learned equation written as a string argument. At this current moment expect this arg.
                        to just basically be given as u_t + model.Learned_EQ(). In any case what ever this string argument
                        is it will be written to the file as the learned equations
            * file_name (str) - name of the file that the results will be written/appended to. NOTE: Do not attatch a file
                        extension as the results will always be written to a .txt file for simplicity with the name as 
                        file_name.txt. Attatching a file extension will have unknown behavior at this current moment of
                        writing (03/11/2022)
            * precision (int) - The precision with wich the lambda values throughout all the learning steps will be 
                        printed in the txt file. The values are writen in exponential form and so this defines the
                        number of decimal places shown. 
            * true equation: An optional input argument. If the true PDE equation is know by the user then passing that
                        equations written as a string to this argument will result in the results file containing this 
                        true equation. If given it must be a string 
            * errors: An optional input argument. If the user knows what the true equation is and has a way of 
                        determining the error between that equation and what was learned as a float then the written
                        results will have this included on its own line. Error must be given as a float.
        TODO -  (1) Rework the learned equation method of the class to create a class variable that contains the learned equation
                or possibly have this already created and the method just updates it when it is run so that it does not have to be an
                argument to this function and can just be called using self.lrnd_eq or something like that. 
                (2) Handle the case the the ADO-RFE was run with 0 epochs or (inclusive) 0 ADO_iters
        """
        # First some like input arguments checking:
        if not isinstance(file_name, str):
            print('ERROR - The user passed value for the file_name input argument is not a string but is a {} '.format(type(file_name)))
            print('Results will instead be appened to the following file in the Current working directy - Results.txt')
            file_name = 'Results'
        if not (isinstance(true_eq, str)  or true_eq is None):
            print('ERROR - The user passed value for the true_eq input argument is not a string nor None but is a {} '.format(type(true_eq)))
            print('To handle will not print anything about true results')
            true_eq = None
        if not (isinstance(errors, list)  or errors is None):
            print('ERROR - The user passed input for the errors input argument is not a list (of errors) nor None but is a {} '.format(type(errors)))
            print('To handle will not print anything about true results')
            errors = None
        if errors==None: errors=['~~~~', '~~~~~']

        dvc = self.device

        library_names = self.lib_func.get_library_names(net_out_func_names=self.ntwk_out_names)
        n_lib = len(library_names)
        n_trgts = self.lmbda.data.size(1)
        lrnd_eq = self.Learned_EQ(output=False,)
        # get someway to identify at the time that this function was run what training methods have been completed. 
        # NOTE: MAYBE A BETTER IDEA TO HAVE A PRIVATE CLASS LIST IS ONLY UPDATED WHEN A TRAINING METHOD IS RUN TO IDENTIFY WHICH TRAINING METHODS WERE USED AND THE ORDER
        # as I am working under the assumption that the things have been run in the poper order of pretrainings, ADO then post trainings
        trn_ids = np.array([0 if itm is None else 1 for itm in [self.AdamsPreTrnLambdas, self.LbfgsPreTrnLambdas, self._ADO_lambdas, self.AdamsPstTrnLambdas, self.LbfgsPstTrnLambdas]])
        if trn_ids.sum()==0:
            raise RuntimeError("Training has not be done in any form for this learning model")
        plc_hldr = np.empty(shape=self.lmbda.shape)
        FVU_plc_hldr = np.empty(shape=(n_trgts,),)
        headings = ['Lib.Terms', 'Init.']
        if trn_ids[0] == 1: # Adams pretraining was done 
            # print('Adams Pre-Training was done to get these results')
            headings.append('Adams PrTrn')
            plc_hldr = np.concatenate((plc_hldr, self.AdamsPreTrnLambdas.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
            FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.AdamsPreTrnFvus.flatten()), axis=0)
        if trn_ids[1] == 1: # LBFGS pretraining was done 
            # print('LBFGS Pre-Training was done to get these results')
            headings.append('LFBGS PrTrn')
            if trn_ids[:1].sum()==0: # IF this is true than this traing method/step was the first one done.
                plc_hldr = np.concatenate((plc_hldr, self.LbfgsPreTrnLambdas.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.LbfgsPreTrnFvus.flatten()), axis=0)
            else: 
                plc_hldr = np.concatenate((plc_hldr, self.LbfgsPreTrnLambdas[1]), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.LbfgsPreTrnFvus[1]), axis=0)
        if trn_ids[2] == 1: # ADO was done 
            # print('ADO Training was done to get these results')
            for i in range(self._ADO_iters+1):
                headings.append('ADO '+str(i))
            # if self._ADO_iters==0: self._ADO_iters=None
            if trn_ids[:2].sum()==0: # IF this is true than this traing method/step was the first one done.
                plc_hldr = np.concatenate((plc_hldr, self._ADO_lambdas.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self._ADO_FVUs.flatten()), axis=0)
            else: 
                plc_hldr = np.concatenate((plc_hldr, self._ADO_lambdas[1:].reshape((-1,1),order='C').reshape((n_lib,self._ADO_iters+1),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self._ADO_FVUs[1:].flatten()), axis=0)
        if trn_ids[3] == 1: # Adams post training was done
            # print('Adams Post Training was done to get these results')
            headings.append('Adams PostTrn')
            if trn_ids[:3].sum()==0: # IF this is true than this traing method/step was the first one done.
                plc_hldr = np.concatenate((plc_hldr, self.AdamsPstTrnLambdas.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.AdamsPstTrnFvus.flatten()), axis=0)
            else: 
                plc_hldr = np.concatenate((plc_hldr, self.AdamsPstTrnLambdas[1]), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.AdamsPstTrnFvus[1]), axis=0)
        if trn_ids[4] == 1: # Adams post training was done 
            # print('LBFGS Post Training was done to get these results')
            headings.append('LBFGS PostTrn')
            if trn_ids[:4].sum()==0: # IF this is true than this traing method/step was the first one done.
                plc_hldr = np.concatenate((plc_hldr, self.LbfgsPstTrnLambdas.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.LbfgsPstTrnFvus.flatten()), axis=0)
            else: 
                plc_hldr = np.concatenate((plc_hldr, self.LbfgsPstTrnLambdas[1]), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.LbfgsPstTrnFvus[1]), axis=0)

        # Now place holder will hold all the lambas parameter values/vectors throughout all the training 
        # step 
        tbl_vals = np.concatenate((plc_hldr[:,n_trgts:], FVU_plc_hldr[n_trgts:][np.newaxis,:]), axis=0)
        
        # precision = 5
        lst = list(kwargs.keys())
        if 'jobID' in kwargs.keys(): jobID = kwargs['jobID']
        else: jobID = None
        if 'jobVer'in kwargs.keys(): jobVer = kwargs['jobVer']
        else: jobVer = None
        if 'node' in kwargs.keys(): node = kwargs['node']
        else: node = None
        if 'run_time'in kwargs.keys(): run_time = kwargs['run_time']
        else: run_time = 0
        if 'subsample_prcntg' in kwargs.keys(): subsample_prcntg = kwargs['subsample_prcntg']
        else: subsample_prcntg = None
        if 'noisePrcntg' in kwargs.keys(): noisePrcntg = kwargs['noisePrcntg']
        else: noisePrcntg = None
        if 'NpSeed' in kwargs.keys(): NpSeed = kwargs['NpSeed']
        else: NpSeed = None
        if 'Ns' in kwargs.keys(): Ns = kwargs['Ns']
        else: Ns = None

        # All this does is to print out in the data file the evolution of the candidate library coefficients
        # through out the whole process of learning. 
        longest = 0     # what is the longest library function name
        library_names.append('FVU values')
        for term in library_names:
            if longest<len(term):
                longest=len(term)
        if longest < len(' Adams PostTrn '):
            longest = len(' Adams PostTrn ')
        if longest%2==1: longest+=1
        # Now make the rows that form the sort of coefficient evolution table
        space = ' '
        lines = ''
        temp = '%3.{}e'.format(precision)
        n_tbl_cls = tbl_vals.shape[1]
        n_trn_stps = int(n_tbl_cls / n_trgts)
        for k in range(n_lib+1):
            lines+=library_names[k]
            lines += (longest-len(library_names[k]) )*space
            lines += " |"
            for j in range(n_trn_stps):
                for l in range(n_trgts):
                    col = n_trgts*j+l
                    if (l+1)==n_trgts:
                        if tbl_vals[k,col].item()>= 0:
                            lines = lines +  " "+ temp % (tbl_vals[k,col].item()) + " | "
                        else:
                            lines = lines +  temp % (tbl_vals[k,col].item()) + " | "
                    else:    
                        if tbl_vals[k,col].item()>= 0:
                            lines = lines +  " "+ temp % (tbl_vals[k,col].item()) + " , "
                        else:
                            lines = lines +  temp % (tbl_vals[k,col].item()) + " , "
            # for j in range(n_tbl_cls):
            #     if tbl_vals[k,j].item()>= 0:
            #         lines = lines +  " "+ temp % (tbl_vals[k,j].item()) + " | "
            #     else:
            #         lines = lines +  temp % (tbl_vals[k,j].item()) + " | "
            lines+='\n'
        n = lines.find('\n')
        # Now make the header or the top row of the table that contains the columns names
        hline=''        # horizontal line of ------
        header = ''     # the header and or top row containing the column names
        breaks=[0]
        starts = []
        for k, i in enumerate(lines[:n]):
            if i=='|':
                breaks.append(k)
        for k in range(0, len(breaks)-1): 
            mid = int((breaks[k]+breaks[k+1])/2)
            starts.append(mid-int(len(headings[k])/2))
        i = 0
        j = 0
        k = 1
        while i<n:
            if i==starts[j]:
                header+=headings[j]
                hline+=len(headings[j])*'-'
                i+=len(headings[j])
                j+=1
                if j==len(headings):
                    j=0
            elif i==breaks[k]:
                header+='|'
                hline+='-'
                k+=1
                i+=1
                if k==len(breaks):
                    k=1
            else:
                header+=space
                hline+='-'
                i+=1
        # final = header+'\n'+hline+'\n'+lines+'\n'+ '~'*150+'\n'
        # final = header+'\n'+hline+'\n'+lines+'\n'
        idx = lines.find('FVU values')
        final = header+'\n'+hline+'\n'+lines[:idx]+hline+'\n'+lines[idx:]+'\n'
        if self.earl_lmbds is not None:
            final += 'Early termination learned equations (ie. EQs learned if ADO was terminated earlier with original RFE)\n'
            for i, lmbda in enumerate(self.earl_lmbds):
                final += f"ADO iter. {i}:\n"
                eqs = self.Learned_EQ(coefs=lmbda, output=False,).split('\n')
                for j, eqnt in enumerate(eqs):
                    # final += f"(FVU = {self.earl_fvus[i, j].round(8)}) " + eqnt + '\n'
                    final +=  "(FVU = " + temp % (self.earl_fvus[i, j].item()) + ") " + eqnt + '\n'
                
        final +='~'*150+'\n'
        # Write the results to the txt data file
        with open(file_name+'Results.txt', 'a', encoding='UTF-8', errors='replace') as file:
            file.write('Results and Hyperparamter Values for '+data_set_name+' data set using RFE - Job Num = {}, Ver. = {} ran on {} \n'.format(jobID, jobVer, node))                                                                                  # line 1
            file.write('Network Nonlinear activation function  = {} \n'.format(act_func))                                                                                                                                                               # line 2 
            file.write('Device that the results were obtained on - {}\n'.format(dvc))                                                                                                                                                                   # line 3
            file.write('Total wall-clock run time  = {} seconds = {} minutes\n'.format(run_time, (run_time)/60))                                                                                                                                        # line 4
            file.write('Various Hyperparater values and trainining settings used to get these results:\n')                                                                                                                                              # line 5
            file.write('Num of training data points = {}, data sampling percentage {}%, Num of collocation points = {}, noise percentage {}%\n'.format(self.nDpnts, subsample_prcntg*100, self.N_col_pnts, noisePrcntg))                                # line 6
            file.write('Training Batch size = {}, Num training spatial points = {}, Num testing spatial points = {}, Numpy RNG seed/entropy value = {}\n'.format(self.trn_batch_size, Ns[0], Ns[1], NpSeed))                                            # line 7
            file.write('Adams Pretraining Stuff - epochs = {}, init. learning rate = {}, alpha loss value = {}, beta loss value = {}\n'.format(self._AdamsPreTrnEpochs, self._AdamsPreTrnLrnRt, self._AdamsPreTrnAlpha, self._AdmasPreTrnGamma))        # line 8
            file.write('LBFGS Pretraining Stuff - epochs = {}, init. learning rate = {}, alpha loss value = {}, beta loss value = {}\n'.format(self._LBFGsPreTrnEpochs, self._LBFGsPreTrnLrnRt, self._LBFGsPreTrnAlpha, self._LBFGsPreTrnGamma))        # line 9
            file.write('ADO training stuff - iterations = {}, training epochs = {}, Adams alpha hyper parameter values = {}\n'.format(self._ADO_iters, self._ADO_epchs, self._ADO_alphas))                                                              # line 10
            file.write('Post ADO Adams training Stuff - epochs = {}, init. learning rate = {}, alpha loss value = {}\n'.format(self._PstTrnAdamsEpochs, self._PstTrnAdamsLrnRt, self._PstTrnAdamsAlpha))                                                # line 11
            file.write('Post ADO LBFGS training Stuff - epochs = {}, init. learning rate = {}, alpha loss value = {}\n'.format(self._PstTrnLBFGsEpochs, self._PstTrnLBFGsLrnRt, self._PstTrnLBFGsAlpha))                                                # line 12
            file.write('LEARNED Equation is - \n '+lrnd_eq+'\n')                                                                                                                                                                                        # line 13 & 14
            if not (true_eq is None):
                file.write('TRUE Equation is - \n '+true_eq+'\n')                                                                                                                                                                                       # line 15 & 16
            else:
                file.write('TRUE Equation is - \n  \n')                                                                                                                                                                                                 # line 15 & 16
            file.write('Individual Coefficient (Relative) Errors:\n {}\n'.format(errors[1]))                                                                                                                                                            # lines 17 and 18 
            file.write('Mean Coefficient (Relative) Error:\n {}\n'.format(errors[0]))                                                                                                                                                                   # lines 19 and 20 
            file.write('Evolution of candidate library lambda/coefficient values throughout learning\n')                                                                                                                                                # line 21
            file.write(final)                                                                                                                                                                                                                           # line 22 (head), 23(-----) and as 
                                                                                                                                                                                                                                                        # 24-[24+(num. lib terms)-1 + 2]
        return None

    def TrainingLossPlots(self, dataset_name:str, file_name:str, plot_title:str='PreTrain and ADO Training Losses', 
                          font_size:float=10, show_fig:bool=False)->None:
        """
            Create a plot displaying the model's training losses through both pre-training and ADO-RFE training. The losses are
            plottd on a single figure as one single continuous line over all the training epochs such that losses through the 
            pretraining are first drawn in blue then the losses incurred over the ADO steps in differing colors one after the next.
            The x-axis label is super epochs since it is all the pretraining epochs may differ from the ADO training epochs but 
            the total number of epoch that that the model trained under is the summation of the number of pretrained epoch and 
            the ADO-RFE epochs. The figure contain the plot of the training losses will be saved as a png file to a folder titled
            LossFigures/dataset_name where dataset_name the current value of model.dataset_name (see set_Dataset_Name() method).
            Should these folder not be created or there is a problem changing to them either, the folders will be created or the 
            figure/plot will be saved tothe last folder/directory that we were in before trying to change had some error. 
            The input arguements are as follows:
                * file_name (str) - The name of the file under which the plot of the training losses is saved. It should not 
                            include .png or anything like it. 
                * plot_title (str) - Title of the plot
                * font_size (float) - fontsize of the plot title's text
                * show_fig (bool) - Whether or not to display the figure prior to saving it to disk
        """
        # NOTE - just some like input argument cheching. Eventually will need to do something better and stronger than this
        # in the future prior to release
        if not isinstance(file_name, str):
            print('ERROR!!! - The file_name argument needs to be a string object not a {} object'.format(type(file_name)))
            print('Will handle this by using the file_name = ErrorNamedPlot.png')
            file_name = 'ErrorNamedPlot'
        if not isinstance(plot_title, str):
            print('ERROR!!! - The plot_title argument needs to be a string object not a {} object'.format(type(plot_title)))
            print('Will handle this by using the plot_title = Prtrn and ADO Training Losses')
            plot_title = 'Prtrn and ADO Training Losses'
        if not isinstance(font_size, (float, int)):
            print('ERROR!!! - The font_size argument needs to be a float/int object not a {} object'.format(type(font_size)))
            print('Will handle this by using the font_size = 10')
            font_size = 10
        if not isinstance(show_fig, bool):
            print('ERROR!!! - The show_fig argument needs to be a string object not a {} object'.format(type(show_fig)))
            print('Will handle this by using the show_fig = False')
            show_fig = False
        start_dir = os.getcwd()
        # Now lets do the ploting and saving shit
        # First get the losses and shit
        ADO_trn_losses = self._ADOTrnLosses
        ADO_tst_losses = self._ADOTstLosses
        ADOtrnDataLs = self._ADOtrnDataLs
        ADOtstDataLs = self._ADOtstDataLs
        ADOtrnColloLs = self._ADOtrnColloLs
        ADOtstColloLs = self._ADOtstColloLs
        # Output Figs and Vids File names
        try:
            os.mkdir('LossFigures')
        except FileExistsError:
            print('LossFigures Directory already exists so did not create it')
        try: 
            os.chdir('LossFigures')
        except (OSError, FileNotFoundError, PermissionError, NotADirectoryError):
            print('Could not change directory to LossFigures. Results will be writen to file in CWD={}'.format(os.getcwd()))
        # try and get in to the directory for this dataset
        try:
            os.mkdir(dataset_name)
        except FileExistsError:
            print('{} Directory already exists so did not create it'.format(dataset_name))
        try: 
            os.chdir(dataset_name)
        except (OSError, FileNotFoundError,PermissionError,NotADirectoryError):
            print('Could not change directory to {}. Plots of Losses will be writen to file in CWD={}'.format(dataset_name, os.getcwd()))

        # Combine the losses from the pretraining and ado-training epochs for Adams
        adams_pre_trn_epchs = self._AdamsPreTrnLoss.shape[0] if isinstance(self._AdamsPreTrnLoss,np.ndarray) else 0
        LBFGS_pre_trn_epchs = self._LBFGsPreTrnLoss.shape[0] if isinstance(self._LBFGsPreTrnLoss,np.ndarray) else 0
        adams_post_trn_epchs = self._AdamsPstTrnLoss.shape[0] if isinstance(self._AdamsPstTrnLoss,np.ndarray) else 0
        LBFGS_post_trn_epchs = self._LBFGsPstTrnLoss.shape[0] if isinstance(self._LBFGsPstTrnLoss,np.ndarray) else 0
        if isinstance(ADO_trn_losses, np.ndarray):ADO_iters, ADO_epchs = ADO_trn_losses.shape 
        else: ADO_iters, ADO_epchs = 0, 0
        if adams_pre_trn_epchs==0 and LBFGS_pre_trn_epchs==0 and ADO_epchs==0 and ADO_iters==0 and adams_post_trn_epchs==0 and LBFGS_post_trn_epchs==0:
            print('NO plot can be made as there are no losses. You have yet to run the pretraining nor ADO training IDOIT!!!!')
            return None
        # Now need to work out the logic for this to work incase some idiot has run this with only one of the two trianing methods having been done
        # Create the array to contian all the training losses
        Cmbd_trn_losses = np.empty(shape=(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+ADO_epchs*ADO_iters+adams_post_trn_epchs+LBFGS_post_trn_epchs,), dtype=float)
        Cmbd_tst_losses = np.empty(shape=(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+ADO_epchs*ADO_iters+adams_post_trn_epchs+LBFGS_post_trn_epchs,), dtype=float)
        cmbdTrn_data_losses = np.empty(shape=(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+ADO_epchs*ADO_iters+adams_post_trn_epchs+LBFGS_post_trn_epchs,), dtype=float)
        cmbdTst_data_losses = np.empty(shape=(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+ADO_epchs*ADO_iters+adams_post_trn_epchs+LBFGS_post_trn_epchs,), dtype=float)
        cmbdTrn_collo_losses = np.empty(shape=(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+ADO_epchs*ADO_iters+adams_post_trn_epchs+LBFGS_post_trn_epchs,), dtype=float)
        cmbdTst_collo_losses = np.empty(shape=(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+ADO_epchs*ADO_iters+adams_post_trn_epchs+LBFGS_post_trn_epchs,), dtype=float)
        # Now start adding things to the array - NOTE some of these statements can be combined to have less if statements
        if isinstance(self._AdamsPreTrnLoss,np.ndarray): 
            Cmbd_trn_losses[:adams_pre_trn_epchs] = self._AdamsPreTrnLoss
            Cmbd_tst_losses[:adams_pre_trn_epchs] = self._AdamsPreTstLoss
            cmbdTrn_data_losses[:adams_pre_trn_epchs] = self._AdamsPreTrnDataLoss
            cmbdTst_data_losses[:adams_pre_trn_epchs] = self._AdamsPreTstDataLoss
            cmbdTrn_collo_losses[:adams_pre_trn_epchs] = self._AdamsPreTrnColloLoss
            cmbdTst_collo_losses[:adams_pre_trn_epchs] = self._AdamsPreTstColloLoss
        if isinstance(self._LBFGsPreTrnLoss,np.ndarray): 
            Cmbd_trn_losses[adams_pre_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs] = self._LBFGsPreTrnLoss
            Cmbd_tst_losses[adams_pre_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs] = self._LBFGsPreTstLoss
            cmbdTrn_data_losses[adams_pre_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs] = self._LBFGsPreTrnDataLoss
            cmbdTst_data_losses[adams_pre_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs] = self._LBFGsPreTstDataLoss
            cmbdTrn_collo_losses[adams_pre_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs] = self._LBFGsPreTrnColloLoss
            cmbdTst_collo_losses[adams_pre_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs] = self._LBFGsPreTstColloLoss

        # Now add the ADO losses
        for i in range(ADO_iters):
            Cmbd_trn_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs] = ADO_trn_losses[i,:]
            Cmbd_tst_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs] = ADO_tst_losses[i,:]
            cmbdTrn_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs] = ADOtrnDataLs[i,:]
            cmbdTst_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs] = ADOtstDataLs[i,:]
            cmbdTrn_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs] = ADOtrnColloLs[i,:]
            cmbdTst_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs] = ADOtstColloLs[i,:]
        if ADO_iters==0: i=0
        # Now add the post ADO losses
        if isinstance(self._AdamsPstTrnLoss,np.ndarray): 
            Cmbd_trn_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs] = self._AdamsPstTrnLoss
            Cmbd_tst_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs] = self._AdamsPstTstLoss
            cmbdTrn_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs] = self._AdamsPstTrnDataLoss
            cmbdTst_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs] = self._AdamsPstTstDataLoss
            cmbdTrn_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs] = self._AdamsPstTrnColloLoss
            cmbdTst_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs] = self._AdamsPstTstColloLoss
        if isinstance(self._LBFGsPstTrnLoss,np.ndarray): 
            Cmbd_trn_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+LBFGS_post_trn_epchs] = self._LBFGsPstTrnLoss
            Cmbd_tst_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+LBFGS_post_trn_epchs] = self._LBFGsPstTstLoss
            cmbdTrn_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+LBFGS_post_trn_epchs] = self._LBFGsPstTrnDataLoss
            cmbdTst_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+LBFGS_post_trn_epchs] = self._LBFGsPstTstDataLoss
            cmbdTrn_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+LBFGS_post_trn_epchs] = self._LBFGsPstTrnColloLoss
            cmbdTst_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+LBFGS_post_trn_epchs] = self._LBFGsPstTstColloLoss
        # now with losses from each individual array combined and placed in one larger array plot the things
        # temp_fig = plt.figure(figsize=(10, 8))
        cmbnd_fig = plt.figure(figsize=(18, 10))
        if adams_pre_trn_epchs>0:
            plt.plot(np.arange(adams_pre_trn_epchs)+1, Cmbd_tst_losses[:adams_pre_trn_epchs], linestyle='solid', linewidth=1, color='black', label='Test Data Losses')
            plt.plot(np.arange(adams_pre_trn_epchs)+1, Cmbd_trn_losses[:adams_pre_trn_epchs], linestyle='solid', linewidth=1, label='Pretrn Adams')
        if LBFGS_pre_trn_epchs>0:
            if adams_pre_trn_epchs==0: plt.plot(np.arange(adams_pre_trn_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs)+1, Cmbd_tst_losses[adams_pre_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs], linestyle='solid', linewidth=1, color='black', label='Test Data Losses')
            else: plt.plot(np.arange(adams_pre_trn_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs)+1, Cmbd_tst_losses[adams_pre_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs], linestyle='solid', linewidth=1, color='black')
            l1 = plt.plot(np.arange(adams_pre_trn_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs)+1, Cmbd_trn_losses[adams_pre_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs], linestyle='solid', linewidth=1, label='Pretrn LBFGS')
            if adams_pre_trn_epchs>0:
                plt.plot(np.arange(adams_pre_trn_epchs-1,adams_pre_trn_epchs+1)+1, 
                        Cmbd_trn_losses[adams_pre_trn_epchs-1:adams_pre_trn_epchs+1],
                        linestyle='dashed', linewidth=1, color=l1[0].get_color())

        for i in range(ADO_iters):
            l1 = plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs)+1,
                    Cmbd_trn_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs], 
                    linestyle='solid', linewidth=1, label='ADO {} data'.format(i+1))
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs)+1,
                    Cmbd_tst_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs], 
                    linestyle='solid', linewidth=1,color='black')
            if adams_pre_trn_epchs!=0 or LBFGS_pre_trn_epchs!=0: # IE some pretraitorch.nning was done.
                plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs-1,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs+1)+1, 
                        Cmbd_trn_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs+1],
                        linestyle='dashed', linewidth=1, color=l1[0].get_color())
                plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs-1,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs+1)+1, 
                        Cmbd_tst_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs+1],
                        linestyle='dashed', linewidth=1, color='black')
            if adams_pre_trn_epchs==0 and LBFGS_pre_trn_epchs==0 and i>=1: # No pretraining was done - NOTE I may not need this as the the would both be zero
                plt.plot(np.arange(i*ADO_epchs-1,i*ADO_epchs+1)+1, 
                        Cmbd_trn_losses[i*ADO_epchs-1:i*ADO_epchs+1],
                        linestyle='dashed', linewidth=1, color=l1[0].get_color())
                plt.plot(np.arange(i*ADO_epchs-1,i*ADO_epchs+1)+1, 
                        Cmbd_tst_losses[i*ADO_epchs-1:i*ADO_epchs+1],
                        linestyle='dashed', linewidth=1, color='black')
        if adams_post_trn_epchs>0:
            l1 = plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs)+1, 
            Cmbd_trn_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs],
            linestyle='solid', linewidth=1, label='PstTrn Adams')
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs)+1, 
            Cmbd_tst_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs],
            linestyle='solid', linewidth=1, color='black',)
            # now the dashed line so show the change from one method to the next one
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs-1,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+1)+1, 
                    Cmbd_trn_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+1],
                    linestyle='dashed', linewidth=1, color=l1[0].get_color())
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs-1,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+1)+1, 
                    Cmbd_tst_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+1],
                    linestyle='dashed', linewidth=1, color='black')
        if LBFGS_post_trn_epchs>0:
            l1 = plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs+LBFGS_post_trn_epchs)+1, 
            Cmbd_trn_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs: adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+LBFGS_post_trn_epchs],
            linestyle='solid', linewidth=1, label='PstTrn LBFGS')
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs+LBFGS_post_trn_epchs)+1, 
            Cmbd_tst_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs: adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+LBFGS_post_trn_epchs],
            linestyle='solid', linewidth=1, color='black',)
            # now the dashed line so show the change from one method to the next one
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs-1, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs+1)+1, 
                    Cmbd_trn_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs+1],
                    linestyle='dashed', linewidth=1, color=l1[0].get_color())
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs-1, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs+1)+1, 
                    Cmbd_tst_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs+1],
                    linestyle='dashed', linewidth=1, color='black')

        # Now that the lines have been plotted, plot some dots to indicate where things transition over from one training step/method to the next
        plt.scatter(x=1, y=Cmbd_trn_losses[0], s=15, c=1000)
        if adams_pre_trn_epchs!=0 and LBFGS_pre_trn_epchs!=0:
            plt.scatter(x=adams_pre_trn_epchs+1, y=Cmbd_trn_losses[adams_pre_trn_epchs], s=15, c=1000)
        for i in range(ADO_iters):
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs +1, y=Cmbd_trn_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs], s=15, c=1000)
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs +1, y=Cmbd_tst_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs], s=15, c=1000)
        if adams_post_trn_epchs>0:
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs +1, y=Cmbd_trn_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs], s=15, c=1000)
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs +1, y=Cmbd_tst_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs], s=15, c=1000)
        if LBFGS_post_trn_epchs>0:
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+1, y=Cmbd_trn_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs], s=15, c=1000)
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+1, y=Cmbd_tst_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs], s=15, c=1000)

        plt.xlabel('(Super) epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title(plot_title, fontsize=font_size)
        if show_fig:
            plt.show()
        cmbnd_fig.savefig(fname=file_name+'CombinedLosses.png', format='png')
        # plot for "data" loss
        dataLs_fig = plt.figure(figsize=(18, 10))
        if adams_pre_trn_epchs>0:
            plt.plot(np.arange(adams_pre_trn_epchs)+1, cmbdTst_data_losses[:adams_pre_trn_epchs], linestyle='solid', linewidth=1, color='black', label='Test Data Losses')
            plt.plot(np.arange(adams_pre_trn_epchs)+1, cmbdTrn_data_losses[:adams_pre_trn_epchs], linestyle='solid', linewidth=1, label='Pretrn Adams')
        if LBFGS_pre_trn_epchs>0:
            if adams_pre_trn_epchs==0: 
                plt.plot(np.arange(adams_pre_trn_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs)+1, cmbdTst_data_losses[adams_pre_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs], linestyle='solid', linewidth=1, color='black', label='Test Data Losses')
            else: 
                plt.plot(np.arange(adams_pre_trn_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs)+1, cmbdTst_data_losses[adams_pre_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs], linestyle='solid', linewidth=1, color='black')
            l1 = plt.plot(np.arange(adams_pre_trn_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs)+1, cmbdTrn_data_losses[adams_pre_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs], linestyle='solid', linewidth=1, label='Pretrn LBFGS')
            if adams_pre_trn_epchs>0:
                plt.plot(np.arange(adams_pre_trn_epchs-1,adams_pre_trn_epchs+1)+1, 
                        cmbdTrn_data_losses[adams_pre_trn_epchs-1:adams_pre_trn_epchs+1],
                        linestyle='dashed', linewidth=1, color=l1[0].get_color())

        for i in range(ADO_iters):
            l1 = plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs)+1,
                    cmbdTrn_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs], 
                    linestyle='solid', linewidth=1, label='ADO {} data'.format(i+1))
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs)+1,
                    cmbdTst_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs], 
                    linestyle='solid', linewidth=1,color='black')
            if adams_pre_trn_epchs!=0 or LBFGS_pre_trn_epchs!=0: # IE some pretraitorch.nning was done.
                plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs-1,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs+1)+1, 
                        cmbdTrn_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs+1],
                        linestyle='dashed', linewidth=1, color=l1[0].get_color())
                plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs-1,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs+1)+1, 
                        cmbdTst_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs+1],
                        linestyle='dashed', linewidth=1, color='black')
            if adams_pre_trn_epchs==0 and LBFGS_pre_trn_epchs==0 and i>=1: # No pretraining was done - NOTE I may not need this as the the would both be zero
                plt.plot(np.arange(i*ADO_epchs-1,i*ADO_epchs+1)+1, 
                        cmbdTrn_data_losses[i*ADO_epchs-1:i*ADO_epchs+1],
                        linestyle='dashed', linewidth=1, color=l1[0].get_color())
                plt.plot(np.arange(i*ADO_epchs-1,i*ADO_epchs+1)+1, 
                        cmbdTst_data_losses[i*ADO_epchs-1:i*ADO_epchs+1],
                        linestyle='dashed', linewidth=1, color='black')
        if adams_post_trn_epchs>0:
            l1 = plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs)+1, 
            cmbdTrn_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs],
            linestyle='solid', linewidth=1, label='PstTrn Adams')
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs)+1, 
            cmbdTst_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs],
            linestyle='solid', linewidth=1, color='black',)
            # now the dashed line so show the change from one method to the next one
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs-1,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+1)+1, 
                    cmbdTrn_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+1],
                    linestyle='dashed', linewidth=1, color=l1[0].get_color())
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs-1,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+1)+1, 
                    cmbdTst_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+1],
                    linestyle='dashed', linewidth=1, color='black')
        if LBFGS_post_trn_epchs>0:
            l1 = plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs+LBFGS_post_trn_epchs)+1, 
            cmbdTrn_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs: adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+LBFGS_post_trn_epchs],
            linestyle='solid', linewidth=1, label='PstTrn LBFGS')
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs+LBFGS_post_trn_epchs)+1, 
            cmbdTst_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs: adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+LBFGS_post_trn_epchs],
            linestyle='solid', linewidth=1, color='black',)
            # now the dashed line so show the change from one method to the next one
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs-1, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs+1)+1, 
                    cmbdTrn_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs+1],
                    linestyle='dashed', linewidth=1, color=l1[0].get_color())
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs-1, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs+1)+1, 
                    cmbdTst_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs+1],
                    linestyle='dashed', linewidth=1, color='black')

        # Now that the lines have been plotted, plot some dots to indicate where things transition over from one training step/method to the next
        plt.scatter(x=1, y=cmbdTrn_data_losses[0], s=15, c=1000)
        if adams_pre_trn_epchs!=0 and LBFGS_pre_trn_epchs!=0:
            plt.scatter(x=adams_pre_trn_epchs+1, y=cmbdTrn_data_losses[adams_pre_trn_epchs], s=15, c=1000)
        for i in range(ADO_iters):
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs +1, y=cmbdTrn_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs], s=15, c=1000)
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs +1, y=cmbdTst_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs], s=15, c=1000)
        if adams_post_trn_epchs>0:
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs +1, y=cmbdTrn_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs], s=15, c=1000)
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs +1, y=cmbdTst_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs], s=15, c=1000)
        if LBFGS_post_trn_epchs>0:
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+1, y=cmbdTrn_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs], s=15, c=1000)
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+1, y=cmbdTst_data_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs], s=15, c=1000)

        plt.xlabel('(Super) epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title('Data Loss: '+plot_title, fontsize=font_size)
        dataLs_fig.savefig(fname=file_name+'DataLosses.png', format='png')
        
        # plot for "data" loss
        EqLs_fig = plt.figure(figsize=(18, 10))
        if adams_pre_trn_epchs>0:
            plt.plot(np.arange(adams_pre_trn_epchs)+1, cmbdTst_collo_losses[:adams_pre_trn_epchs], linestyle='solid', linewidth=1, color='black', label='Test Data Losses')
            plt.plot(np.arange(adams_pre_trn_epchs)+1, cmbdTrn_collo_losses[:adams_pre_trn_epchs], linestyle='solid', linewidth=1, label='Pretrn Adams')
        if LBFGS_pre_trn_epchs>0:
            if adams_pre_trn_epchs==0: 
                plt.plot(np.arange(adams_pre_trn_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs)+1, cmbdTst_collo_losses[adams_pre_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs], linestyle='solid', linewidth=1, color='black', label='Test Data Losses')
            else: 
                plt.plot(np.arange(adams_pre_trn_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs)+1, cmbdTst_collo_losses[adams_pre_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs], linestyle='solid', linewidth=1, color='black')
            l1 = plt.plot(np.arange(adams_pre_trn_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs)+1, cmbdTrn_collo_losses[adams_pre_trn_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs], linestyle='solid', linewidth=1, label='Pretrn LBFGS')
            if adams_pre_trn_epchs>0:
                plt.plot(np.arange(adams_pre_trn_epchs-1,adams_pre_trn_epchs+1)+1, 
                        cmbdTrn_collo_losses[adams_pre_trn_epchs-1:adams_pre_trn_epchs+1],
                        linestyle='dashed', linewidth=1, color=l1[0].get_color())

        for i in range(ADO_iters):
            l1 = plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs)+1,
                    cmbdTrn_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs], 
                    linestyle='solid', linewidth=1, label='ADO {} data'.format(i+1))
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs)+1,
                    cmbdTst_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs], 
                    linestyle='solid', linewidth=1,color='black')
            if adams_pre_trn_epchs!=0 or LBFGS_pre_trn_epchs!=0: # IE some pretraitorch.nning was done.
                plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs-1,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs+1)+1, 
                        cmbdTrn_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs+1],
                        linestyle='dashed', linewidth=1, color=l1[0].get_color())
                plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs-1,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs+1)+1, 
                        cmbdTst_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs+1],
                        linestyle='dashed', linewidth=1, color='black')
            if adams_pre_trn_epchs==0 and LBFGS_pre_trn_epchs==0 and i>=1: # No pretraining was done - NOTE I may not need this as the the would both be zero
                plt.plot(np.arange(i*ADO_epchs-1,i*ADO_epchs+1)+1, 
                        cmbdTrn_collo_losses[i*ADO_epchs-1:i*ADO_epchs+1],
                        linestyle='dashed', linewidth=1, color=l1[0].get_color())
                plt.plot(np.arange(i*ADO_epchs-1,i*ADO_epchs+1)+1, 
                        cmbdTst_collo_losses[i*ADO_epchs-1:i*ADO_epchs+1],
                        linestyle='dashed', linewidth=1, color='black')
        if adams_post_trn_epchs>0:
            l1 = plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs)+1, 
            cmbdTrn_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs],
            linestyle='solid', linewidth=1, label='PstTrn Adams')
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs)+1, 
            cmbdTst_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs],
            linestyle='solid', linewidth=1, color='black',)
            # now the dashed line so show the change from one method to the next one
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs-1,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+1)+1, 
                    cmbdTrn_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+1],
                    linestyle='dashed', linewidth=1, color=l1[0].get_color())
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs-1,adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+1)+1, 
                    cmbdTst_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+1],
                    linestyle='dashed', linewidth=1, color='black')
        if LBFGS_post_trn_epchs>0:
            l1 = plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs+LBFGS_post_trn_epchs)+1, 
            cmbdTrn_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs: adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+LBFGS_post_trn_epchs],
            linestyle='solid', linewidth=1, label='PstTrn LBFGS')
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs + adams_post_trn_epchs+LBFGS_post_trn_epchs)+1, 
            cmbdTst_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs: adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+LBFGS_post_trn_epchs],
            linestyle='solid', linewidth=1, color='black',)
            # now the dashed line so show the change from one method to the next one
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs-1, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs+1)+1, 
                    cmbdTrn_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs+1],
                    linestyle='dashed', linewidth=1, color=l1[0].get_color())
            plt.plot(np.arange(adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs-1, adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs+1)+1, 
                    cmbdTst_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs-1:adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+ adams_post_trn_epchs+1],
                    linestyle='dashed', linewidth=1, color='black')

        # Now that the lines have been plotted, plot some dots to indicate where things transition over from one training step/method to the next
        plt.scatter(x=1, y=cmbdTrn_collo_losses[0], s=15, c=1000)
        if adams_pre_trn_epchs!=0 and LBFGS_pre_trn_epchs!=0:
            plt.scatter(x=adams_pre_trn_epchs+1, y=cmbdTrn_collo_losses[adams_pre_trn_epchs], s=15, c=1000)
        for i in range(ADO_iters):
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs +1, y=cmbdTrn_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs], s=15, c=1000)
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs +1, y=cmbdTst_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+i*ADO_epchs], s=15, c=1000)
        if adams_post_trn_epchs>0:
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs +1, y=cmbdTrn_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs], s=15, c=1000)
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs +1, y=cmbdTst_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs], s=15, c=1000)
        if LBFGS_post_trn_epchs>0:
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+1, y=cmbdTrn_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs], s=15, c=1000)
            plt.scatter(x=adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs+1, y=cmbdTst_collo_losses[adams_pre_trn_epchs+LBFGS_pre_trn_epchs+(i+1)*ADO_epchs+adams_post_trn_epchs], s=15, c=1000)

        plt.xlabel('(Super) epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title('Diff. EQ Loss: '+plot_title, fontsize=font_size)
        EqLs_fig.savefig(fname=file_name+'DiffEqLosses.png', format='png')
        os.chdir(start_dir)
        return None

    def FVU_Plot(self,dataset_name:str, file_name:str, save_fig:bool=True)->None:
        """
        
        """
        library_names = self.lib_func.get_library_names(net_out_func_names=self.ntwk_out_names)
        n_lib = len(library_names)
        n_trgts = self.lmbda.data.size(1)
        # get someway to identify at the time that this function was run what training methods have been completed. 
        # NOTE: MAYBE A BETTER IDEA TO HAVE A PRIVATE CLASS LIST IS ONLY UPDATED WHEN A TRAINING METHOD IS RUN TO IDENTIFY WHICH TRAINING METHODS WERE USED AND THE ORDER
        # as I am working under the assumption that the things have been run in the poper order of pretrainings, ADO then post trainings
        trn_ids = np.array([0 if itm is None else 1 for itm in [self.AdamsPreTrnLambdas, self.LbfgsPreTrnLambdas, self._ADO_lambdas, self.AdamsPstTrnLambdas, self.LbfgsPstTrnLambdas]])
        if trn_ids.sum()==0:
            raise RuntimeError("Training has not be done in any form for this learning model")
        lmb_vals_plc_hldr = np.empty(shape=self.shape)
        FVU_plc_hldr = np.empty(shape=(n_trgts,),)
        headings = ['Init.']
        if trn_ids[0] == 1: # Adams pretraining was done 
            # print('Adams Pre-Training was done to get these results')
            headings.append('Adams PrTrn')
            lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
            FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.flatten()), axis=0)
        if trn_ids[1] == 1: # LBFGS pretraining was done 
            # print('LBFGS Pre-Training was done to get these results')
            headings.append('LFBGS PrTrn')
            if trn_ids[:1].sum()==0: # IF this is true than this traing method/step was the first one done.
                lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.flatten()), axis=0)
            else: 
                lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self[1]), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self[1]), axis=0)
        if trn_ids[2] == 1: # ADO was done 
            # print('ADO Training was done to get these results')
            for i in range(self+1):
                headings.append('ADO '+str(i))
            # if self._ADO_iters==0: self._ADO_iters=None
            if trn_ids[:2].sum()==0: # IF this is true than this traing method/step was the first one done.
                lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.flatten()), axis=0)
            else: 
                lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self[1:].reshape((-1,1),order='C').reshape((n_lib,self+1),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self[1:].flatten()), axis=0)
        if trn_ids[3] == 1: # Adams post training was done
            # print('Adams Post Training was done to get these results')
            headings.append('Adams PostTrn')
            if trn_ids[:3].sum()==0: # IF this is true than this traing method/step was the first one done.
                lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self.AdamsPstTrnLambdas.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.AdamsPstTrnFvus.flatten()), axis=0)
            else: 
                lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self.AdamsPstTrnLambdas[1]), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.AdamsPstTrnFvus[1]), axis=0)
        if trn_ids[4] == 1: # Adams post training was done 
            # print('LBFGS Post Training was done to get these results')
            headings.append('LBFGS PostTrn')
            if trn_ids[:4].sum()==0: # IF this is true than this traing method/step was the first one done.
                lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self.LbfgsPstTrnLambdas.reshape((-1,1),order='C').reshape((n_lib,2),order='F')), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.LbfgsPstTrnFvus.flatten()), axis=0)
            else: 
                lmb_vals_plc_hldr = np.concatenate((lmb_vals_plc_hldr, self.LbfgsPstTrnLambdas[1]), axis=1)
                FVU_plc_hldr = np.concatenate((FVU_plc_hldr, self.LbfgsPstTrnFvus[1]), axis=0)

        lmb_vals = lmb_vals_plc_hldr[:,n_trgts:]
        FVU_vals = FVU_plc_hldr[n_trgts:]

        num_trn_stps = len(headings)
        eqs = [self.Learned_EQ(coefs=lmb_vals[:, n_trgts*i:n_trgts*(i+1)], output=False, sup_zeros=True, dec_rnd=5, prnt_sig_dif=3) for i in range(0, num_trn_stps)]
        idx = headings.index("ADO 0")

        # if FVU_vals[k]<=FVU_vals[k-1] and FVU_vals[k]<=FVU_vals[k+1]:
        #     verticalalignment = 'top'
        #     horizontalalignment = 'center'
        # elif FVU_vals[k]>=FVU_vals[k-1] and FVU_vals[k]>=FVU_vals[k+1]:
        #     verticalalignment = 'bottom'
        #     horizontalalignment = 'center'
        # elif FVU_vals[k-1]>=FVU_vals[k]>=FVU_vals[k+1]:
        #     verticalalignment = 'bottom'
        #     horizontalalignment = 'left'
        # elif FVU_vals[k-1]<=FVU_vals[k]<=FVU_vals[k+1]:
        #     verticalalignment = 'top'
        #     horizontalalignment = 'left'

        fig, axis = plt.subplots(nrows=1, ncols=1, )
        x_vals = np.arange(1, len(headings)+1)
        fig.set_size_inches(w=8, h=8)
        axis.semilogy(x_vals, FVU_vals, '.-', markersize=10)
        # axis.plot(np.arange(len(headings))+1, FVU_vals)
        axis.set_xlim(0, len(headings)+1)
        axis.set_xticks(ticks=x_vals, labels=headings, rotation='vertical',)
        axis.set_title('FVU Evolution')
        axis.set_ylabel('FVU')
        # align_dict = {'verticalalignment':'bottom', 'horizontalalignment':'left'}
        # axis.text(x=x_vals[idx+1], y=FVU_vals[idx+1], s=eqs[idx+1], verticalalignment='bottom', horizontalalignment='left')

        OG_cwd = os.getcwd()
        try:
            os.mkdir('FvuEquationPlots')
        except FileExistsError:
            print('FvuEquationPlots Directory already exists so did not create it')
        try: 
            os.chdir('FvuEquationPlots')
        except (OSError, FileNotFoundError,PermissionError,NotADirectoryError):
            print('Could not change directory to FvuEquationPlots. Results will be writen to file in CWD={}'.format(os.getcwd()))
        # try and get in to the directory for this dataset
        try:
            os.mkdir(dataset_name)
        except FileExistsError:
            print('{} Directory already exists so did not create it'.format(dataset_name))
        try: 
            os.chdir(dataset_name)
        except (OSError, FileNotFoundError,PermissionError,NotADirectoryError):
            print('Could not change directory to {}. Animated Learned Plots will be writen to mp4 file in CWD={}'.format(dataset_name, os.getcwd()))
        if save_fig:
            fig.savefig(fname=file_name+'.png', format='png')
        try: 
            os.chdir(OG_cwd)
        except (OSError, FileNotFoundError, PermissionError, NotADirectoryError):
            print('Could change back to the original directory but program is done so oh well')
        
    def Loss_Score_Complexity_Plot(self, dataset_name:str, save_dir_name:str, save_figs:bool=True)->None:
        """
        
        """
        sv_dir = os.path.join("ComplexityVersusPlots", dataset_name, save_dir_name)
        os.makedirs(name=sv_dir, exist_ok=True)

        for k in range(self.lmbda.data.shape[1]):
            sv_name = os.path.join(sv_dir, f"LearnedEq{k+1}Plot.png")
            fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), layout="constrained")
            axis.semilogy(self._model_complxtys[k], self._model_losses[k], "r.-", label="losses")
            axis.semilogy(self._model_complxtys[k][1:], self._model_scores[k], "bx-", label="score")
            axis.set_xlabel("complexitity value")
            axis.set_xlim(-2, self._model_complxtys[k].max()+1)
            axis.set_ylabel("")
            axis.legend()

            axis.set_title(f"Learned EQ. Num. {k+1} - Models Losses and Scores vs Complexity")
            if save_figs:
                fig.savefig(fname=sv_name, format='png')

        return None


class OldEqLearner1D(OldEqLearner):

    def __init__(self, 
            net, 
            Lmbda, 
            lib_func, 
            sprs_slvr, 
            data_dict, 
            tmprl_ords, 
            col_pnts_smplr = Rand_Col_Sampler(), 
            N_col_pnts = 10000, 
            ntwk_out_names:List[str]=None,
            device = torch.device('cpu'), 
            data_type = None):
        super().__init__(net, Lmbda, lib_func, sprs_slvr, data_dict, tmprl_ords, col_pnts_smplr, N_col_pnts, ntwk_out_names, device, data_type)

    def AnimatedPlot(self, dataset_name:str, spdx:float, pts:Union[np.ndarray,torch.Tensor], values:Union[np.ndarray,torch.Tensor], 
                     ani_title:str='UntiltedAnimatedPlot', fig_title:str='', state:str=''):
        """
        Function that creates an animated plot of the data that has been given in the pts
        and values tensors/arrays (values vs pts) along side the function that was learned.
        The values of the function that were learned are determined by passing the pts 
        through the torch.nn. The animation plot is saved as a .gif file.
        Input arguments are the following:
            * pts - A K by 2 tensor/numpy array that contains the spatiotemporal (x,t) points 
                    where the values in the values array have been determined at. 
                    The order of the points should be like the following:
                      (x_1,t_1), (x_2,t_1),...,(x_N,t_1), (x_1,t_2),...(x_N,t_M)
                      where x_j < x_i for j<i and t_k < t_l for k < l
                    Ideally this function would be ran by passing X_trn or
                    X_tst tensor/array found in the LearningMatData1D DataSet classes.
            * values - A tensor/numpy array that contains the evaluations of the
                    spatiotemporal points (x,t) points under some function 
                    which ideally is the one that is trying to be learned.  
                    the values array have been determined at. 
                    The ordering of the values should be like the following:
                      u(x_1,t_1), u(x_2,t_1),...,u(x_N,t_1), u(x_1,t_2),...u(x_N,t_M)
                      where x_j < x_i for j<i and t_k < t_l for k < l and u is 
                      the function to the learned
                    Ideally this function would be ran by passing u_trn or
                    u_tst tensor/array found in the LearningMatData1D DataSet classes.
            * ani_title - The title of the saved animation .gif file. 
            * fig_title - The figure title of the animated plot. Just needs to be a string
                    argument. If no titled is wanted just pass it the value of ''
            * state - Animated plots title will be state + Learned Equation/Model at t = 
                    and so this argument can be any string that you would like to replace
                    the state word or not thing at all (i.e state = '')
        """
            
        def learned_animation(i, ax, pts:np.ndarray, exact:np.ndarray, maxs:list, mins:list,t:np.array, dgt:int, trn_state:str):
            k = int(i*spdx)
            n_t = t.shape[0]
            n_x = int(pts.shape[0]/n_t)
            x = pts[k*n_x:(k+1)*n_x,0:1]
            # x.sort()
            lrn = self.net(torch.from_numpy(pts[k*n_x:(k+1)*n_x,:]).to(device=self.device, dtype=self.data_type, non_blocking=True)).cpu().detach().numpy()
            x_min, y_min = mins
            x_max, y_max = maxs
            ax.clear()
            ax.plot(x, exact[k*n_x:(k+1)*n_x:, :], color='blue', marker='o', linestyle='solid', linewidth=1, markersize=2, label='exact')
            ax.plot(x, lrn, color='red', marker='*', linestyle='solid', linewidth=1, markersize=2, label='lrnd')
            ax.set_xlabel('x')
            ax.set_xlim(left=x_min, right=x_max)
            ax.set_ylabel('u(x,t)')
            ax.set_ylim(bottom=y_min, top=y_max)
            ax.set_title(trn_state+' Learned Equation/Model at t = {}'.format(t[k].round(dgt)))
            ax.legend()

        # Something to make sure that we have the prediction and the exact values on the cpu and in numpy arrays for ploting
        if isinstance(values, torch.Tensor):vals = values.cpu().detach().numpy()
        elif isinstance(values, np.ndarray):vals = np.copy(values)
        else: raise TypeError(f"The values function argument needs to be either a torch.Tensor or numpy.ndarray object not a {type(values).__name__} object as was given!")
        # N by 2 array/tensors these need to be. 
        if isinstance(pts, torch.Tensor):X = pts.cpu().detach().numpy()
        elif isinstance(pts, np.ndarray): X = np.copy(pts)
        else:raise TypeError(f"The pts function argument needs to be either a torch.Tensor or numpy.ndarray object not a {type(pts).__name__} object as was given!")
        
        sv_dir = os.path.join('AnimatedPlotsFigs', dataset_name)
        os.makedirs(name=sv_dir, exist_ok=True)
    
        t = np.unique(X[:, 1])
        t.sort()
        rnd = int(np.log10(np.min(t[1:]-t[:-1]))) + 3
        y_max = np.ceil(vals.max())
        y_min = np.floor(vals.min())
        x_max, x_min = np.ceil(pts[:, 0].max()), np.floor(pts[:,0].min())
        mins = [x_min, y_min]
        maxs = [x_max, y_max]
        
        plot_fig, ax = plt.subplots()
        plot_fig.set_size_inches(w=10, h=8)
            
        ani_plot = animation.FuncAnimation(
            plot_fig, learned_animation, fargs=(ax, X, vals, maxs, mins, t, rnd, state), save_count=50, 
            frames=int(t.shape[0]/spdx), interval=100, repeat=True, blit=False)
        plot_fig.suptitle(fig_title,fontsize=10)
        writer = animation.FFMpegWriter(fps=30, codec='mpeg4', metadata=dict(artist='Me'), bitrate=-1)
        # ani_plot.save(ani_title+'.gif', writer=writer)
        ani_plot.save(os.path.join(sv_dir, ani_title+'.gif'), writer=writer)
        
        return None

    def ContourLikeComparisonPlot(self, dataset_name:str, pts:Union[np.ndarray,torch.Tensor], values:Union[np.ndarray,torch.Tensor], 
                                  fig_title:str='', state:str='', show_fig:bool=False, 
                                  save_fig:bool=True, save_title:str='',
                                  **kwargs)->None:
        """
        DESCRIPTIVE TEXT GOES HERE EVENTUALLY DESCRIBING THE FUNC ARGUMENTS as well as input parameter checking
        The values are expected to be N by 
        """
        
        
        # Something to make sure that we have the prediction and the exact values on the cpu and in numpy arrays for ploting
        if isinstance(values, torch.Tensor):vals = values.cpu().detach().numpy()
        elif isinstance(values, np.ndarray):vals = np.copy(values)
        else: raise TypeError(f"The values function argument needs to be either a torch.Tensor or numpy.ndarray object not a {type(values).__name__} object as was given!")
        # N by 2 array/tensors these need to be. 
        if isinstance(pts, torch.Tensor):X = pts.cpu().detach().numpy()
        elif isinstance(pts, np.ndarray): X = np.copy(pts)
        else:raise TypeError(f"The pts function argument needs to be either a torch.Tensor or numpy.ndarray object not a {type(pts).__name__} object as was given!")
        
        sv_dir = os.path.join('LearnedEquationPlots', dataset_name)
        os.makedirs(name=sv_dir, exist_ok=True)
        
        ## if needed, check the kwargs keys - if end up using additional keyword argument
        #  
        if 'dif' in kwargs.keys() and isinstance(kwargs['dif'], bool):
            dif = kwargs['dif']
        else:
             dif = False
        if 'exact' in kwargs.keys() and isinstance(kwargs['exact'], bool):
            exact = kwargs['exact']
        else:
             exact = False
        if 'learned' in kwargs.keys() and isinstance(kwargs['learned'], bool):
            learned = kwargs['learned']
        else:
            learned = False

        t = np.unique(X[:, 1])
        t.sort()
        n_t = t.shape[0]
        n_x = int(X.shape[0]/n_t)
        preds = np.empty_like(vals)
        # This is done just in case number of points is so larger that they catorch.nnot all be placed on the device at the same time
        # do to memory constrains 
        for i in range(n_t):
            preds[i*n_x:(i+1)*n_x,:] = self.net(torch.from_numpy(X[i*n_x:(i+1)*n_x,:]).to(device=self.device, dtype=self.data_type, non_blocking=True)).cpu().detach().numpy()
        preds = preds.reshape((n_t, n_x, ))
        vals = vals.reshape((n_t, n_x, ))
        diff = np.absolute(preds - vals)
        T = X[:,1].reshape((n_t, n_x, ))
        XX = X[:,0].reshape((n_t, n_x, ))

        # Now create the plots/figures and save them. 
        # First the figure that has all of the other plots within them.
        fig = plt.figure(figsize=(16,10), layout="constrained")
        mosaic = """AB;CC"""
        axs =  fig.subplot_mosaic(mosaic) # axs here is a dictionary with keywords being A, B and C see matplotlib mosaic for details
        c_map = 'jet'

        if vals.min()<0 and vals.max()<=0: norm = Normalize(vmin=vals.min(), vmax=0.0)
        elif vals.min()>=0 and vals.max()>0: norm = Normalize(vmin=0.0, vmax=vals.max())
        else: norm = TwoSlopeNorm(0.0, vals.min(), vals.max())
        c = axs["A"].imshow(vals, cmap=c_map, norm=norm,
                      extent=[XX.min(), XX.max(), T.min(), T.max()],
                      interpolation='bilinear', origin='lower', aspect='auto')
        axs["A"].set_title('True/Actual Function')
        axs["A"].set_xlabel('x')
        axs["A"].set_ylabel('t')
        fig.colorbar(c, ax=axs["A"], location='right', orientation='vertical',)
        # NOTE: that this is a problem when there are nan values in preds or any of the matrices as nan is always < & and > 0.0
        if preds.min()<0 and preds.max()<=0: norm = Normalize(vmin=preds.min(), vmax=0.0)
        elif preds.min()>=0 and preds.max()>0: norm = Normalize(vmin=0.0, vmax=preds.max())
        else: norm = TwoSlopeNorm(0.0, preds.min(), preds.max())    # This block is entered when preds has a nan value also does not through and error
        c = axs["B"].imshow(preds, cmap=c_map, norm=norm,
                      extent=[XX.min(), XX.max(), T.min(), T.max()],
                      interpolation='bilinear', origin='lower', aspect='auto')
        axs["B"].set_title('Pinns Learned Equation/Model '+state)
        axs["B"].set_xlabel('x')
        axs["B"].set_ylabel('t')
        fig.colorbar(c, ax=axs["B"], location='right', orientation='vertical',)

        if preds.min()<0 and preds.max()<=0: norm = Normalize(vmin=preds.min(), vmax=0.0)
        elif preds.min()>=0 and preds.max()>0: norm = Normalize(vmin=0.0, vmax=preds.max())
        else: norm = TwoSlopeNorm(0.0, preds.min(), preds.max())
        c = axs["C"].imshow(diff, cmap=c_map, norm=norm,
                      extent=[XX.min(), XX.max(), T.min(), T.max()],
                      interpolation='bilinear', origin='lower', aspect='auto')
        axs["C"].set_title('Absolute Difference')
        axs["C"].set_xlabel('x')
        axs["C"].set_ylabel('t')
        fig.colorbar(c, ax=axs["C"], location='right', orientation='vertical',)

        fig.suptitle(fig_title)
        if show_fig: 
            plt.show()
        if save_fig: 
            fig.savefig(fname=os.path.join(sv_dir, save_title+'.png'), format='png')

        # Now create and save the figure of just the actual/exact data given the keyword argument value. 
        if exact:
            fig, axis = plt.subplots(nrows=1, ncols=1, layout="constrained", figsize=(8,5))
            if vals.min()<0 and vals.max()<=0: norm = Normalize(vmin=vals.min(), vmax=0.0)
            elif vals.min()>=0 and vals.max()>0: norm = Normalize(vmin=0.0, vmax=vals.max())
            else: norm = TwoSlopeNorm(0.0, vals.min(), vals.max())
            c = axis.imshow(vals, cmap=c_map, norm=norm,
                          extent=[XX.min(), XX.max(), T.min(), T.max()],
                          interpolation='bilinear', origin='lower', aspect='auto')
            axis.set_title('True/Actual Function')
            axis.set_xlabel('x')
            axis.set_ylabel('t')
            fig.colorbar(c, ax=axis, location='right', orientation='vertical',)
            if save_fig: 
                fig.savefig(fname=os.path.join(sv_dir, save_title+'FunctionData.png'), format='png')
        # Now create and save the figure of just the difference between the true/exact data and the learned 
        # function/solution/equation using given dif keyword argument value. 
        if dif:
            fig, axis = plt.subplots(nrows=1, ncols=1, layout="constrained", figsize=(8,5))
            if preds.min()<0 and preds.max()<=0: norm = Normalize(vmin=preds.min(), vmax=0.0)
            elif preds.min()>=0 and preds.max()>0: norm = Normalize(vmin=0.0, vmax=preds.max())
            else: norm = TwoSlopeNorm(0.0, preds.min(), preds.max())
            c = axis.imshow(diff, cmap=c_map, norm=norm,
                          extent=[XX.min(), XX.max(), T.min(), T.max()],
                          interpolation='bilinear', origin='lower', aspect='auto')
            axis.set_title('Absolute Difference')
            axis.set_xlabel('x')
            axis.set_ylabel('t')
            fig.colorbar(c, ax=axis, location='right', orientation='vertical',)
            if save_fig: 
                fig.savefig(fname=os.path.join(sv_dir, save_title+'DifInExactandLrned.png'), format='png')
        # Now create and save the figure of just the learned equation/solution/function ussing the given 
        # learned keyword argument value. 
        if learned:
            fig, axis = plt.subplots(nrows=1, ncols=1, layout="constrained", figsize=(8,5))
            if preds.min()<0 and preds.max()<=0: norm = Normalize(vmin=preds.min(), vmax=0.0)
            elif preds.min()>=0 and preds.max()>0: norm = Normalize(vmin=0.0, vmax=preds.max())
            else: norm = TwoSlopeNorm(0.0, preds.min(), preds.max())    # This block is entered when preds has a nan value also does not through and error
            c = axis.imshow(preds, cmap=c_map, norm=norm,
                          extent=[XX.min(), XX.max(), T.min(), T.max()],
                          interpolation='bilinear', origin='lower', aspect='auto')
            axis.set_title('Pinns Learned Equation/Model '+state)
            axis.set_xlabel('x')
            axis.set_ylabel('t')
            fig.colorbar(c, ax=axis, location='right', orientation='vertical',)
            if save_fig: 
                fig.savefig(fname=os.path.join(sv_dir, save_title+'LearnedFunc.png'), format='png')
        
        return None

