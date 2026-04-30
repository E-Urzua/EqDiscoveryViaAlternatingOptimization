import os
from typing import List, Tuple, Union
from copy import deepcopy
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
from time import perf_counter
import scipy.io as sio
import matplotlib.pyplot as plt
import torch

from PartialDerivFunctions import Nth_temporal_prtls
from trackers import LossTracker
from extra_loss_functions import Torch_Lp_Loss
from EqLearner import EqLearner1D
from func_libraries import Poly_Deriv_Library
from sparse_regress_algs import SparseRegressAlg, RFE, Cross_Val_RFE, Cross_Val_RFE_V2, SSR
from data_sampling import Rand_Col_Sampler
from PinnSrPlusPlusComparisonSettings import DefaultEQsModelParameters, OptimizedEQsModelParameters 
from data_loaders import PDELearningMatDataVerB, PDELearningMatData
import Networks
# from PinnSrPlusEqSolutions import OneDimSols
from OneDimSols import OneDimSols


##### NOTE #####
# the network classes are being create here instead of importing them from the other file since
# I have found the code to run faster when it is all in this file and when I am not implenting
# a few different classes that call some other class. 

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

class KlnGrd_Net(torch.nn.Module):

    def __init__(self, 
                in_dim:int,
                out_dim:int,
                n_hid_lyr:int,
                nodes_per_lyr:int):

        super().__init__()
        # network needs to match what was used in the normal plain old MLP results 
        lyrs_lst = [torch.nn.Linear(in_features=in_dim, out_features=nodes_per_lyr, bias=True)]
        torch.nn.init.xavier_normal_(lyrs_lst[-1].weight.data, gain=1.41)
        lyrs_lst.append(torch.nn.Tanh())
        for _ in range(n_hid_lyr):
            lyrs_lst.append(torch.nn.Linear(in_features=nodes_per_lyr, out_features=nodes_per_lyr, bias=True))
            torch.nn.init.xavier_normal_(lyrs_lst[-1].weight.data, gain=1.41)
            lyrs_lst.append(torch.nn.Tanh())
        lyrs_lst.append(torch.nn.Linear(in_features=nodes_per_lyr, out_features=out_dim, bias=True))
        torch.nn.init.xavier_normal_(lyrs_lst[-1].weight.data, gain=1.41)
        self.ntwrk = torch.nn.Sequential(*lyrs_lst)

    def forward(self, x:torch.Tensor):
        """
            B.C are that U(-1,t)=U(1,t)=0 for all t. So force this, we multiply the 
            output of the network by (x^2 - 1)=(x-1)(x+1)
        """
        return self.ntwrk(x)*(x[:,0:1].pow(2) - 1)

class PeriodicBC_Network(torch.nn.Module):

    def __init__(self, 
                in_dim:int,
                out_dim:int,
                n_hid_lyr:int,
                nodes_per_lyr:int,
                period:float):

        super().__init__()
        # network needs to match what was used in the normal plain old MLP results 
        lyrs_lst = [torch.nn.Linear(in_features=in_dim+1, out_features=nodes_per_lyr, bias=True)]
        torch.nn.init.xavier_normal_(lyrs_lst[-1].weight.data, gain=1.41)
        lyrs_lst.append(torch.nn.Tanh())
        for _ in range(n_hid_lyr):
            lyrs_lst.append(torch.nn.Linear(in_features=nodes_per_lyr, out_features=nodes_per_lyr, bias=True))
            torch.nn.init.xavier_normal_(lyrs_lst[-1].weight.data, gain=1.41)
            lyrs_lst.append(torch.nn.Tanh())
        lyrs_lst.append(torch.nn.Linear(in_features=nodes_per_lyr, out_features=out_dim, bias=True))
        torch.nn.init.xavier_normal_(lyrs_lst[-1].weight.data, gain=1.41)
        self.ntwrk = torch.nn.Sequential(*lyrs_lst)

        self.prd = period

    def forward(self, x:torch.Tensor):
        """
        
        """

        embedded = torch.hstack([ torch.cos(self.prd*x[:,0:1]), torch.sin(self.prd*x[:,0:1]), x[:,1:]])

        return self.ntwrk(embedded)

class Special_Sampler_Lrnr(EqLearner1D):

    """
        All this does is changing the Adam and LBFGS trainnig method 
        so that the collocation points are resampled every epoch of 
        training instead of just resampling at the start of each 
        training method.
    """

    def __init__(self, 
            net, 
            Lmbda, 
            lib_func, 
            sprs_slvr, 
            data_dict, 
            tmprl_ords, 
            col_pnts_smplr, 
            col_bnds:torch.Tensor,
            N_col_pnts = 10000, 
            ntwk_out_names:List[str]=None,
            device = torch.device('cpu'), 
            data_type = None):
        super().__init__(net, Lmbda, lib_func, sprs_slvr, data_dict, tmprl_ords, None, N_col_pnts, ntwk_out_names, device, data_type)

        self.col_bnds = torch.from_numpy(col_bnds).to(device=self.device, dtype=self.data_type)

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
        # np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        # colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        self.nDpnts = train_inputs.size(0)
        ndims = test_targets.shape[-1]
        
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
            # colpnts.grad = None
            colpnts = torch.rand(size=(self.N_col_pnts,ndims), device=self.device, dtype=self.data_type,) *(self.col_bnds[1] - self.col_bnds[0]) + self.col_bnds[0]
            colpnts.requires_grad_(True)
                
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
        # np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        # colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
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
            colpnts = torch.from_numpy(self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)).to(device=self.device, dtype=self.data_type).requires_grad_(True)
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
        # np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        # colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
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
            colpnts = torch.from_numpy(self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)).to(device=self.device, dtype=self.data_type).requires_grad_(True)
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
        ndims = test_inputs.shape[-1]

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
            
            # np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
            # colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
            # if not colpnts.requires_grad:
            #     colpnts.requires_grad_(True)
            colpnts = torch.rand(size=(self.N_col_pnts,ndims), device=self.device, dtype=self.data_type,) *(self.col_bnds[1] - self.col_bnds[0]) + self.col_bnds[0]
            colpnts.requires_grad_(True)

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
        # np_colpnts = self.col_pnts_smplr.sample(n_pnts=self.N_col_pnts)
        # colpnts = torch.from_numpy(np_colpnts).to(device=self.device, dtype=self.data_type).requires_grad_(True)
        colpnts = torch.rand(size=(self.N_col_pnts,ndims), device=self.device, dtype=self.data_type,) *(self.col_bnds[1] - self.col_bnds[0]) + self.col_bnds[0]
        colpnts.requires_grad_(True)
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
    

def main(dflag:int, ver:int, data_slct_type:str, jobId:int, arrayId:int, node:str):
    
    jobID , jobVer = jobId, arrayId
    dvc = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # dvc = torch.device('cpu')

    # (Dname, nDpnts, nTrn, nCpnts, noise, data_seed, sptl_ord, tmp_order, polyDeg, 
    #     numLyrsU, nPrLU, prd_stuff, four_stuff, rnd_wght_fct,
    #     ADO_iters, Kfolds, ADO_alphas, 
    #     min_epchs, max_epchs, lrn_rt, preAlpha, preGamma,
    #     min_pst_epchs, max_pst_epchs, pst_alpha, pst_lrn_rt) = DefaultEQsModelParameters(dFlag=dflag, ver=ver, run=jobVer, data_type=data_slct_type, computer=node)
    
    (Dname, nDpnts, nTrn, nCpnts, noise, data_seed, sptl_ord, tmp_order, polyDeg, 
        numLyrsU, nPrLU, prd_stuff, four_stuff, rnd_wght_fct,
        ADO_iters, Kfolds, ADO_alphas, 
        min_epchs, max_epchs, lrn_rt, preAlpha, preGamma,
        min_pst_epchs, max_pst_epchs, pst_alpha, pst_lrn_rt) = OptimizedEQsModelParameters(dFlag=dflag, ver=ver, run=jobVer, data_type=data_slct_type, computer=node)

    # slvr = RFE(alpha=torch.zeros(size=(1,), device=dvc), normalize=True, annealing_factor=2, criteria="coefficient_value", selection="individual", ADO_iters=5)
    # slvr = Cross_Val_RFE(Kfolds=Kfolds, alpha=torch.zeros(size=(1,), device=dvc), normalize=True, annealing_factor=2, criteria="coefficient_value", selection="individual", ADO_iters=ADO_iters)
    # slvr = Cross_Val_RFE_V2(Kfolds=Kfolds, alpha=torch.zeros(size=(1,), device=dvc), normalize=True, annealing_factor=2, criteria="coefficient_value", selection="individual", best=False, ADO_iters=ADO_iters, thrshld_val=1e-8)
    slvr = Cross_Val_RFE_V2(Kfolds=Kfolds, alpha=torch.zeros(size=(1,), device=dvc), normalize=True, annealing_factor=2, criteria="coefficient_value", selection="individual", best=False, ADO_iters=ADO_iters, thrshld_val=1e-8)
    # slvr = SSR(Kfolds=Kfolds, alpha=torch.zeros(size=(1,), device=dvc), percent_redux=0.70, criteria="coefficient_value", ADO_iters=ADO_iters, normalize=True)

    DataFile = '../Testing_DataSets/'+Dname+'.mat'
    data_set = (Dname + "_" + "N" + str(int(noise)) + "_" +
                    "P" + str(nDpnts))

    saveName = 'MoreSamplingResultsForSlurmJob'+str(jobID)+'ArrayNum'+str(jobVer)
    sv_mdl_name = 'LrnrForSlurmJob'+str(jobID)+'ArrayNum'+str(jobVer)
    # Optim Results Names
    # saveName = 'OptimalVer4ResultsForSlurmJob'+str(jobID)+'ArrayNum'+str(jobVer)
    # sv_mdl_name = 'OptimalVer4LrnrForSlurmJob'+str(jobID)+'ArrayNum'+str(jobVer)

    data = sio.loadmat(file_name=DataFile)
    x = data['x'].flatten()
    t = data['t'].flatten()
    u = data['usol']

    # n_x = x.shape[0]
    # n_t = t.shape[0]

    [X, T] = np.meshgrid(x, t)
    if u.shape!=X.shape:
        u = u.T

    pts = np.concatenate((X.reshape((-1,1),order='C'), T.reshape((-1,1),order='C')), axis=1)
    U = u.reshape((-1,1), order='C')

    (d_seed, Nsp, Ns, subsample_prcntg,
                X_trn, U_trn,
                X_tst, U_tst,
                bounds) = PDELearningMatDataVerB(fname=DataFile, Sptldims=1, Ntrn=nDpnts, Ntst=nDpnts//4, noisePrcntg=noise, 
                     seed=None, to_float=True,)

    # (d_seed, _, Ns, subsample_prcntg,
    #             X_trn, U_trn,
    #             X_tst, U_tst,
    #             bounds) = PDELearningMatData(fname=DataFile, Sptldims=1, split=0.80, smpleprcnt=0.20, noisePrcntg=noise, 
    #                  seed=None, to_float=True, N_trn_pnts=nDpnts)

    res_dict = {'jobID':jobID, 'jobVer':jobVer, 'node':node, 'subsample_prcntg':subsample_prcntg, 'noisePrcntg':noise, 'NpSeed':d_seed, 'Ns':Ns}

    libr = Poly_Deriv_Library(
            poly_degree=polyDeg,
            include_poly_interaction=True,
            poly_interaction_only=False,
            derivative_order=sptl_ord,
            sptl_dims=1,
            include_bias=False,
            include_deriv_interaction=True,
            multi_indices=None,
            device=dvc,
            data_type=torch.float32)

    d_dict = {"Train_Inputs":X_trn, "Train_Targets":U_trn, "Test_Inputs":X_tst, "Test_Targets":U_tst}

    col_smplr = Rand_Col_Sampler(sampler='halton', 
                                 dims=2, seed=None, 
                                 bounds=bounds, requires_grad=True, 
                                 device=dvc, data_type=torch.float32)
    col_smplr = Rand_Col_Sampler(sampler='halton', 
                                 dims=2, seed=None, 
                                 bounds=bounds, requires_grad=True, 
                                 device=dvc, data_type=torch.float32)

    if dflag==14:
        net = KlnGrd_Net(
            in_dim=2,
            out_dim=1,
            n_hid_lyr=5,
            nodes_per_lyr=nPrLU
            ).to(device=dvc, dtype=torch.float32)
    else:
        ntwrk_prd = (2*torch.pi) / (bounds[0,1] - bounds[0,0])
        net = PeriodicBC_Network(
            in_dim=2,
            out_dim=1 ,
            n_hid_lyr=5,
            nodes_per_lyr=nPrLU,
            period=ntwrk_prd,
            ).to(device=dvc, dtype=torch.float32)
        
    libr.fit(network=net, inpts=torch.rand(size=(2,2), device=dvc, dtype=torch.float32))
    lib_names = libr.get_library_names(net_out_func_names=['u'])

    init_lmbda = torch.zeros(size=(len(lib_names),1), device=dvc,)
    # lrnr = EqLearner1D(net=net,
    #         Lmbda=init_lmbda.requires_grad_(True),
    #         lib_func=libr,
    #         sprs_slvr=slvr,
    #         data_dict=d_dict,
    #         tmprl_ords=[tmp_order],
    #         col_pnts_smplr=col_smplr,
    #         N_col_pnts=nCpnts,
    #         ntwk_out_names=["u"],
    #         device=dvc,
    #         data_type=torch.float32
    # )
    lrnr = Special_Sampler_Lrnr(net=net,
            Lmbda=init_lmbda.requires_grad_(True),
            lib_func=libr,
            sprs_slvr=slvr,
            data_dict=d_dict,
            tmprl_ords=[tmp_order],
            col_pnts_smplr=col_smplr,
            col_bnds = bounds.T,
            N_col_pnts=50_000,
            ntwk_out_names=["u"],
            device=dvc,
            data_type=torch.float32
    )

    os.makedirs(name="MoreSamplingRunStuff", exist_ok=True)
    os.chdir('MoreSamplingRunStuff')

    run_time = 0
    strt = perf_counter()
    lrnr.AdamsOptimTraining(mode="pre", alpha=preAlpha, gamma=preGamma, min_epochs=min_epchs, max_epochs=max_epchs, 
                            lrn_rt=lrn_rt, lp_ord=1.0, outputFreq=200,
                            betas=(0.9, 0.999), threshold=False,)
    run_time += (perf_counter() - strt)

    lrnr.ContourLikeComparisonPlot(dataset_name=data_set, pts=pts, values=U, fig_title=' ', state="PreTrained", show_fig=False, save_fig=True, save_title=saveName+'PreTrained', dif=False,exact=False, learned=False)
    
    strt = perf_counter()
    lrnr.ADO_Training(iters=ADO_iters, optim_alpha_grwth_methd="poly", LBFGS=False, lbfgs_epochs=10, Early_Term=True, Save_File_Name=None, Early_RFE_Term=True, p=0.5)
    # lrnr.ADO_Training(iters=ADO_iters, optim_alpha_grwth_methd="poly", LBFGS=False, lbfgs_epochs=10, Early_Term=False, Save_File_Name=None, Early_RFE_Term=True, p=0.5, train_alphas=ADO_alphas/ADO_alphas)
    run_time += (perf_counter() - strt)

    strt = perf_counter()
    lrnr.AdamsOptimTraining(mode="post", alpha=pst_alpha, gamma=0.0, min_epochs=min_pst_epchs, max_epochs=max_pst_epchs, 
                            lrn_rt=pst_lrn_rt, lp_ord=1.0, outputFreq=200,
                            betas=(0.9, 0.999), threshold=False,)
    run_time += (perf_counter() - strt)

    res_dict['run_time'] =  run_time
    err, errs, RHS_eq, = OneDimSols(dataset=Dname, lib=lib_names, lrnd_sol=lrnr.lmbda.data.cpu().numpy())
    learned = lrnr.Learned_EQ(output=False, sup_zeros=True,)
    print("The learned equation(s) was ...\n" + learned)
    crctEQ = 'u_'+'t'*tmp_order +' = '  + RHS_eq
    fig_title = 'Learned EQ - '+learned+'\n Correct EQ - '+crctEQ
    lrnr.WriteResults(data_set_name=data_set, file_name=data_set+"MoreSampling", precision=5, true_eq=crctEQ, errors=[err, errs], act_func='Tanh()', **res_dict)
    lrnr.TrainingLossPlots(dataset_name=data_set, file_name=saveName, plot_title=fig_title,)
    lrnr.ContourLikeComparisonPlot(dataset_name=data_set, pts=pts, values=U, fig_title=fig_title, state="PostTrained", show_fig=False, save_fig=True, save_title=saveName+'PostTrained', dif=False, exact=False, learned=False)
    lrnr.Loss_Score_Complexity_Plot(dataset_name=data_set, save_dir_name=saveName, save_figs=True)
    # lrnr.AnimatedPlot(dataset_name=data_set, spdx=2.0, pts=pts, values=U, 
    #                      ani_title=saveName+'PostTrained', fig_title=fig_title, state='PostTrained')
    lrnr.FVU_Plot(dataset_name=data_set, file_name=saveName, save_fig=True)
    lrnr.Save_Model(data_set=data_set, fname=sv_mdl_name)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Testing Effects of Number of Batches in Learned EQ')
    parser.add_argument('-dset', type=int, choices=range(1,20), required=True, help='Int value that indicates the data set to use in the test')
    parser.add_argument('-ver', type=int, choices=range(1,5), required=True, help='Int value that indicates the data set to use in the test')
    parser.add_argument('-data_type', type=int,choices=range(2), required=True, help='Selection train/test-ing as if sensors(1) or just randomly (0)')
    parser.add_argument('-JobID', type=int, required=True,
                        help='Job ID from the slurm batch manager/scheduler. Used in the file name when saving plots and animations after trainng. ')
    parser.add_argument('-arrayID', type=int, required=True, 
                        help='Array ID from the slurm batch job. Used in the plot/animation save names.')
    parser.add_argument('-node', type=str,required=True, help='The name of the node that training was run on ')
    args = parser.parse_args()
    dtype = "SensorData" if args.data_type else "RandPointsData"
    print(f"dflag={args.dset}, jobId={args.JobID}, arrayId={args.arrayID}, node={args.node}")
    main(dflag=args.dset, ver=args.ver, data_slct_type=dtype, jobId=args.JobID, arrayId=args.arrayID, node=args.node)