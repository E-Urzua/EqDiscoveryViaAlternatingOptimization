import os
import abc
from typing import List, Tuple, Union
from secrets import randbits
import numpy as np
from numpy.random import default_rng, SeedSequence
import torch

def partitionIndices(size:int, Nprts:int, rng_seed:int=None):
    """
    Partion the indices 0, 1,...., size-1 into parts disjoint subsets.
    Returns a list of Nprts elements where each element in the list 
    is a numpy integer array of values between 0 and size-1 and is 
    one of the Nprts partitions of the indices/integers 0,1,...,size-1
    """
    if rng_seed==None:
        seed = SeedSequence().entropy
    else:
        seed = rng_seed
    partitons = list()
    rng = np.random.default_rng(seed)
    probs = np.ones((size,), dtype=float)
    Psizes = np.zeros((Nprts, ), dtype=int)
    Psizes[:] = size//Nprts
    Psizes[0:size - (size//Nprts)*Nprts] +=1
    for i in range(Nprts):
        probs = probs/probs.sum()
        ids = rng.choice(a=size, size=Psizes[i], replace=False, p=probs)
        probs[ids] = 0
        partitons.append(ids)
    
    return partitons

def KFold_indices(n_samples:int, N_flds:int, rng_seed:int=None)->Tuple[List[np.ndarray]]:
    """
    Function that splits n_samples in to N_flds for the purpose of K-folds
    cross validation. If n_samples is not divisible by N_flds, the extra 
    samples are evenly distributed amongs all the N_flds. Specifically if 

        n_samples = N_flds * q + r
    
    Then the the first r of the N_flds have q+1 samples and the remaining
    N_flds-r folds have q samples. Note that we assume that the samples
    are not split into classes (i.e we only assume one class). 

    Input arguments:

        * n_samples (int > 0) - The number of samples that are to be 
                split into the K-folds (N_flds). 

        * N_flds (int > 0) - The number of folds to split the samples
                into (i.e the K in K-fold cross validation)
        
        * rng_seed (int or None) - The rng seed that is used to create
                the folds. Seed is used as the input to the SeedSequence
                function with in the numpy.random library and then used
                as the seed for the numpy default_rng() generator. 

    Return type: Tuple of lists
        
        * train_indices - list where the i-th element in the list is a
                1D numpy integer array containing the indices for the 
                i-th folds training data that should be used to create
                the i-th cross val. model. 
        
        * test_indices - list where the i-th element in the list is a
                1D numpy integer array containing the indices for the 
                i-th folds testing data that should be used to test
                the i-th cross val. model. 
    """
    if not isinstance(n_samples, int):
        raise TypeError(f"'n_samples' input argument needs to be a positive integer.")
    
    if not isinstance(N_flds, int):
        raise TypeError(f"'N_flds' input argument needs to be a positive integer.")
    
    if n_samples<2 or N_flds<1 or n_samples<N_flds:
        raise ValueError(f"Input arguments 'n_samples' and 'N_flds' need to be positive integers and N_flds <= n_samples")
    
    if not isinstance(rng_seed, int) and rng_seed!=None:
        raise TypeError(f"'rng_seed' input argument needs to be an integer of None value")
    if rng_seed!=None:
        if rng_seed<0: raise ValueError(f"'rng_seed'needs to be positive") 
    
    seed = np.random.SeedSequence(rng_seed)
    # print(f"The rng seed used is {seed.entropy}")
    rng = np.random.default_rng(seed)
    train_indices, test_indices = [], []
    probs = np.ones((n_samples), )
    probs = probs/probs.sum()
    fld_szs = np.zeros((N_flds, ), dtype=int)
    fld_szs[:] = n_samples//N_flds
    if n_samples%N_flds!=0:
        xtras = n_samples - N_flds*(n_samples//N_flds)
        fld_szs[:xtras] += np.ones((xtras,), dtype=int)
    for i in range(N_flds):
        tst_ids = rng.choice(a=n_samples, size=fld_szs[i], replace=False, p=probs)
        trn_ids = np.setdiff1d(np.arange(n_samples), tst_ids,)
        train_indices.append(trn_ids)
        test_indices.append(tst_ids)
        probs[tst_ids] = 0
        if probs.sum()!=0: probs = probs/probs.sum()

    if N_flds==1:
        return test_indices, test_indices
    return train_indices, test_indices, 

def np_solve_svd(matrx, trg, alph:float):
    """
    Numpy version/implementation of the sklearn.linear_model._solve_svd
    function found here 
    (https://github.com/scikit-learn/scikit-learn/blob/c5497b7f7/sklearn/linear_model/_ridge.py#L285)
    Written so can double check the torch version seen below (solve_svd). 
    """
    if alph<0:
        raise ValueError(f"alpha needs to be non-negative!")
    if alph==0:
        return np.linalg.lstsq(matrx, trg)[0]
    alph = np.full(shape=(1,), fill_value=alph,)
    U, s, vh = np.linalg.svd(matrx, full_matrices=False)
    ids = s>1e-15 # remove all the really (eigen) values (if any)
    s_nnz = s[ids][:, None]
    UTy = U.T @ trg
    d = np.zeros((s.shape[0], alph.shape[0]), dtype=matrx.dtype,)
    d[ids] = s_nnz / (s_nnz**2 + alph)
    d_UT_y = d * UTy
    return vh.T @ d_UT_y

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

def coef_regress(mat:torch.Tensor, trgt:torch.Tensor, apha:float):
    rs = solve_svd(matrix=mat, trgs=trgt, alphas=apha)
    lsts = torch.abs(rs).argmin(dim=0)
    return rs, lsts

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

def lossed_based_weghted_ave(losses:torch.Tensor, lambds:torch.Tensor, softmax:bool=False):
    """
        Most weight in the weight sum most go to the lambda that has the smallest loss
        weither the weights are found using the softmax or not. In the case of the 
        softmax function, the want the losses with the smallest value to have exp()
        raised to the highest power. 
    """
    if softmax:
        ws = torch.exp(1/losses)/torch.sum(torch.exp(1/losses))
        # ws = torch.exp(1 - losses/losses.max())/torch.sum(torch.exp(1 - losses/losses.max()))
    else:
        # the small added value is so that the lambda with the largest loss still contributes to the weighted average
        # if lambds.nonzero(as_tuple=False).numel()==2:
        #     ws = torch.ones_like(losses)
        if losses.numel()==1:
            ws = torch.ones_like(losses)
        else: 
            l_max = losses.max() + 1e-8
            rs = 1 - losses/l_max
            r = rs.sum()
            ws = rs/r
    aved_lmbda =  ws @lambds
    return (ws, aved_lmbda)
    
def Select_Coef_Vect(lossses:torch.Tensor, vects:torch.Tensor, best:bool):
    """
    
    """
    m, n = vects.shape
    if best:
        idx =lossses.argmin()
        return vects[idx, :]
    # else we do the things with the most combinations. 
    # find which is the most common combination of the library terms across all the vects 
    B = torch.where(vects==0,0,1)
    counts = torch.zeros(size=(m,), device=vects.device, dtype=vects.dtype)
    for i in range(m):
        c = torch.all( B[i,:] == B[i:,:], dim=1).sum()
        if torch.any( torch.all( B[i,:] == B[:i,:], dim=1) ):
            continue
        else:
            counts[i] = c
    # Now get the the most common combination of the library functions. Note that it is
    # possible that we hav differing combination that both have the same max number of 
    # occurances. 
    ids = torch.nonzero(counts==counts.max())[0]
    if ids.shape[0]==1:
        lmbdas_ids = torch.all(B[ids,:]==B, dim=1)
        ws , ave_lmbd = lossed_based_weghted_ave(losses=lossses[lmbdas_ids], lambds=vects[lmbdas_ids,:])
        # batch_ids = torch.nonzero(lmbdas_ids==True)[ws.argmax()]
    else:
        # ids has multiple entries so multiple maxes. Now choose which one
        ave_lmbds = torch.empty((ids.shape[0], n), device=vects.device, dtype=vects.dtype)
        ws = torch.empty((ids.shape[0], m), device=vects.device, dtype=vects.dtype)
        batch_ids = torch.empty((ids.shape[0],), device=vects.device, dtype=torch.int)
        for k in range(ids.shape[0]):
            lmbdas_ids = torch.all(B[ids[k],:]==B, dim=1)
            ws[k], ave_lmbds[k] = lossed_based_weghted_ave(losses=lossses[:,lmbdas_ids], lambds=vects[lmbdas_ids,:])
            batch_ids[k] = torch.nonzero(lmbdas_ids==True)[ws[k].argmax()]
        ave_lmbd = ave_lmbds[0]
    return ave_lmbd

def _check_alphas(alphs):
    """
    
    """
    # raise NotImplementedError
    return alphs


class SparseRegressAlg():
    """
    
    """
    def __init__(self, 
                 ADO_iters:int, 
                 normalize:bool=True, 
                 init_guess:torch.Tensor=None, 
                 thrshld_val:float=0.0
                 ):
        
        if not isinstance(ADO_iters, int):
            raise TypeError(f"The ADO_iters argument needs to be an integers value")
        if ADO_iters<1:
            raise ValueError(f"the ADO_iters needs to be an integer greater than 0 (i.e >= 1)")
        if not isinstance(normalize, bool):
            raise TypeError(f"The normalize argument needs to be an boolean")
        if init_guess and (not isinstance(init_guess, torch.Tensor)):
            raise TypeError(f"The init_guess needs to be a torch.tensor object to just None")
        if not isinstance(thrshld_val, float):
            raise TypeError(f"The \'thrshld_val\' input argument needs to be a float object")
        
        
        self.ADO_iters = ADO_iters
        self.normalize = normalize
        self.init_guess = init_guess
        self.thrs_val = np.abs(thrshld_val).item()
        self.cmplted_ADO_iter = 0
        self.Kfolds = 1

    @abc.abstractmethod
    def solve(self, A, b):
        raise NotImplementedError

class SSR(SparseRegressAlg):
    """
    Sparse Optimization regression method that implements the Stepwise Sparse Regression method
    found in the paper "Sparse learning of stochastic dynamical equations." by Boninsegna, Lorenzo, 
    Feliks Nüske, and Cecilia Clementi (see The Journal of chemical physics 148.24 (2018): 241723. )

    This implementation unlike the implemenation found in the PySindy library 
    (https://pysindy.readthedocs.io/en/stable/index.html) determines the its sparse solution using
    K-fold cross validation as detailed in the paper. Additions made to the original implementation
    given by Boninsegna, et al is that the columns of the Matrix A can be normalized to have unit
    L2 norm, the maximum number of 0 entries that the sparse solution can have is determined by a 
    reduction percentage rather than a maximum iteration number, an additional criteria that can 
    be use to zero out a coefficient, the method can be used for multiple targets and that the 
    minimization problem is formulated as a ridge regresssion problem and not a least squares 
    problem. 
    
    So this class tries to minimize 

        ||Ax - b||_{2}^{2} + alpha*||x||_{2}^{2}

    instead of minimizing

        ||Ax - b||_{2}^{2}
    
    where the columns of A can be normalized to have unit L2 norm, the coefficients iteratively 
    eliminated can be done so according to an additional ('optimal') criteria and the b can have
    multiple columns.

    Lastly this class is meant to work within the the broader PINNs based stuff I have to learn Eqs 
    from data and so other stuff found in this class are here to work with that stuff/code.
    """

    def __init__(self,
                Kfolds:int=10,
                alpha:Union[float, torch.Tensor]=0.0,
                percent_redux:float=0.50,
                criteria:str="coefficient_value",
                ADO_iters:int=5,
                normalize:bool=True,
                thrshld_val:float=0.0
                ):
        super().__init__(ADO_iters, normalize, None, thrshld_val)
        if not isinstance(Kfolds, int):
            raise TypeError("Input argument 'Kfolds' needs to be an int type object")
        if Kfolds<0:
            raise ValueError("Input argument 'Kfolds' needs to be a postive integer")
        
        if not isinstance(alpha, (float, torch.Tensor)):
            raise TypeError("Input argument 'alpha' needs to be a float or torch.Tensor type object")
    
        if not isinstance(percent_redux, float):
            raise TypeError("Input argument 'percent_redux' needs to be a float type object")
        if percent_redux <= 0 or percent_redux>1.0:
            raise ValueError("max iteration must be > 0 but < 1.0")
        
        if not isinstance(criteria, str):
            raise TypeError("Input argument 'criteria' needs to be a str type object")
        if criteria != "coefficient_value" and criteria != "model_residual":
            raise ValueError(
            "The only implemented criteria for sparsifying models "
            " are coefficient_value (zeroing out the smallest coefficient)"
            " or model_residual (choosing the N-1 term model with)"
            " the smallest residual error."
            )
        
        self.Kfolds = Kfolds
        self.alpha = _check_alphas(alpha)
        self.percent_redux = percent_redux
        self.criteria = criteria

    def solve(self, A:torch.Tensor, b:torch.Tensor)->torch.Tensor:
        """
        
        """
        mags = torch.linalg.vector_norm(A, ord=2, dim=0) if self.normalize else torch.ones_like(A[1], )
        x = A / mags
        if b.ndim==1:
            y = torch.clone(b)[:, np.newaxis]
        else:
            y = torch.clone(b)
        n_samples, n = A.shape
        if self.cmplted_ADO_iter==self.ADO_iters:
            self.percent_redux=1.0

        max_iters = int(np.around(n*self.percent_redux))

        n_trgts = y.shape[-1]
        cs = torch.zeros( size=(self.Kfolds, max_iters+1, n, n_trgts),device=A.device, dtype=A.dtype)
        Rls = torch.empty( size=(self.Kfolds, max_iters+1, n_trgts),device=A.device, dtype=A.dtype)

        trn_splts, tst_splts = KFold_indices(n_samples=n_samples, N_flds=self.Kfolds,)
        for q in range(self.Kfolds):
            ids = [ [i for i in range(n)] for _ in range(n_trgts)]
            for i in range(0, max_iters+1):
                if i==n:
                    Rls[:,-1] = torch.sum(y[tst_splts[q]].pow(2), axis=0)
                    continue
                for j in range(n_trgts):
                    if self.criteria=="coefficient_value":
                        cp, idx = coef_regress(x[trn_splts[q]][:,ids[j]], y[trn_splts[q],j:j+1], apha=self.alpha[j])
                    else:
                        cp, idx = model_regress(x[trn_splts[q]][:,ids[j]], y[trn_splts[q],j:j+1], apha=self.alpha[j])
                    cs[q,i,ids[j], j:j+1] = cp
                    Rls[q, i, j] = torch.sum((x[tst_splts[q]][:,ids[j]]@cp - y[tst_splts[q], j:j+1])**2, )
                    ids[j].pop(idx)

        coefs_out = torch.empty((n,n_trgts), device=A.device, dtype=A.dtype) 
        # crs_val_scrs = torch.mean(Rls, dim=0)
        crs_val_scrs = torch.sqrt(torch.mean(Rls, dim=0))
        chsn_sprsityd_id = torch.argmax(crs_val_scrs[1:]/crs_val_scrs[:-1], dim=0)
        for i in range(n_trgts):
            chsn_fld_ids = Rls[:, chsn_sprsityd_id[i], :].argmin(dim=0)[i]
            coefs_out[:,i] = torch.clone(cs[chsn_fld_ids,chsn_sprsityd_id[i], :, i ])
            coefs_out[:,i] = cs[chsn_fld_ids,chsn_sprsityd_id[i], :, i ]

        slvd_coefs = coefs_out / mags[:, None]

        if self.thrs_val>0.0:
            slvd_coefs = torch.where(slvd_coefs.abs()<self.thrs_val, torch.tensor(0.0, device=A.device, dtype=A.dtype), slvd_coefs,)

        return slvd_coefs

class RFE(SparseRegressAlg):
    """
    
    """

    def __init__(self,
                alpha:Union[float, torch.Tensor]=0.0,
                normalize:bool=True,
                annealing_factor:Union[int, float]=2,
                criteria:str="coefficient_value",
                selection="individual",
                ADO_iters:int=5,
                thrshld_val:float=0.0
                ):
        super().__init__(ADO_iters, normalize, None, thrshld_val)
        if not isinstance(alpha, (float, torch.Tensor)):
            raise TypeError("Input argument 'alpha' needs to be a float or torch.Tensor type object")
    
        if not isinstance(annealing_factor, (int, float)):
            raise TypeError("Input argument 'percent_redux' needs to be a float type object")
        if annealing_factor <1:
            raise ValueError("annealing_factor must be >= 1.0")
        
        if not isinstance(criteria, str):
            raise TypeError("Input argument 'criteria' needs to be a str type object")
        if criteria != "coefficient_value" and criteria != "model_residual":
            raise ValueError(
            "The only implemented criteria for sparsifying models "
            " are coefficient_value (zeroing out the smallest coefficient)"
            " or model_residual (choosing the N-1 term model with)"
            " the smallest residual error."
            )
        if not isinstance(selection, str):
            raise TypeError("Input argument 'criteria' needs to be a str type object")
        if selection != "individual" and selection != "group":
            raise ValueError(
                "The only implemented selection method choosing which sparse "
                " model with multiple targets as the solution to the linear " 
                "system in terms of least squares are individual (normal RFE " 
                "applied to each target) or group (losses for each target are " \
                "combined). Highly suggested to used default of individual."
            )
        
        self.alpha = _check_alphas(alpha)
        self.annealing_factor = annealing_factor
        self.criteria = criteria
        self.selection = selection

    def solve(self, A:torch.Tensor, b:torch.Tensor)->torch.Tensor:
        """
        
        """
        mags = torch.linalg.vector_norm(A, ord=2, dim=0) if self.normalize else torch.ones_like(A[1], )
        x = A / mags
        if b.ndim==1:
            y = torch.clone(b)[:, np.newaxis]
        else:
            y = torch.clone(b)
        _, n = A.shape
        max_iters = n

        if self.cmplted_ADO_iter==self.ADO_iters:
            self.annealing_factor=1

        n_trgts = y.shape[-1]
        cs = torch.zeros( size=(max_iters+1, n, n_trgts), device=A.device, dtype=A.dtype)
        # Rls = torch.empty( size=(max_iters+1, n_trgts), device=A.device, dtype=A.dtype)
        ids = [ [i for i in range(n)] for _ in range(n_trgts)]
        for i in range(0, n+1):
            if i==n:
                # Rls[-1] = torch.sum(y.pow(2), axis=0)
                continue
            for j in range(n_trgts):
                if self.criteria=="coefficient_value":
                    cp, idx = coef_regress(x[:,ids[j]], y[:,j:j+1], apha=self.alpha[j])
                else:
                    cp, idx = model_regress(x[:,ids[j]], y[:,j:j+1], apha=self.alpha[j])
                cs[i,ids[j], j:j+1] = cp
                # Rls[i, j] = torch.sum((x[:,ids[j]]@cp - y[j:j+1])**2, )
                ids[j].pop(idx)
        coefs_out = torch.empty((n,n_trgts), device=A.device, dtype=A.dtype)
        Rls = (x @ cs - y).pow(2).sum(dim=1)
        if self.selection == "individual":
            chsn_sprsityd_id = torch.round(torch.argmax(Rls[1:] / Rls[:-1], dim=0) / self.annealing_factor,).to(int)
            for i in range(n_trgts):
                # coefs_out[:,i] = torch.clone(cs[chsn_sprsityd_id[i], :, i ])
                coefs_out[:,i] = cs[chsn_sprsityd_id[i], :, i ]
        else:
            slctd = int(torch.argmax(Rls.sum(dim=1)[1:] / Rls.sum(dim=1)[:-1]) / self.annealing_factor)
            coefs_out = cs[slctd]

        self.cs = cs
        # self.cs = cs.cpu().numpy()

        slvd_coefs = coefs_out / mags[:, None]

        if self.thrs_val>0.0:
            slvd_coefs = torch.where(slvd_coefs.abs()<self.thrs_val, torch.tensor(0.0, device=A.device, dtype=A.dtype), slvd_coefs,)

        return slvd_coefs

class Cross_Val_RFE(SparseRegressAlg):
    """
    
    """

    def __init__(self,
                Kfolds:int=10,
                alpha:Union[float, torch.Tensor]=0.0,
                normalize:bool=True,
                annealing_factor:Union[int, float]=2,
                criteria:str="coefficient_value",
                selection="individual",
                ADO_iters:int=5,
                thrshld_val:float=0.0
                ):
        super().__init__(ADO_iters, normalize, None, thrshld_val)
        if not isinstance(Kfolds, int):
            raise TypeError("Input argument 'Kfolds' needs to be an int type object")
        if Kfolds<0:
            raise ValueError("Input argument 'Kfolds' needs to be a postive integer")
        
        if not isinstance(alpha, (float, torch.Tensor)):
            raise TypeError("Input argument 'alpha' needs to be a float or torch.Tensor type object")
    
        if not isinstance(annealing_factor, (int, float)):
            raise TypeError("Input argument 'percent_redux' needs to be a float type object")
        if annealing_factor <1:
            raise ValueError("annealing_factor must be >= 1.0")
        
        if not isinstance(criteria, str):
            raise TypeError("Input argument 'criteria' needs to be a str type object")
        if criteria != "coefficient_value" and criteria != "model_residual":
            raise ValueError(
            "The only implemented criteria for sparsifying models "
            " are coefficient_value (zeroing out the smallest coefficient)"
            " or model_residual (choosing the N-1 term model with)"
            " the smallest residual error."
            )
        if not isinstance(selection, str):
            raise TypeError("Input argument 'criteria' needs to be a str type object")
        if selection != "individual" and selection != "group":
            raise ValueError(
                "The only implemented selection method choosing which sparse "
                " model with multiple targets as the solution to the linear " 
                "system in terms of least squares are individual (normal RFE " 
                "applied to each target) or group (losses for each target are " \
                "combined). Highly suggested to used default of individual."
            )

        
        self.Kfolds = Kfolds
        self.alpha = _check_alphas(alpha)
        self.annealing_factor = annealing_factor
        self.criteria = criteria
        self.selection = selection

    def solve(self, A:torch.Tensor, b:torch.Tensor)->torch.Tensor:
        """
        
        """
        mags = torch.linalg.vector_norm(A, ord=2, dim=0) if self.normalize else torch.ones_like(A[1], )
        x = A / mags
        if b.ndim==1:
            y = torch.clone(b)[:, np.newaxis]
        else:
            y = torch.clone(b)
        n_samples, n = A.shape
        max_iters = n

        if self.cmplted_ADO_iter==self.ADO_iters:
            self.annealing_factor=1

        n_trgts = y.shape[-1]

        trn_splts, tst_splts = KFold_indices(n_samples=n_samples, N_flds=self.Kfolds,)
        fold_coefs = torch.zeros( size=(self.Kfolds, n, n_trgts), device=A.device, dtype=A.dtype)
        fold_tst_ls = torch.zeros(size=(self.Kfolds, n_trgts), device=A.device, dtype=A.dtype) 
        # There is probably a quicker way of doing x[trn_splts[q]][:,ids[j]] like using np.ix_
        for q in range(self.Kfolds):
            cs = torch.zeros( size=(max_iters+1, n, n_trgts), device=A.device, dtype=A.dtype)
            # Rls = torch.empty( size=(max_iters+1, n_trgts), device=A.device, dtype=A.dtype)

            ids = [ [i for i in range(n)] for _ in range(n_trgts)]
            for i in range(0, n+1):
                if i==n:
                    # Rls[-1] = torch.sum(y.pow(2), axis=0)
                    continue
                for j in range(n_trgts):
                    if self.criteria=="coefficient_value":
                        cp, idx = coef_regress(x[trn_splts[q]][:,ids[j]], y[trn_splts[q],j:j+1], apha=self.alpha[j])
                    else:
                        cp, idx = model_regress(x[trn_splts[q]][:,ids[j]], y[trn_splts[q],j:j+1], apha=self.alpha[j])
                    cs[i,ids[j], j:j+1] = cp
                    # Rls[i, j] = torch.sum((x[trn_splts[q]][:,ids[j]]@cp - y[trn_splts[q],j:j+1])**2, )
                    ids[j].pop(idx)
            Rls = (x @ cs - y).pow(2).sum(dim=1)
            if self.selection == "individual":
                slctd = torch.argmax(Rls[1:] / Rls[:-1], dim=0)
                an_slctd = torch.round(slctd / self.annealing_factor,).to(int)
                for i in range(n_trgts):
                    # coefs_out[:,i] = torch.clone(cs[chsn_sprsityd_id[i], :, i ])
                    fold_coefs[q, :,i] = cs[an_slctd[i], :, i ]
                    fold_tst_ls[q, i] = torch.sum((x[tst_splts[q]] @ cs[an_slctd] - y[tst_splts[q],j:j+1])**2, )
            else:
                slctd = torch.argmax(Rls.sum(dim=1)[1:] / Rls.sum(dim=1)[:-1])
                an_slctd = int(slctd / self.annealing_factor)
                fold_coefs[q] = cs[an_slctd]
                fold_tst_ls[q] = torch.sum((x[tst_splts[q]] @ cs[an_slctd] - y[tst_splts[q]])**2, )

        coefs_out = torch.empty(size= (n, n_trgts), device=A.device, dtype=A.dtype)
        if self.selection == "individual":
            chsn_sprsityd_id = fold_tst_ls.argmin(dim=0)
            for i in range(n_trgts):
                coefs_out[:,i] = fold_coefs[chsn_sprsityd_id[i], :, i]
        else:
            slctd = fold_tst_ls.sum(dim=1).argmin()
            coefs_out = fold_coefs[slctd]

        slvd_coefs = coefs_out / mags[:, None]

        if self.thrs_val>0.0:
            slvd_coefs = torch.where(slvd_coefs.abs()<self.thrs_val, torch.tensor(0.0, device=A.device, dtype=A.dtype), slvd_coefs,)

        return slvd_coefs

class Cross_Val_RFE_V2(SparseRegressAlg):
    """
    
    """

    def __init__(self,
                Kfolds:int=10,
                alpha:Union[float, torch.Tensor]=0.0,
                normalize:bool=True,
                annealing_factor:Union[int, float]=2,
                criteria:str="coefficient_value",
                selection="individual",
                best:bool=True,
                ADO_iters:int=5,
                thrshld_val:float=0.0
                ):
        super().__init__(ADO_iters, normalize, None, thrshld_val)
        if not isinstance(Kfolds, int):
            raise TypeError("Input argument 'Kfolds' needs to be an int type object")
        if Kfolds<0:
            raise ValueError("Input argument 'Kfolds' needs to be a postive integer")
        
        if not isinstance(alpha, (float, torch.Tensor)):
            raise TypeError("Input argument 'alpha' needs to be a float or torch.Tensor type object")
    
        if not isinstance(annealing_factor, (int, float)):
            raise TypeError("Input argument 'percent_redux' needs to be a float type object")
        if annealing_factor <1:
            raise ValueError("annealing_factor must be >= 1.0")
        
        if not isinstance(criteria, str):
            raise TypeError("Input argument 'criteria' needs to be a str type object")
        if criteria != "coefficient_value" and criteria != "model_residual":
            raise ValueError(
            "The only implemented criteria for sparsifying models "
            " are coefficient_value (zeroing out the smallest coefficient)"
            " or model_residual (choosing the N-1 term model with)"
            " the smallest residual error."
            )
        if not isinstance(selection, str):
            raise TypeError("Input argument 'criteria' needs to be a str type object")
        if selection != "individual" and selection != "group":
            raise ValueError(
                "The only implemented selection method choosing which sparse "
                " model with multiple targets as the solution to the linear " 
                "system in terms of least squares are individual (normal RFE " 
                "applied to each target) or group (losses for each target are " \
                "combined). Highly suggested to used default of individual."
            )
        
        if not isinstance(best, bool):
            raise TypeError(f"The coef_ave argument needs to be a bool object")

        
        self.Kfolds = Kfolds
        self.alpha = _check_alphas(alpha)
        self.annealing_factor = annealing_factor
        self.best_coef = best
        self.criteria = criteria
        self.selection = selection
        self._all_iter_coefs = []

    def solve(self, A:torch.Tensor, b:torch.Tensor)->torch.Tensor:
        """
        
        """
        mags = torch.linalg.vector_norm(A, ord=2, dim=0) if self.normalize else torch.ones_like(A[1], )
        x = A / mags
        if b.ndim==1:
            y = torch.clone(b)[:, np.newaxis]
        else:
            y = torch.clone(b)
        n_samples, n = A.shape
        max_iters = n

        if self.cmplted_ADO_iter==self.ADO_iters:
            self.annealing_factor=1
            self.best_coef=True

        n_trgts = y.shape[-1]

        trn_splts, tst_splts = KFold_indices(n_samples=n_samples, N_flds=self.Kfolds,)
        cs = torch.zeros( size=(self.Kfolds, max_iters+1, n, n_trgts), device=A.device, dtype=A.dtype)
        Rls = torch.empty( size=(self.Kfolds, max_iters+1, n_trgts), device=A.device, dtype=A.dtype)
        fold_coefs = torch.zeros( size=(self.Kfolds, n, n_trgts), device=A.device, dtype=A.dtype)
        fold_tst_ls = torch.zeros(size=(self.Kfolds, n_trgts), device=A.device, dtype=A.dtype) 
        # There is probably a quicker way of doing x[trn_splts[q]][:,ids[j]] like using np.ix_
        for q in range(self.Kfolds):
            cs = torch.zeros( size=(max_iters+1, n, n_trgts), device=A.device, dtype=A.dtype)
            # Rls = torch.empty( size=(max_iters+1, n_trgts), device=A.device, dtype=A.dtype)

            ids = [ [i for i in range(n)] for _ in range(n_trgts)]
            for i in range(0, n+1):
                if i==n:
                    # Rls[-1] = torch.sum(y.pow(2), axis=0)
                    continue
                for j in range(n_trgts):
                    if self.criteria=="coefficient_value":
                        cp, idx = coef_regress(x[trn_splts[q]][:,ids[j]], y[trn_splts[q],j:j+1], apha=self.alpha[j])
                    else:
                        cp, idx = model_regress(x[trn_splts[q]][:,ids[j]], y[trn_splts[q],j:j+1], apha=self.alpha[j])
                    cs[i,ids[j], j:j+1] = cp
                    # Rls[i, j] = torch.sum((x[trn_splts[q]][:,ids[j]]@cp - y[trn_splts[q],j:j+1])**2, )
                    ids[j].pop(idx)
            Rls = (x @ cs - y).pow(2).sum(dim=1)
            if self.selection == "individual":
                slctd = torch.argmax(Rls[1:] / Rls[:-1], dim=0)
                an_slctd = torch.round(slctd / self.annealing_factor,).to(int)
                for i in range(n_trgts):
                    # coefs_out[:,i] = torch.clone(cs[chsn_sprsityd_id[i], :, i ])
                    fold_coefs[q, :,i] = cs[an_slctd[i], :, i ]
                    fold_tst_ls[q, i] = torch.sum((x[tst_splts[q]] @ cs[an_slctd] - y[tst_splts[q],j:j+1])**2, )
            else:
                slctd = torch.argmax(Rls.sum(dim=1)[1:] / Rls.sum(dim=1)[:-1])
                an_slctd = int(slctd / self.annealing_factor)
                fold_coefs[q] = cs[an_slctd]
                fold_tst_ls[q] = torch.sum((x[tst_splts[q]] @ cs[an_slctd] - y[tst_splts[q]])**2, )

        coefs_out = torch.empty(size= (n, n_trgts), device=A.device, dtype=A.dtype)
        if self.selection == "individual":
            # chsn_sprsityd_id = fold_tst_ls.argmin(dim=0)
            # for i in range(n_trgts):
            #     coefs_out[:,i] = fold_coefs[chsn_sprsityd_id[i], :, i]
            for i in range(n_trgts):
                coefs_out[:,i] = Select_Coef_Vect(lossses=fold_tst_ls[:,i], vects=fold_coefs[:,:,i], best=self.best_coef)
        else:
            if self.best_coef:
                slctd = fold_tst_ls.sum(dim=1).argmin()
                coefs_out = fold_coefs[slctd]
            else:# need to figure this one out. Will leave as is for right now. 
                slctd = fold_tst_ls.sum(dim=1).argmin()
                coefs_out = fold_coefs[slctd]

        self._all_iter_coefs.append(fold_coefs.cpu().numpy())
        slvd_coefs = coefs_out / mags[:, None]

        if self.thrs_val>0.0:
            slvd_coefs = torch.where(slvd_coefs.abs()<self.thrs_val, torch.tensor(0.0, device=A.device, dtype=A.dtype), slvd_coefs,)

        return slvd_coefs

class STRidges(SparseRegressAlg):
    """
    
    """
    def __init__(self, 
                threshold:float=0.1,
                alpha:Union[float, torch.Tensor]=0.0,
                max_iter:int=20,
                ADO_iters:int=5,
                normalize:bool=True, 
                init_guess:torch.Tensor=None,):
        super().__init__(ADO_iters, normalize, init_guess)


    def solve(self, A, b):
        """
        
        """
        raise NotImplementedError

def RFESparseRegression(A:torch.Tensor, b:torch.Tensor, annealing_factor:Union[int, float]=2, best:bool=False)->torch.Tensor:
    """
    Adapted Recursive Feature Elimination (RFE) based on Stephany and Earl's idea of least important feature.
    For more see Their paper PDE-READ: Human-readable Partial Differential Equation Discovery using Deep Lrning.
    This method is a way to solve Lc = b where c should be sparse without having tunable (hyper-)parameters
    Input arguments/parameters are the following:
        A - 2D Tensor that contains the candidate library functions evaluations that we are
                trying to extra an PDE representation from. Note the (i,j)-th entry contains the 
                value of the j-th candidate functino evaluated at the i-th point
        b - 1D (column) tensor/vector that contains the values that are to be matched via a linear
                combination of the candidate library's functions
        annealing_factor - .... NOTE that this value has a lower bound; must be larger than the 
                           index of the choosen best vector divided by the number of columns in A
        best - boolean that determined whether or not to also return the best sparse solution in 
               addition to the one determined by the annealing factor. Note that if the annealing
               factor value is set to 1 and this is set to True, there will only be one return 
               argument as they would be the same
    """
    # NOTE: As normal do some better input parameter argument check more than what I currenly have (data - 18/08/22)
    if not isinstance(A, torch.Tensor):
        print('ERROR!!! - A input parameter needs to be a torch.Tensor but you gave {}'.format(type(A)))
        print('Killing the function and returning a value of None')
        return None
    if not isinstance(b, torch.Tensor):
        print('ERROR!!! - b input parameter needs to be a torch.Tensor but you gave {}'.format(type(b)))
        print('Killing the function and returning a value of None')
        return None
    if A.dim()!=2 or b.dim()==0 or b.dim()>2:
        print('ERROR!!! - One of the given tensors did not have the correct dimensions')
        print('A dimensions should be 2 and the others can be 1 or 2 dimensional')
        print('Killing the function and returning a value of None')
        return None
    if not isinstance(annealing_factor, (int, float)):
        print('ERROR!!! - annealing_factor input parameter needs to be an int or float but you gave {}'.format(type(annealing_factor)))
        print('Killing the function and returning a value of None')
        return None
    # now lets get to doing the RFE shit
    mags = torch.linalg.vector_norm(A, ord=2, dim=0)
    m = A.shape[1]
    # below are some tensor (arrays) to store the cs that come from removing a feature and the
    # storing of the RLS values for later.
    cs = torch.zeros(size=(m,m+1), dtype=A.dtype, device=A.device)
    Rls = torch.empty(size=(m+1,), dtype=A.dtype, device=A.device)
    ids = [i for i in range(m)]
    # now scale the A matrix
    Ap = A/mags
    for i in range(0, m):
        cp = torch.linalg.lstsq(Ap[:, ids], b)[0]                   # Get the new candidate solution with the new Lt matrix
        cs[ids,i:i+1] = cp
        # get the least important feature
        lst = cp.abs().argmin()
        ids.pop(lst)                                        # Remove this feature from being updates in the Cs that tracks the removal of feature
    # Ok so now cp and ids only have one element meaning that Lt and B have one feature and element
    # after the loop cp should be an all zeros vector
    Rls[:] = torch.linalg.vector_norm(Ap@cs - b, ord=2, dim=0,).pow(2)
    metrics = Rls[1:]/Rls[:-1]   
    # return cs[:,metrics.argmax():metrics.argmax()+1] / mags.reshape(-1,1)
    i = int(metrics.argmax())  # the "Best" sparse vector solution to Ax=b is stored in this column of the cs matrix
    if i/m>annealing_factor: # NOTE - Never enter this part of if block if annealing_factor>=1 as i/m must be less than 1 (i is 0,1,...m-1)
        print('WARNING - The given annealing factor value is too small well; Will return the last of the sparse vector solutions')
        annealing_factor = i/m + i/(2*m*(m-1))   # this small addition to i/m makes it so that k below equals m-1
    k = int(i/annealing_factor)  # k must inclusicely be between 0 and m-1 
    if (best and annealing_factor==1) or (not best and annealing_factor==1):
        return cs[:,i:i+1] / mags.reshape(-1,1)
    elif best and annealing_factor!=1:
        return cs[:,k:k+1] / mags.reshape(-1,1), cs[:,i:i+1] / mags.reshape(-1,1)
    elif not(best or annealing_factor==1):
        return cs[:,k:k+1] / mags.reshape(-1,1)

def RFEMatrixSparseRegression(A:torch.Tensor, B:torch.Tensor, annealing_factor:int=2)->torch.Tensor:
    """
    Adapted Recursive Feature Elimination (RFE) based on Stephany and Earl's idea of least important feature.
    For more see Their paper PDE-READ: Human-readable Partial Differential Equation Discovery using Deep Lrning.
    This method is a way to solve AC = B where C and B are both matrices with two columns and where C should be 
    sparse without having tunable (hyper-)parameters. The columns of C are solved for by just using the RFE
    Spares Regression method on each columns of B since it is the easiest thing to come up with at the moment.
    Input arguments/parameters are the following:
        A - 2D Tensor that contains the candidate library functions evaluations that we are
                trying to extra an PDE representation from. Note the (i,j)-th entry contains the 
                value of the j-th candidate functino evaluated at the i-th point
        B - 1D (column) tensor/vector that contains the values that are to be matched via a linear
                combination of the candidate library's functions
        annealing_factor - ....
    """
    # NOTE: As normal do some better input parameter argument check more than what I currenly have (data - 18/08/22)
    if not isinstance(A, torch.Tensor):
        print('ERROR!!! - A input parameter needs to be a torch.Tensor but you gave {}'.format(type(A)))
        print('Killing the function and returning a value of None')
        return None
    if not isinstance(B, torch.Tensor):
        print('ERROR!!! - b input parameter needs to be a torch.Tensor but you gave {}'.format(type(B)))
        print('Killing the function and returning a value of None')
        return None
    if A.dim()!=2 or B.dim()==0 or B.dim()>3:
        print('ERROR!!! - One of the given tensors did not have the correct dimensions')
        print('A dimensions should be 2 and the others can be 1 or 2 dimensional')
        print('Killing the function and returning a value of None')
        return None
    # now lets get to doing the RFE shit
    mags = torch.linalg.vector_norm(A, ord=2, dim=0)
    m = A.shape[1]
    p = B.shape[1]
    C = torch.empty(size=(m, p), dtype=B.dtype, device=B.device)
    # below are some tensor (arrays) to store the cs that come from removing a feature and the
    # storing of the RLS values for later.
    cs = torch.zeros(size=(m,m+1), dtype=A.dtype, device=A.device)
    Rls = torch.empty(size=(m+1,), dtype=A.dtype, device=A.device)
    # now scale the A matrix
    Ap = A/mags
    # Just do RFE on each fo the columns of B
    for j in range(p):
        ids = [i for i in range(m)]
        for i in range(0, m):
            cp = torch.linalg.lstsq(Ap[:, ids], B[:, j:j+1])[0]                   # Get the new candidate solution with the new Lt matrix
            cs[ids,i:i+1] = cp
            # get the least important feature
            lst = cp.abs().argmin()
            ids.pop(lst)                                        # Remove this feature from being updates in the Cs that tracks the removal of feature
        # Ok so now cp and ids only have one element meaning that Lt and B have one feature and element
        # after the loop cp should be an all zeros vector
        Rls[:] = torch.linalg.vector_norm(Ap@cs - B[:, j:j+1], ord=2, dim=0,).pow(2)
        metrics = Rls[1:]/Rls[:-1]
        C[:,j:j+1] = cs[:,int(metrics.argmax()/annealing_factor):int(metrics.argmax()/annealing_factor)+1] / mags.reshape(-1,1)
    # return cs[:,metrics.argmax():metrics.argmax()+1] / mags.reshape(-1,1)
    return C

def SSR_func(A:torch.Tensor, b:torch.Tensor, Kfolds:int, normalize:bool=True, alpha:Union[float, torch.Tensor]=0.0, percent_redux:float=0.50, criteria:str="coefficient_value")->Tuple[np.ndarray]:
    """
    The Pysindy Implementation of SSR is not entirely correct to how is is described in the paper by
    Boninsegna, Lorenzo, Feliks Nüske, and Cecilia Clementi ("Sparse learning of stochastic dynamical
    equations."). Yes they have added some features, normalizing the columns, functionality for Ridge
    Regression and a different selection criteria however one of the main feature of the original 
    description of SSR was the used of k-fold cross validation to determine the optimal level of 
    sparsity. So that has been enabled here. 

    """
    if not isinstance(A, torch.Tensor):
        print('ERROR!!! - features input parameter needs to be a np.ndarray but you gave {}'.format(type(A)))
        print('Killing the function and returning a value of None')
        raise TypeError("A input is expected to be a np.ndarray object type.")
    if not isinstance(b, torch.Tensor):
        print('ERROR!!! - b input parameter needs to be a np.ndarray but you gave {}'.format(type(b)))
        print('Killing the function and returning a value of None')
        raise TypeError("b input is expected to be a np.ndarray object type.")
    if A.ndim!=2 or b.ndim==0 or b.ndim>2:
        print('ERROR!!! - One of the given tensors did not have the correct dimensions')
        print('features dimensions shoule be 2 and the others can be 1 or 2 dimensional')
        print('Killing the function and returning a value of None')
        raise ValueError(" Either input arguement A or b has the wrong number of array dimensions. ")
    if Kfolds<=0:
        raise ValueError("Kfolds must be positive")
    
    if isinstance(alpha, float):
       alpha = torch.full(size=(b.shape[1],), fill_value=alpha, device=A.device)
    if torch.all(alpha<0):
        raise ValueError(f"alpha needs to be non-negative!")
    
    if percent_redux <= 0:
        raise ValueError("max iteration must be > 0 but < 1.0")

    if criteria != "coefficient_value" and criteria != "model_residual":
        raise ValueError(
            "The only implemented criteria for sparsifying models "
            " are coefficient_value (zeroing out the smallest coefficient)"
            " or model_residual (choosing the N-1 term model with)"
            " the smallest residual error."
        )
    mags = torch.linalg.vector_norm(A, ord=2, dim=0) if normalize else torch.ones_like((A.shape[1],), )
    x = A / mags
    if b.ndim==1:
        y = torch.clone(b)[:, np.newaxis]
    else:
        y = torch.clone(b)
    n_samples, n = A.shape
    max_iters = int(np.around(n*percent_redux))

    def coef_regress(mat:torch.Tensor, trgt:torch.Tensor, apha:float):
        rs = solve_svd(matrix=mat, trgs=trgt, alphas=apha)
        lsts = torch.abs(rs).argmin(dim=0)
        return rs, lsts
    
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
    
    n_trgts = y.shape[-1]
    cs = torch.zeros( size=(Kfolds, max_iters+1, n, n_trgts),device=A.device, dtype=A.dtype)
    Rls = torch.empty( size=(Kfolds, max_iters+1, n_trgts),device=A.device, dtype=A.dtype)

    trn_splts, tst_splts = KFold_indices(n_samples=n_samples, N_flds=Kfolds,)
    for q in range(Kfolds):
        ids = [ [i for i in range(n)] for _ in range(n_trgts)]
        for i in range(0, max_iters+1):
            if i==n:
                Rls[:,-1] = torch.sum(y[tst_splts[q]].pow(2), axis=0)
                continue
            for j in range(n_trgts):
                if criteria=="coefficient_value":
                    cp, idx = coef_regress(x[trn_splts[q]][:,ids[j]], y[trn_splts[q],j:j+1], apha=alpha[j])
                else:
                    cp, idx = model_regress(x[trn_splts[q]][:,ids[j]], y[trn_splts[q],j:j+1], apha=alpha[j])
                cs[q,i,ids[j], j:j+1] = cp
                Rls[q, i, j] = torch.sum((x[tst_splts[q]][:,ids[j]]@cp - y[tst_splts[q], j:j+1])**2, )
                ids[j].pop(idx)
    
    coefs_out = torch.empty((n,n_trgts), device=A.device, dtype=A.dtype) 
    crs_val_scrs = torch.mean(Rls, dim=0)
    crs_val_scrs = torch.sqrt(torch.mean(Rls, dim=0))
    chsn_sprsityd_id = torch.argmax(crs_val_scrs[1:]/crs_val_scrs[:-1], dim=0)
    for i in range(n_trgts):
        chsn_fld_ids = Rls[:, chsn_sprsityd_id[i], :].argmin(dim=0)[i]
        coefs_out[:,i] = torch.clone(cs[chsn_fld_ids,chsn_sprsityd_id[i], :, i ])
        coefs_out[:,i] = cs[chsn_fld_ids,chsn_sprsityd_id[i], :, i ]

    return coefs_out / mags[:,np.newaxis], cs, Rls

def Numpy_SSR(A:np.ndarray, b:np.ndarray, Kfolds:int, normalize:bool=True, alpha:float=0.0, max_iter:int=100, criteria:str="coefficient_value")->Tuple[np.ndarray]:
    """
    The Pysindy Implementation of SSR is not entirely correct to how is is described in the paper by
    Boninsegna, Lorenzo, Feliks Nüske, and Cecilia Clementi ("Sparse learning of stochastic dynamical
    equations."). Yes they have added some features, normalizing the columns, functionality for Ridge
    Regression and a different selection criteria however one of the main feature of the original 
    description of SSR was the used of k-fold cross validation to determine the optimal level of 
    sparsity. So that has been enabled here. 

    """
    if not isinstance(A, np.ndarray):
        print('ERROR!!! - features input parameter needs to be a np.ndarray but you gave {}'.format(type(A)))
        print('Killing the function and returning a value of None')
        raise TypeError("A input is expected to be a np.ndarray object type.")
    if not isinstance(b, np.ndarray):
        print('ERROR!!! - b input parameter needs to be a np.ndarray but you gave {}'.format(type(b)))
        print('Killing the function and returning a value of None')
        raise TypeError("b input is expected to be a np.ndarray object type.")
    if A.ndim!=2 or b.ndim==0 or b.ndim>2:
        print('ERROR!!! - One of the given tensors did not have the correct dimensions')
        print('features dimensions shoule be 2 and the others can be 1 or 2 dimensional')
        print('Killing the function and returning a value of None')
        raise ValueError(" Either input arguement A or b has the wrong number of array dimensions. ")
    if Kfolds<=0:
        raise ValueError("Kfolds must be positive")
    if alpha < 0:
        raise ValueError("alpha cannot be negative")

    if max_iter <= 0:
        raise ValueError("max iteration must be > 0")

    if criteria != "coefficient_value" and criteria != "model_residual":
        raise ValueError(
            "The only implemented criteria for sparsifying models "
            " are coefficient_value (zeroing out the smallest coefficient)"
            " or model_residual (choosing the N-1 term model with)"
            " the smallest residual error."
        )
    mags = np.linalg.vector_norm(A, ord=2, axis=0) if normalize else np.ones((A.shape[1],), )
    x = A / mags
    if b.ndim==1:
        y = np.copy(b)[:, np.newaxis]
    else:
        y = np.copy(b)
    n_samples, n = A.shape
    max_iters = n if max_iter>n else max_iter

    def coef_regress(mat, trgt):
        # l = mat.shape[1]
        if alpha==0:
            rs = np.linalg.lstsq(mat, trgt,)[0]
        else:
            rs = solve_svd(matrx=mat, trg=trgt, alph=alpha)
        lst = np.abs(rs).argmin()
        return rs, lst
    
    def model_regress(mat, trgt):
        l = mat.shape[1]
        xs = np.zeros( shape=(l,l), dtype=mat.dtype)
        for j in range(l):
            updts = [k for k in range(l) if k!=j]
            if alpha == 0:
                xs[updts, j:j+1] = np.linalg.lstsq(mat[:,updts], trgt)[0]
            else:
                # xs[updts, j:j+1] = np.linalg.lstsq(mat[:,updts].T @ mat[:,updts] + alpha*np.identity(l-1), mat[:,updts].T @ trgt)[0]
                xs[updts, j:j+1] = solve_svd(matrx=mat[:,updts], trg=trgt, alph=alpha)
        # get the least important feature - removed on the next pass
        lst = np.argmin(np.linalg.vector_norm(mat@xs - trgt, ord=2, axis=0),0)

        if alpha==0:
            rs = np.linalg.lstsq(mat, trgt,)[0]
        else:
            rs = solve_svd(matrx=mat, trg=trgt, alph=alpha)
        return rs, lst
    
    n_trgts = y.shape[-1]
    cs = np.zeros( shape=(Kfolds, max_iters+1, n, n_trgts), dtype=A.dtype)
    Rls = np.empty( shape=(Kfolds, max_iters+1, n_trgts), dtype=A.dtype)

    trn_splts, tst_splts = KFold_indices(n_samples=n_samples, N_flds=Kfolds,)
    for q in range(Kfolds):
        ids = [ [i for i in range(n)] for _ in range(n_trgts)]
        for i in range(0, max_iters+1):
            if i==n:
                Rls[:,-1] = np.sum(y[tst_splts[q]], axis=0)
                continue
            for j in range(n_trgts):
                if criteria=="coefficient_value":
                    cp, idx = coef_regress(x[trn_splts[q]][:,ids[j]], y[trn_splts[q],j:j+1])
                    cs[q,i,ids[j], j:j+1] = cp
                    Rls[q, i, j] = np.sum((x[tst_splts[q]][:,ids[j]]@cp - y[tst_splts[q], j:j+1])**2)
                    ids[j].pop(idx)
                else:
                    cp, idx = model_regress(x[trn_splts[q]][:,ids[j]], y[trn_splts[q],j:j+1])
                    cs[q,i,ids[j], j:j+1] = cp
                    Rls[q, i, j] = np.sum((x[tst_splts[q]][:,ids[j]]@cp - y[tst_splts[q], j:j+1])**2)
                    ids[j].pop(idx)

    coefs_out = np.empty((n,n_trgts)) 
    crs_val_scrs = np.mean(Rls, axis=0)
    crs_val_scrs = np.sqrt(np.mean(Rls, axis=0))
    chsn_sprsityd_id = np.argmax(crs_val_scrs[1:]/crs_val_scrs[:-1], axis=0)
    for i in range(n_trgts):
        chsn_fld_ids = Rls[:, chsn_sprsityd_id[i], :].argmin(axis=0)[i]
        coefs_out[:,i] = np.copy(cs[chsn_fld_ids,chsn_sprsityd_id[i], :, i ])

    return coefs_out / mags[:,np.newaxis], cs, Rls



