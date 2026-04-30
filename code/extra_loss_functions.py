from torch import Tensor, pow, detach, full_like, abs, isinf, sum, zeros, where, empty_like
from torch.nn import Module
from torch import device, linalg
from numpy import  newaxis
from numpy import abs as np_abs
from numpy import isinf as np_isinf
from numpy import logical_not as np_logical_not


def Torch_Lp_Loss(Xis:Tensor, p:float):
    """ 
    
    """

    assert(p > 0 and p < 2)
    assert(Xis.ndim==2)

    # First, square the components of Xi. Also, make a double precision copy of
    # Xi that is detached from Xi's graph.
    delta = .0000001
    Xis_2 = pow(Xis, 2)
    Xis_Detach = detach(Xis);       

    # Now, define a weights array.
    W_ks = full_like(Xis_Detach, fill_value=1/delta, requires_grad=False);
    n_funs, n_trgts = W_ks.shape
    
    Abs_Xis = pow(abs(Xis_Detach), 2-p)
    # W_ks = np.where(delta_array<=Abs_Xis, 1.0/Abs_Xis, 1.0/delta_array)
    for i in range(n_funs):
        for j in range(n_trgts):
            # abs_xi_i = abs(Xis[i])

            if delta<=Abs_Xis[i,j]:
                W_ks[i,j] = 1/Abs_Xis[i,j]

            if isinf(1/W_ks[i,j]): 
                W_ks[i,j] = 0

    return sum(W_ks*Xis_2, dim=0)

def Network_Weights_Bias_L2_Loss(network:Module, dvc=device)->Tensor:
    """
    
    """
    raise NotImplementedError(f"Function is still a work in progress")

    loss = zeros(size=(1,), device=dvc)[0]

    for param in network.parameters():
        # loss += linalg.norm(param, ord=None)
        # loss += linalg.norm(param, ord=2)
        loss += linalg.norm(param, ord=None, dim=None)

    return loss


def Lp_Loss(Xis:Tensor, Mask:Tensor, p:float, row_wise:bool=False):
    """ 
    
    """

    assert(p > 0 and p < 2)

    # First, square the components of Xi. Also, make a double precision copy of
    # Xi that is detached from Xi's graph.
    delta : float = .0000001;
    Xis_2          = pow(Xis, 2)
    Xis_Detach     = detach(Xis);       # cannot detach a numpy array so just create a copy of it. 
    
    # Now, define a weights array.
    W               = empty_like(Xis_Detach);
    
    Abs_Xis = pow(np_abs(Xis), 2-p)
    # delta_array = np.full_like(a=Xis, fill_value=delta, dtype=Xis.dtype, )
    # W_ks = np.where(delta_array<=Abs_Xis, 1.0/Abs_Xis, 1.0/delta_array)
    W_ks = where(delta<=Abs_Xis, 1.0/Abs_Xis, 1.0/delta)
    W_ks[np_isinf(W_ks)] = 0.0
    if row_wise:
        act_mask = np_logical_not(Mask).astype(float)
        axs = 1
    else:
        act_mask = np_logical_not(Mask).astype(float)[:,newaxis]
        axs = 0
    W_ks = act_mask * W_ks

    return sum(W_ks*Xis_2, dim=axs, keepdims=False)



