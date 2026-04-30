import os
from typing import Union, List
from numpy import ndarray, asarray, logical_and, abs
from datetime import datetime
from torch import Tensor as torch_Tensor
from torch import save as torch_save
from torch import clone as torch_clone
from torch import load as torch_load
from torch.nn import Module

class StabilizedLoss():
    """
    Descriptive text for later...
    Probably needs work and is not the best way to check if the loss had "stabilized"
    """

    def __init__(self, min_iters:int, max_iters:int, iter_range:int, chck_frq:int=50):
        """
        Possibly Descriptive text for later
        """
        if not all([type(arg).__name__=='int' for arg in [min_iters,max_iters,iter_range, chck_frq]]):
            raise TypeError("All the input arguments need to be interger type variables. At least one was evaluated to something else ")
        if min_iters>=max_iters:
            raise ValueError("The min_inters argument value must be less than the max_iters argument value")
        self.min_iters = min_iters
        self.max_iters = max_iters
        self.iter_range = iter_range
        self.chck_frq = chck_frq
        self.chck_cool_down = 0

    def __call__(self, losses:Union[ndarray, List[float]], trn_iteration:int):
        """
        Descriptive text for later...
        """

        # if the min number of training iterations has not been done yet
        # the loss has not "stabilized" but if the maximum number of training
        # iterations has been met, then stop training altogether (no worry about)
        # stability 
        if trn_iteration+1<self.min_iters:
            return False
        
        if trn_iteration+1>=self.max_iters:
            return True
        
        if self.chck_cool_down>0:
            self.chck_cool_down -= 1
            return False

        # now check if the loss over the last iter_range has stabilized or not.
        # note that since the losses input might be a list of floats, convert it 
        # to an array so that we can do the elements wise comparison below to get
        # the count. 
        loss_vals = asarray(losses[-self.iter_range:])
        loss_mu = loss_vals.mean()
        loss_std = loss_vals.std()
        
        cnt = logical_and(loss_vals<=loss_mu + loss_std/2 , loss_vals>= loss_mu - loss_std/2).sum().item()
        self.chck_cool_down = self.chck_frq

        return 2*cnt>self.iter_range

class LossTracker():
    """
    Pytorch Learning Rate Reduce on Plateau Scheduler adapted for the
    for my own purposes of tracking the loss and ending training 
    routine if the loss is not improving in one of two ways instead 
    of reducing the learning rate.
    """

    def __init__(self, mode:str="increasing", chng_mode:str='rel',
                 patience:int=20, eps:float=1e-4, 
                 track_mode:str="network",
                 min_iters:int=0):
        """
        
        """
        if mode!="increasing" and mode!="decreasing":
            raise ValueError("The \'mode\' argument needs to be either \'increasing\' or \'decreasing\'. Nothing else")
        if type(patience).__name__!="int":
            raise TypeError("The \'patience\' argument needs to be a integer")
        if patience<0: raise ValueError(f"The \'patience\' value must be at least zero. Given patience={patience}")
        if type(eps).__name__!="float":
            raise TypeError("The \'eps\' argument needs to be a integer")
        if eps<=0: raise ValueError(f"The \'eps\' value must be at greater than zero. Given eps={eps}")
        if chng_mode!="rel" and chng_mode!="abs":
            raise ValueError("The \'chng_mode\' argument needs to be either \'rel\' or \'abs\'. Nothing else")
        if track_mode.lower()!="network" and track_mode.lower()!="lp":
            raise ValueError("The \'track_mode\' argument needs to be either \'network\' or \'lp\'. Nothing else")
        if not(min_iters is None or isinstance(min_iters, int)):
            raise TypeError(f"The \'min_iters\' argument needs to be an integer object or equal to None. Nothing else")
        
        self.mode = mode
        if mode=="increasing":
            # this the worse value that the loss value could be.
            # Need this at the start so that no matter what the 
            # loss is it well be "better" than this. 
            self.lp_best = float('-inf')
            self.data_best = float('-inf')
            self.eq_best = float('-inf')

        else:
            # this the worse value that the loss value could be.
            # Need this at the start so that no matter what the 
            # loss is it well be "better" than this. 
            self.lp_best = float('inf')
            self.data_best = float('inf')
            self.eq_best = float('inf')


        self.patience = patience
        self.eps = eps
        self.chng_mode = chng_mode
        self.tracking_mode = track_mode.lower()
        # self.ntwrk_best_dict = None
        self.best_coefs = None
        # self.last_epoch = 0
        self.num_bad_lp_epochs = 0
        self.num_bad_data_epochs = 0
        self.num_bad_eq_epochs = 0
        self.stablized_data = False
        self.min_iters = 0 if min_iters is None else abs(min_iters).item()
        self.cur_iter = 0
        dtime = datetime.now()
        os.makedirs(name="tmp_net_states", exist_ok=True,)
        self.wghts_bs_file = os.path.join("tmp_net_states", f"LossTrackerCreatedOnY{dtime.year}M{dtime.month}D{dtime.day}at{dtime.hour}Hr{dtime.minute}Min{dtime.second}Sec{dtime.microsecond}MicroS.tar")


    def __call__(self, data_loss:Union[float, torch_Tensor],
                 eq_loss:Union[float, torch_Tensor], 
                 lp_loss:Union[float, torch_Tensor],
                 net:Module, lib_coefs:torch_Tensor)->bool:

        self.cur_iter +=1 
        # If the min_iter check is done up here then the loss is never
        # even compared to the best and so in actuality the minimum 
        # number of epochs run is min_iters + 2*patience since the 
        # number of bad epochs never increases. 

        if self.tracking_mode == "lp":
            if self._is_better(lp_loss, self.lp_best):
                self.lp_best = lp_loss
                self._save_weights_bais(network=net, eq_coefs=lib_coefs)
                self.num_bad_lp_epochs = 0
            else:
                self.num_bad_lp_epochs += 1
            ##############################################################
            # if the min_iters check is done here, then network best states
            # continue to get saved throughout all the epochs even if the 
            # epoch is bad and thus could possibly restart the counter#
            if self.cur_iter>=self.min_iters:
                
                if self.num_bad_lp_epochs >= self.patience:
                    self._load_weights_bais(network=net, eq_coefs=lib_coefs)
                    os.remove(path=self.wghts_bs_file)
                    return True
                else:
                    return False
            else:
                return False

        else:
            #  I.E  self.tracking_mode=="network":
            self._check_network_losses(d_loss=data_loss, q_loss=eq_loss, ntwrk=net, coefs=lib_coefs)

            if not(self.num_bad_data_epochs < self.patience or self.stablized_data):
            # if self.num_bad_data_epochs >= self.patience and not self.stablized_data:
                self.stablized_data = True
            ##############################################################
            # if the min_iters check is done here, then network best states
            # continue to get saved throughout all the epochs even if the 
            # epoch is bad and thus could possibly restart the counter#
            if self.cur_iter>=self.min_iters:
                
                if self.num_bad_eq_epochs >= self.patience:
                    self._load_weights_bais(network=net, eq_coefs=lib_coefs)
                    os.remove(path=self.wghts_bs_file)
                    return True
                else:
                    return False
            else:
                return False

    def _is_better(self, a, best):  
        if self.mode == "decreasing" and self.chng_mode == "rel":
            rel_epsilon = 1.0 - self.eps
            return a < best * rel_epsilon

        elif self.mode == "decreasing" and self.chng_mode == "abs":
            return a < best - self.eps

        elif self.mode == "increasing" and self.chng_mode == "rel":
            rel_epsilon = self.eps + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'increasing' and chng_mode == 'abs':
            return a > best + self.eps

    def _check_network_losses(self, d_loss:Union[float, torch_Tensor],
                              q_loss:Union[float, torch_Tensor],
                              ntwrk, coefs):
        """
            RECALL - stop training when both data and eq loss have stopped decreasing/increasing.
            More specifically only see if the eq loss has stopped decreasing when the data
            loss has stablized (stopped decreasing)
        """
        if not self.stablized_data:
            # The data loss has not stablized
            if self._is_better(d_loss, self.data_best):
                self.data_best = d_loss
                self._save_weights_bais(network=ntwrk, eq_coefs=coefs)
                self.num_bad_data_epochs = 0
            else:
                self.num_bad_data_epochs += 1
        else: 
            if self._is_better(q_loss, self.eq_best):
                self.eq_best = q_loss
                self._save_weights_bais(network=ntwrk, eq_coefs=coefs)
                self.num_bad_eq_epochs = 0
            else:
                self.num_bad_eq_epochs += 1

    def _save_weights_bais(self, network:Module, eq_coefs:torch_Tensor):
        """
        
        """
        torch_save(obj=network.state_dict(), f=self.wghts_bs_file)
        self.best_coefs = torch_clone(eq_coefs.data.detach())
        return self
        # return None

    def _load_weights_bais(self, network:Module, eq_coefs:torch_Tensor):
        """
        
        """
        loaded_dict = torch_load(f=self.wghts_bs_file,)
        network.load_state_dict(state_dict=loaded_dict, strict=True)
        eq_coefs.data = self.best_coefs.data
        return self
        # return None


