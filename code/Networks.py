"""
Python file that contains all the Neural Network Architectures to test/try
out for learing a PDE from data. This file is mostly a pytorch implementation
of the archs.py file found in the jaxpi for PIRATES Net which can be found
here https://github.com/PredictiveIntelligenceLab/jaxpi/tree/pirate
"""

from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict
import torch
from collections import OrderedDict
from copy import deepcopy



class Fourier_Embedding(torch.nn.Module):
    """
        Torch.nn.Module sub-class that maps input coordinates into high frequency 
        signals (a randonm Fourier feature embedding). It has been shown that 
        Multi-Layer Perceptrons (MLPs) suffer from a phenomona called spectral 
        bias (i.e that are biased towards learning low frequency functions) which
        prevents PINNs from learning high frequencies and fine structures of the 
        target solution. This feature embedding was proposed to to mitigate this
        bias and it is meant to be used at before the inputs are passed through 
        an MLP. So this should be used the first (or second - more on this later)
        part of a network that has a MLP (or just right before it). If you have 
        in input X which has N data points each of dimensionality M (i.e the 
        input X is an N by M tensor) then this encoding is defined by 
                Y = [cos(BX), sin(BX)]
        where B is an M by D array with entries randomly sampled from the 
        normal distrubutino with 0 mean and user choosen standard deviation
        greater than zero. So the ouput will be a N by 2D tensor. Note this 
        none of the frequencies are learnable/trainable. They are static 
        (registered buffer and not parameters) yet this class is a sub-class of
        the torch.nn.Module class so that if it is part of another Module 
        object, it matrix B will be moved with the model properly when sending
        it to a device. For more on Fourier Features to mitigate the problem 
        of spectral bias, see this paper by Tancik et al.
        https://www.semanticscholar.org/reader/a0dc3135c40e150f0271002a96b7c9680b6cac40
    """
    def __init__(self, scale:float, inpt_dim:int, embd_dim:int):
        """
        Input arguments for the class are as follows:

            scale - positive float input that is the stand. dev
                    of the normal distribution from which the 
                    entries in the B matrix are randomly sampled
                    from. Recommended that 1<=scale<=10.
            inpt_dim - The number of dimensions that any input 
                    to this model will have. Just the number of
                    columns that an input tensor to this module
                    will have. Needs to be a positive integer
            embd_dim - The dimensionality of the embedding. If
                    you want the embedding to have a dimension 
                    d, the randomly sampled B matrix will have
                    N rows and d//2 columns so that 
                        Y = [cos(BX), sin(BX)]
                    has d columns. Needs to be a positive integer. 
        """
        super().__init__()
        if not isinstance(scale, float):
            raise TypeError(f"The \'scale\' input argument was not a float type object")
        if  scale<=0.0:
            raise ValueError(f"The \'scale\' input argument must be positive valued.")
        if not isinstance(inpt_dim, int):
            raise TypeError(f"The \'inpt_dim\' input argument was not a int type object")
        if  inpt_dim<1:
            raise ValueError(f"The \'inpt_dim\' input argument must be at least 1.")
        if not isinstance(embd_dim, int):
            raise TypeError(f"The \'embd_dim\' input argument was not a int type object")
        if  embd_dim<2:
            raise ValueError(f"The \'embd_dim\' input argument must be at least 2.")

        kernel = torch.distributions.normal.Normal(loc=0.0, scale=scale).sample((inpt_dim, embd_dim//2))
        self.register_buffer(name='kernel', tensor=kernel)
        # self.register_parameter(name='kernel', 
        #             param= torch.nn.Parameter(data=kernel, requires_grad=False))
        self.inpt_dim = inpt_dim
        self.scale = scale
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        The forward operation of the module. How to pass any 
        given input tensor through the module. 
        """
        if self.inpt_dim != x.shape[-1]:
            raise ValueError(f"Input X argument does not have the correct input size")
        y = torch.cat((torch.cos(torch.matmul(x, self.kernel)),torch.sin(torch.matmul(x, self.kernel)) ), dim=1)
        return y
    
    def reinitialize(self):
        """
        
        """
        shp = self.kernel.shape
        self.kernel = torch.distributions.normal.Normal(loc=0.0, scale=self.scale).sample(shp)

        return self

class Periodic_Embedding(torch.nn.Module):
    """
        Torch.nn.Module sub-class to strictly impose periodic boundary conditions
        in PINNs as hard constraints. The main ideas is that if you have a function
        of one variable that has a period of P (ie. u(x) = u(x+P) ) we want a
        network architecture such that u^{l}(x) = u^{l}(x+P) for any layer l. To do
        this a special Fourier feature embedding of the form 
                    v(x) = ( cos(wx), sin(wx) )
        with w=2pi/P. For any network N it can be proved that N(v(x)) exactly 
        satisfies the periodic boundary conditions. The same can be done for high
        dimension 
            v(x,y) = [cos(w_{x}x) , sin(w_{x}x) , cos(w_{y}y) , sin(w_{y}y)]
        The same is also done in the case of time-dependent problem when the 
        solution/function is shown to be periodic in time. We also allow for the
        axis/coordinate to not be periodic. In this case they do not have an 
        embedding as shown. i.e v(x,y) = [x , cos(w_{y}y) , sin(w_{y}y)]
        We also allow for the periods to be learnable/trainable or not. It is 
        expected that this sub-class module will be the very first module that 
        an N by M input tensor of a network is passed through. Note also that if
        the input X is an N by M matrix then the output tensor of this module will
        have a size/shape N by M + T where 1<=T<=M is the number of trainable 
        periods (i.e the length of the tuple input arguments). Remember this when
        creating other networks that their input size should match this output
        size. For more on periodic bondary conditions imposed as as hard 
        constaints see the orignal paper by Wang et al....
        https://www.semanticscholar.org/reader/3134073bbcd4a261752f083371c17eb9e2489491
    """

    def __init__(self, period:Tuple[float], axis:Tuple[int], trainable:Tuple[bool]):
        """
            Input arguments are all expected to tbe tuples of the same length
                such that the i-th element of one tuple relates to the i-th
                tuple element of the other arguments. The are as follows:
                period    - The period of an axis/coordinte. Each element of
                    of the tuple is expected to be a float object. 
                axis      - The axis for which the period is to be applied. 
                    Expected/Needs to be an integer argument. If the i-th 
                    element of this tuple is 2 it means that the 2nd (ordering
                    starts from 0) axis/coordinate has a period of period[i].
                trainable - Tuple of booleans to indicate if a period for any
                    axis is trainable or not. So if the i-th entry is true/false
                    then the i-th entry in period tuple will be trainable or not
                    respectively. 
        """
        super().__init__()
        if not(len(period) == len(axis) == len(trainable)):
            raise ValueError(f"For some reaoson one of the input arguments has more or less entries than the other(s)")
        if not isinstance((period, axis, trainable), tuple):
            raise TypeError(f"One of the input arguments was not a tuple object")
        items = [isinstance(el, float) for el in period]
        item_vals =  [el==0.0 for el in period]
        if not all(items):
            raise TypeError(f"One of the entries in the \'period\' input argument tuple was not a float object")
        if any(item_vals):
            raise ValueError(f"One of the entries in the \'period\' input argument tuple was equal to 0")
        items = [isinstance(el, int) for el in axis]
        item_vals =  [el<0 for el in axis]
        if not all(items):
            raise TypeError(f"One of the entries in the \'axis\' input argument tuple was not an int object")
        if any(item_vals):
            raise ValueError(f"One of the entries in the \'axis\' input argument less than or equal to 0")
        items = [isinstance(el, bool) for el in trainable]
        if not all(items):
            raise TypeError(f"One of the entries in the \'trainable\' input argument tuple was not a boolean object")

        self.period = period
        self.axis = axis
        self.trainable = trainable
        # the period is learnable/trainable register a parameter
        # if it is not then register it as a buffer. Done this 
        # way so that the .parameters() method returns only the 
        # learnable parameters to the optimizer. Also registered
        # as buffer because they basically are parameters in 
        # the model, which should be saved and restored in the
        # state_dict, but not trained by the optimizer. Also 
        # all buffers and parameters will be pushed to the 
        # device, if called on the parent model:
        # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
        for idx, trnbl in enumerate(trainable):
            name = f"period{idx}"
            if trnbl:
                self.register_parameter(name=name, 
                                        param=torch.nn.Parameter(data=period[idx]*torch.ones((1,)), requires_grad=True))
            else:
                # buf = period[idx]*torch.ones((1,),)
                buf = torch.fill(torch.empty((1,),), period[idx])
                self.register_buffer(name=name, 
                                        tensor=buf,)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        The forward operation of the module. How to pass any 
        given input tensor through the module. The code is 
        sloppy since I was not able to find a neat/clean way
        to access the period for each axis/dimension when
        some are are learnable/trainable (model parameters)
        and some are not (module buffers)
        """
        num_c = x.shape[-1]
        y = []
        # each column of the x tensor is an axis of the data, so going
        # through the columns is the same as going through the axes of
        # the data. 
        for i in range(num_c):
            if i in self.axis:
                idx = self.axis.index(i)
                # prd = self.prd_vals[f"period{idx}"]
                if self.trainable[idx]:
                    prd = self.__dict__['_parameters'][f"period{idx}"]
                else:
                    prd = self.__dict__['_buffers'][f"period{idx}"]
                y.extend([torch.cos(prd*x[:,i:i+1]), torch.sin(prd*x[:,i:i+1])])
            else:
                y.append(x[:,i:i+1])

        return torch.hstack(y)

    def reinitialize(self):
        """
        
        """

        return self

class Periodic_EmbeddingVer2(torch.nn.Module):
    """
        Torch.nn.Module sub-class to strictly impose periodic boundary conditions
        in PINNs as hard constraints. The main ideas is that if you have a function
        of one variable that has a period of P (ie. u(x) = u(x+P) ) we want a
        network architecture such that u^{l}(x) = u^{l}(x+P) for any layer l. To do
        this a special Fourier feature embedding of the form 
                    v(x) = ( cos(wx), sin(wx) )
        with w=2pi/P. For any network N it can be proved that N(v(x)) exactly 
        satisfies the periodic boundary conditions. The same can be done for high
        dimension 
            v(x,y) = [cos(w_{x}x) , sin(w_{x}x) , cos(w_{y}y) , sin(w_{y}y)]
        The same is also done in the case of time-dependent problem when the 
        solution/function is shown to be periodic in time. We also allow for the
        axis/coordinate to not be periodic. In this case they do not have an 
        embedding as shown. i.e v(x,y) = [x , cos(w_{y}y) , sin(w_{y}y)]
        We also allow for the periods to be learnable/trainable or not. It is 
        expected that this sub-class module will be the very first module that 
        an N by M input tensor of a network is passed through. Note also that if
        the input X is an N by M matrix then the output tensor of this module will
        have a size/shape N by M + T where 1<=T<=M is the number of trainable 
        periods (i.e the length of the tuple input arguments). Remember this when
        creating other networks that their input size should match this output
        size. For more on periodic bondary conditions imposed as as hard 
        constaints see the orignal paper by Wang et al....
        https://www.semanticscholar.org/reader/3134073bbcd4a261752f083371c17eb9e2489491
    """

    def __init__(self, period:Tuple[float], Ns:Tuple[int], axis:Tuple[int], trainable:Tuple[bool]):
        """
            Input arguments are all expected to tbe tuples of the same length
                such that the i-th element of one tuple relates to the i-th
                tuple element of the other arguments. The are as follows:
                period    - The period of an axis/coordinte. Each element of
                    of the tuple is expected to be a float object. 
                axis      - The axis for which the period is to be applied. 
                    Expected/Needs to be an integer argument. If the i-th 
                    element of this tuple is 2 it means that the 2nd (ordering
                    starts from 0) axis/coordinate has a period of period[i].
                trainable - Tuple of booleans to indicate if a period for any
                    axis is trainable or not. So if the i-th entry is true/false
                    then the i-th entry in period tuple will be trainable or not
                    respectively. 
        """
        super().__init__()
        if not(len(period) == len(Ns) == len(axis) == len(trainable)):
            raise ValueError(f"For some reaoson one of the input arguments has more or less entries than the other(s)")
        if not isinstance((period, axis, trainable), tuple):
            raise TypeError(f"One of the input arguments was not a tuple object")
        items = [isinstance(el, float) for el in period]
        item_vals =  [el==0.0 for el in period]
        if not all(items):
            raise TypeError(f"One of the entries in the \'period\' input argument tuple was not a float object")
        if any(item_vals):
            raise ValueError(f"One of the entries in the \'period\' input argument tuple was equal to 0")
        items = [isinstance(el, int) for el in Ns]
        item_vals =  [el==0.0 for el in Ns]
        if not all(items):
            raise TypeError(f"One of the entries in the \'Ns\' input argument tuple was not a int object")
        if any(item_vals):
            raise ValueError(f"One of the entries in the \'Ns\' input argument tuple was equal to 0")
        items = [isinstance(el, int) for el in axis]
        item_vals =  [el<0 for el in axis]
        if not all(items):
            raise TypeError(f"One of the entries in the \'axis\' input argument tuple was not an int object")
        if any(item_vals):
            raise ValueError(f"One of the entries in the \'axis\' input argument less than or equal to 0")
        items = [isinstance(el, bool) for el in trainable]
        if not all(items):
            raise TypeError(f"One of the entries in the \'trainable\' input argument tuple was not a boolean object")

        self.period = period
        self.Ns = Ns
        self.axis = axis
        self.trainable = trainable
        # the period is learnable/trainable register a parameter
        # if it is not then register it as a buffer. Done this 
        # way so that the .parameters() method returns only the 
        # learnable parameters to the optimizer. Also registered
        # as buffer because they basically are parameters in 
        # the model, which should be saved and restored in the
        # state_dict, but not trained by the optimizer. Also 
        # all buffers and parameters will be pushed to the 
        # device, if called on the parent model:
        # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
        for idx, trnbl in enumerate(trainable):
            name = f"period{idx}"
            if trnbl:
                self.register_parameter(name=name, 
                                        param=torch.nn.Parameter(data=period[idx]*torch.ones((1,)), requires_grad=True))
            else:
                # buf = period[idx]*torch.ones((1,),)
                buf = torch.fill(torch.empty((1,),), period[idx])
                self.register_buffer(name=name, 
                                        tensor=buf,)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        The forward operation of the module. How to pass any 
        given input tensor through the module. The code is 
        sloppy since I was not able to find a neat/clean way
        to access the period for each axis/dimension when
        some are are learnable/trainable (model parameters)
        and some are not (module buffers)
        """
        num_c = x.shape[-1]
        dvc = x.device
        y = []
        # each column of the x tensor is an axis of the data, so going
        # through the columns is the same as going through the axes of
        # the data. 
        for i in range(num_c):
            if i in self.axis:
                idx = self.axis.index(i)
                # prd = self.prd_vals[f"period{idx}"]+
                m = self.Ns[idx]
                if self.trainable[idx]:
                    prd = self.__dict__['_parameters'][f"period{idx}"]
                else:
                    prd = self.__dict__['_buffers'][f"period{idx}"]
                y.extend([torch.cos(torch.linspace(1,m,m).to(device=dvc)*prd*x[:,i:i+1]), torch.sin(torch.linspace(1,m,m).to(device=dvc)*prd*x[:,i:i+1])])
            else:
                y.append(x[:,i:i+1])

        if len(self.period)>=1:
            y.append(torch.ones_like(x[:,i:i+1]))

        return torch.hstack(y)
    
    def reinitialize(self):
        """
        
        """

        return self

class Embeddings(torch.nn.Module):
    """
    IF NEED - descripe this class. At the moment it is not used at all
    so just leave it be. 
    """
    def __init__(self, prd_stuff:Union[None, Dict], four_stuff:Union[None, Dict]):
        super().__init__()
        self.embds = torch.nn.Sequential()
        if prd_stuff:
            self.embds.append(Periodic_Embedding(**prd_stuff))
        if four_stuff:
            self.embds.append(Fourier_Embedding(**four_stuff))
        if len(self.embds)==0:
            self.embds.append(torch.nn.Identity())
        
    def forward(self, x):
        """
        The forward operation of the module. How to pass any 
        given input tensor through the module. 
        """
        return self.embds(x)

# make one of the dense layers a subclass of the other. 
class DenseLayerV1(torch.nn.Module):
    """
        A torch.nn.Module sub-class which is effectively a special version of the
        torch.nn.Linear class as it has the same functionality as that class but
        with the additional ability to use Random Weight Factorization (RWF) to
        improve the performance of PINNs. RWF was originally proposed by Wang et
        al. showing it to consistently improve the perforance of PINNs. Basically
        RWF just factorizes the initialized weight matrices W of an MLP into 
        two separate parameters s and V and training is instead down on these
        parameters in stead of W. Generally the parameters of an MLP are 
        initialized acoording to some scheme (e.g Xavier/Glorot normal scheme). 
        After, for every weight matrices W, a "scale vector" exp(s) is 
        initialized where s is sampled from a multivariate normal distribution 
        with mean u and standard deviation std ( i.e N(u, std I) ). This scale 
        vector is then used to factorize W find a matric V such that
                W = diag(exp(s)) * V
        Once s and V are had, gradient descent is applied directly to s and V.
        Of course RWF does not have to be used with this class and regardlesss
        of using RFW or not, the initialization of the weight matrices W can
        be any user given function does in place operations to a tensor. 
        See the paper by Wang et. al here...
        https://www.semanticscholar.org/reader/38487547d70f7b271dcec63ee684b66a5782719e

    """
    def __init__(self, in_dim:int,
                out_dim:int,
                weight_init_func:Callable, # should be a torch.nn.init. function that does in place operations on a tensor
                bias:bool,
                bias_init_func:Callable,  # should be a torch.nn.init. function that does in place operations on a tensor
                rnd_wght_fact:bool=True,
                **kwargs):
        """
            Input arguments are as follows:
                in_dim - The dimension for any put tensor that will be passed through 
                    this module. Just the number of columns on any input tensor will have
                out_dim - The out put dimension. The tensor that is output from any input
                    tensor will have this many columns 
                weight_init_func - tensor function that does in-place operations on a
                    tensor. Used to initialize the weights of the weight matrix W
                bias - Boolean argument indicating whether or not to incluse a bias
                    term
                bias_init_func -  tensor function that does in-place operations on a 
                    tensor. Used to initialize the weights of the bais tensor b
                rnd_wght_fact - Boolean argument that indicates whether or not to do/use
                    random weight factorization. 
                kwargs - additional key word argument to use. Only used if the 
                    rnd_wght_fact input value is True and if so the expected keywords 
                    are 'mean' and 'std' and should be float object. These values are
                    used as the mean and the standard deviation of the multivariate 
                    normal distribution function from which s is sampled from in RWF
                    method. "Empirially so small of values may lead to performance that
                    is similar to a conventional MLP and too large of values are result 
                    in unstable training". Recommended values are that mean = 0.5 or 1 
                    and std = 0.1
        """
        super().__init__()
        # don't exactly know how to check for the init_funcs
        if not isinstance(in_dim, int):
            raise TypeError(f"The \'in_dim\' input needs to be an integer type object")
        if in_dim<1:
            raise ValueError(f"The \'in_dim\' input needs to be at least 1. ")
        if not isinstance(out_dim, int):
            raise TypeError(f"The \'out_dim\' input needs to be an integer type object")
        if out_dim<1:
            raise ValueError(f"The \'out_dim\' input needs to be at least 1. ")
        if not isinstance(weight_init_func, Callable):
            raise TypeError(f"The \'weight_init_func\' input argument was not a callable object")
        if not isinstance(bias, int):
            raise TypeError(f"The \'bias\' input needs to be a boolean type object")
        if not isinstance(bias_init_func, Callable):
            raise TypeError(f"The \'bias_init_func\' input argument was not a callable object")
        if not isinstance(rnd_wght_fact, int):
            raise TypeError(f"The \'rnd_wght_fact\' input needs to be a boolean type object")
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.incld_bias = bias
        self.weight_init_func = weight_init_func
        self.rnd_wght_fact = rnd_wght_fact
        self.bias_init_func = bias_init_func
        if rnd_wght_fact:
            self.fct_mean = kwargs['mean']
            self.fct_std = kwargs['std']
            W = weight_init_func(torch.empty((out_dim, in_dim), ))
            s = torch.distributions.normal.Normal(loc=self.fct_mean, scale=self.fct_std).sample((out_dim, ))
            V = torch.linalg.lstsq(torch.diag(torch.exp(s)), W)[0]
            self.register_parameter(name='s',
                    param=torch.nn.Parameter(data=s, requires_grad=True))
            self.register_parameter(name='V',
                    param=torch.nn.Parameter(data=V, requires_grad=True))

        else:
            self.register_parameter(name='weight', 
                    param=torch.nn.Parameter(data=weight_init_func(torch.empty((out_dim, in_dim), )), requires_grad=True))
            
        if bias:
            self.register_parameter(name='bias',
                    param=torch.nn.Parameter(data=bias_init_func(torch.empty((out_dim, ), )), requires_grad=True))
        else:
            self.register_buffer(name='bias', tensor=torch.zeros((out_dim, ), ), persistent=True)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        The forward operation of the module. How to pass any 
        given input tensor through the module. 
        """
        if self.rnd_wght_fact:
            W = torch.diag(torch.exp(self.s)) @ self.V
            y = x @ W.T
        else:
            y = x @ self.weight.T

        return y + self.bias

    def reinitialize(self):
        """
        
        """
        if self.rnd_wght_fact:
            W = self.weight_init_func(torch.empty((self.out_dim, self.in_dim), ))
            s = torch.distributions.normal.Normal(loc=self.fct_mean, scale=self.fct_std).sample((self.out_dim, ))
            V = torch.linalg.lstsq(torch.diag(torch.exp(s)), W)[0]
            self.s.data = s
            self.V.data = V

        else:
            self.weight.data = self.weight_init_func(torch.empty((self.out_dim, self.in_dim), ))
            
        if self.incld_bias:
            self.bias.data = self.bias_init_func(torch.empty((self.out_dim, ), ))

        return self

class DenseLayer(torch.nn.Module):
    """
        A torch.nn.Module sub-class which is effectively a special version of the
        torch.nn.Linear class as it has the same functionality as that class but
        with the additional ability to use Random Weight Factorization (RWF) to
        improve the performance of PINNs. RWF was originally proposed by Wang et
        al. showing it to consistently improve the perforance of PINNs. Basically
        RWF just factorizes the initialized weight matrices W of an MLP into 
        two separate parameters s and V and training is instead down on these
        parameters in stead of W. Generally the parameters of an MLP are 
        initialized acoording to some scheme (e.g Xavier/Glorot normal scheme). 
        After, for every weight matrices W, a "scale vector" exp(s) is 
        initialized where s is sampled from a multivariate normal distribution 
        with mean u and standard deviation std ( i.e N(u, std I) ). This scale 
        vector is then used to factorize W find a matric V such that
                W = diag(exp(s)) * V
        Once s and V are had, gradient descent is applied directly to s and V.
        Of course RWF does not have to be used with this class and regardlesss
        of using RFW or not, the initialization of the weight matrices W can
        be any user given function does in place operations to a tensor. 
        See the paper by Wang et. al here...
        https://www.semanticscholar.org/reader/38487547d70f7b271dcec63ee684b66a5782719e

    """
    def __init__(self, in_dim:int,
                out_dim:int,
                weight_init_func:Callable, # should be a torch.nn.init. function that does in place operations on a tensor
                bias:bool,
                bias_init_func:Callable,  # should be a torch.nn.init. function that does in place operations on a tensor
                rnd_wght_fact:bool=True,
                **kwargs):
        """
            Input arguments are as follows:
                in_dim - The dimension for any put tensor that will be passed through 
                    this module. Just the number of columns on any input tensor will have
                out_dim - The out put dimension. The tensor that is output from any input
                    tensor will have this many columns 
                weight_init_func - tensor function that does in-place operations on a
                    tensor. Used to initialize the weights of the weight matrix W
                bias - Boolean argument indicating whether or not to incluse a bias
                    term
                bias_init_func -  tensor function that does in-place operations on a 
                    tensor. Used to initialize the weights of the bais tensor b
                rnd_wght_fact - Boolean argument that indicates whether or not to do/use
                    random weight factorization. 
                kwargs - additional key word argument to use. Only used if the 
                    rnd_wght_fact input value is True and if so the expected keywords 
                    are 'mean' and 'std' and should be float object. These values are
                    used as the mean and the standard deviation of the multivariate 
                    normal distribution function from which s is sampled from in RWF
                    method. "Empirially so small of values may lead to performance that
                    is similar to a conventional MLP and too large of values are result 
                    in unstable training". Recommended values are that mean = 0.5 or 1 
                    and std = 0.1
        """
        super().__init__()
        # don't exactly know how to check for the init_funcs
        if not isinstance(in_dim, int):
            raise TypeError(f"The \'in_dim\' input needs to be an integer type object")
        if in_dim<1:
            raise ValueError(f"The \'in_dim\' input needs to be at least 1. ")
        if not isinstance(out_dim, int):
            raise TypeError(f"The \'out_dim\' input needs to be an integer type object")
        if out_dim<1:
            raise ValueError(f"The \'out_dim\' input needs to be at least 1. ")
        if not isinstance(weight_init_func, Callable):
            raise TypeError(f"The \'weight_init_func\' input argument was not a callable object")
        if not isinstance(bias, int):
            raise TypeError(f"The \'bias\' input needs to be a boolean type object")
        if not isinstance(bias_init_func, Callable):
            raise TypeError(f"The \'bias_init_func\' input argument was not a callable object")
        if not isinstance(rnd_wght_fact, int):
            raise TypeError(f"The \'rnd_wght_fact\' input needs to be a boolean type object")
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.incld_bias = bias
        self.weight_init_func = weight_init_func
        self.rnd_wght_fact = rnd_wght_fact
        self.bias_init_func = bias_init_func
        if rnd_wght_fact:
            self.fct_mean = kwargs['mean']
            self.fct_std = kwargs['std']
            W = weight_init_func(torch.empty((out_dim, in_dim), ))
            s = torch.distributions.normal.Normal(loc=self.fct_mean, scale=self.fct_std).sample((out_dim, ))
            V = torch.linalg.lstsq(torch.diag(torch.exp(s)), W)[0]
            self.register_parameter(name='s',
                    param=torch.nn.Parameter(data=s, requires_grad=True))
            self.register_parameter(name='V',
                    param=torch.nn.Parameter(data=V, requires_grad=True))

        else:
            self.register_parameter(name='weight', 
                    param=torch.nn.Parameter(data=weight_init_func(torch.empty((out_dim, in_dim), )), requires_grad=True))
            
        if bias:
            self.register_parameter(name='bias',
                    param=torch.nn.Parameter(data=bias_init_func(torch.empty((out_dim, ), )), requires_grad=True))
        else:
            self.register_buffer(name='bias', tensor=torch.zeros((out_dim, ), ), persistent=True)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        The forward operation of the module. How to pass any 
        given input tensor through the module. 
        """
        if self.rnd_wght_fact:
            # W = torch.diag(torch.exp(self.s)) @ self.V
            W = torch.diag(self.s) @ self.V
            y = x @ W.T
        else:
            y = x @ self.weight.T

        return y + self.bias

    def reinitialize(self):
        """
        
        """
        if self.rnd_wght_fact:
            W = self.weight_init_func(torch.empty((self.out_dim, self.in_dim), ))
            s = torch.distributions.normal.Normal(loc=self.fct_mean, scale=self.fct_std).sample((self.out_dim, ))
            V = torch.linalg.lstsq(torch.diag(torch.exp(s)), W)[0]
            self.s.data = s
            self.V.data = V

        else:
            self.weight.data = self.weight_init_func(torch.empty((self.out_dim, self.in_dim), ))
            
        if self.incld_bias:
            self.bias.data = self.bias_init_func(torch.empty((self.out_dim, ), ))

        return self



class MLP(torch.nn.Module):
    """
    +
    """
    def __init__(self, 
                 in_dim:int, num_layers:int, hid_dim:int, out_dim:int,
                 activ_func:Callable=torch.nn.Tanh(),
                 prd_stuff:Union[None, Dict]=None,
                 four_stuff:Union[None, Dict]=None,
                 rnd_wght_fct:bool=True
                 ):
        super().__init__()
        # in the sequential layers do the embeddings then the dense layers
        dct = OrderedDict()
        if prd_stuff:
            # dct['Period_Embed'] = Periodic_Embedding(**prd_stuff)
            dct['Period_Embed'] = Periodic_EmbeddingVer2(**prd_stuff)
        if four_stuff:
            dct['Fourier_Embed'] = Fourier_Embedding(**four_stuff)
        dct['InptLayer'] = DenseLayer(in_dim=in_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        dct['ActFunc0'] = deepcopy(activ_func)
        for i in range(1, num_layers+1):
            dct[f"HidnLay{i}"] = DenseLayer(in_dim=hid_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
            dct[f"ActFunc{i}"] = deepcopy(activ_func)
        dct['OutLayer'] = DenseLayer(in_dim=hid_dim, out_dim=out_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        self.net = torch.nn.Sequential(dct)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        The forward operation of the module. How to pass any 
        given input tensor through the module. 
        """
        return self.net(x)

class ModifiedMLP(torch.nn.Module):
    """
    
    """
    def __init__(self, 
                 in_dim:int, num_layers:int, hid_dim:int, out_dim:int,
                 activ_func:Callable=torch.nn.Tanh(),
                 prd_stuff:Union[None, Dict]=None,
                 four_stuff:Union[None, Dict]=None,
                 rnd_wght_fct:bool=True
                 ):
        super().__init__()
        # in the sequential layers do the embeddings then the dense layers
        self.num_layers = num_layers
        self.mduls = torch.nn.ModuleDict()
        if prd_stuff:
            # self.mduls['Period_Embed'] = Periodic_Embedding(**prd_stuff)
            self.mduls['Period_Embed'] = Periodic_EmbeddingVer2(**prd_stuff)
        if four_stuff:
            self.mduls['Fourier_Embed'] = Fourier_Embedding(**four_stuff)
        self.mduls[f"U"] = DenseLayer(in_dim=in_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        self.mduls['U_Act_Func'] = deepcopy(activ_func)
        self.mduls[f"V"] = DenseLayer(in_dim=in_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        self.mduls['V_Act_Func'] = deepcopy(activ_func)
        
        self.mduls[f"HidnLay{0}"] = DenseLayer(in_dim=in_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        self.mduls['ActFunc0'] = deepcopy(activ_func)
        for i in range(1, num_layers+1):
            self.mduls[f"HidnLay{i}"] = DenseLayer(in_dim=hid_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
            self.mduls[f"ActFunc{i}"] = deepcopy(activ_func)
        self.mduls['OutLayer'] = DenseLayer(in_dim=hid_dim, out_dim=out_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        The forward operation of the module. How to pass any 
        given input tensor through the module. 
        """
        if 'Period_Embed' in self.mduls.keys():
            x = self.mduls['Period_Embed'](x)
        if 'Fourier_Embed' in self.mduls.keys():
            x = self.mduls['Fourier_Embed'](x)
        U = self.mduls['U_Act_Func'](self.mduls[f"U"](x))
        V = self.mduls['V_Act_Func'](self.mduls[f"V"](x))
        for i in range(self.num_layers+1):
            x = self.mduls[f"HidnLay{i}"](x)
            x = self.mduls[f"ActFunc{i}"](x)
            x = x*U + (1-x)*V

        y = self.mduls['OutLayer'](x)
        return y

class ModifiedMLPV1(torch.nn.Module):
    """
    
    """
    def __init__(self, 
                 in_dim:int, num_layers:int, hid_dim:int, out_dim:int,
                 activ_func:Callable=torch.nn.Tanh(),
                 prd_stuff:Union[None, Dict]=None,
                 four_stuff:Union[None, Dict]=None,
                 rnd_wght_fct:bool=True
                 ):
        super().__init__()
        # in the sequential layers do the embeddings then the dense layers
        self.num_layers = num_layers
        self.mduls = torch.nn.ModuleDict()
        if prd_stuff:
            self.mduls['Period_Embed'] = Periodic_Embedding(**prd_stuff)
            # self.mduls['Period_Embed'] = Periodic_EmbeddingVer2(**prd_stuff)
        if four_stuff:
            self.mduls['Fourier_Embed'] = Fourier_Embedding(**four_stuff)
        self.mduls[f"U"] = DenseLayer(in_dim=in_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        self.mduls['U_Act_Func'] = deepcopy(activ_func)
        self.mduls[f"V"] = DenseLayer(in_dim=in_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        self.mduls['V_Act_Func'] = deepcopy(activ_func)
        
        self.mduls[f"HidnLay{0}"] = DenseLayer(in_dim=in_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        self.mduls['ActFunc0'] = deepcopy(activ_func)
        for i in range(1, num_layers+1):
            self.mduls[f"HidnLay{i}"] = DenseLayer(in_dim=hid_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
            self.mduls[f"ActFunc{i}"] = deepcopy(activ_func)
        self.mduls['OutLayer'] = DenseLayer(in_dim=hid_dim, out_dim=out_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        The forward operation of the module. How to pass any 
        given input tensor through the module. 
        """
        if 'Period_Embed' in self.mduls.keys():
            x = self.mduls['Period_Embed'](x)
        if 'Fourier_Embed' in self.mduls.keys():
            x = self.mduls['Fourier_Embed'](x)
        U = self.mduls['U_Act_Func'](self.mduls[f"U"](x))
        V = self.mduls['V_Act_Func'](self.mduls[f"V"](x))
        for i in range(self.num_layers+1):
            x = self.mduls[f"HidnLay{i}"](x)
            x = self.mduls[f"ActFunc{i}"](x)
            x = x*U + (1-x)*V

        y = self.mduls['OutLayer'](x)
        return y


class Bottleneck(torch.nn.Module):
    """
    
    """
    def __init__(self, 
                 in_dim:int, hid_dim:int, out_dim:int,
                 activ_func:Callable=torch.nn.Tanh(),
                 rnd_wght_fct:bool=True
                 ):
        super().__init__()
        
        self.activ_func = activ_func
        dct = OrderedDict()
        dct['InptLayer'] = DenseLayer(in_dim=in_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        dct['ActFunc0'] = deepcopy(activ_func)
        dct[f"HidnLay1"] = DenseLayer(in_dim=hid_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        dct[f"ActFunc1"] = deepcopy(activ_func)
        dct['OutLayer'] = DenseLayer(in_dim=hid_dim, out_dim=out_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        self.net = torch.nn.Sequential(dct)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        The forward operation of the module. How to pass any 
        given input tensor through the module. 
        """
        x = self.net(x) + x
        return self.activ_func(x)

class PIBottleneck(torch.nn.Module):
    """
    
    """
    def __init__(self, 
                 in_dim:int, hid_dim:int, out_dim:int,
                 activ_func:Callable=torch.nn.Tanh(),
                 alpha:float=0.0,
                 rnd_wght_fct:bool=True
                 ):
        super().__init__()
        
        self.activ_func = activ_func
        dct = OrderedDict()
        dct['InptLayer'] = DenseLayer(in_dim=in_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        dct['ActFunc0'] = deepcopy(activ_func)
        dct[f"HidnLay1"] = DenseLayer(in_dim=hid_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        dct[f"ActFunc1"] = deepcopy(activ_func)
        dct['OutLayer'] = DenseLayer(in_dim=hid_dim, out_dim=out_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        dct[f"outAct"] = deepcopy(activ_func)
        self.net = torch.nn.Sequential(dct)
        self.register_parameter(name='alpha', 
                    param=torch.nn.Parameter(data=torch.fill(torch.empty((1,),), alpha), requires_grad=True))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        The forward operation of the module. How to pass any 
        given input tensor through the module. 
        """
        x = self.alpha*self.net(x) + (1-self.alpha)*x
        return x

class PIModifiedBottleneck(torch.nn.Module):
    """
    
    """
    def __init__(self, 
                 in_dim:int, hid_dim:int, out_dim:int,
                 activ_func:Callable=torch.nn.Tanh(),
                 alpha:float=0.0,
                 rnd_wght_fct:bool=True
                 ):
        super().__init__()

        self.mduls = torch.nn.ModuleList([])
        self.mduls.append( DenseLayer(in_dim=in_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
                            )
        self.mduls.append(deepcopy(activ_func)
                            )
        self.mduls.append( DenseLayer(in_dim=hid_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
                            )
        self.mduls.append(deepcopy(activ_func)
                            )
        self.mduls.append( DenseLayer(in_dim=hid_dim, out_dim=out_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
                            )
        self.mduls.append(deepcopy(activ_func)
                            )
        self.register_parameter(name='alpha', 
                    param=torch.nn.Parameter(data=torch.fill(torch.empty((1,),), alpha), requires_grad=True))
        
    def forward(self, x:torch.Tensor, U:torch.Tensor, V:torch.Tensor)->torch.Tensor:
        """
        The forward operation of the module. How to pass any 
        given input tensor through the module. 
        """
        copy = x
        for i in range(len(self.mduls)//2 -1):
            x = self.mduls[2*i](x)
            x = self.mduls[2*i+1](x)
            x = x*U + (1-x)*V
        x = self.mduls[-1](self.mduls[-2](x))
        x = self.alpha*x + (1-self.alpha)*copy
        return x

class ResNet(torch.nn.Module):
    """
    
    """
    def __init__(self, 
                 in_dim:int, num_layers:int, hid_dim:int, out_dim:int,
                 activ_func:Callable=torch.nn.Tanh(),
                 prd_stuff:Union[None, Dict]=None,
                 four_stuff:Union[None, Dict]=None,
                 rnd_wght_fct:bool=True
                 ):
        super().__init__()
        # in the sequential layers do the embeddings then the dense layers
        dct = OrderedDict()
        if prd_stuff:
            dct['Period_Embed'] = Periodic_Embedding(**prd_stuff)
        if four_stuff:
            dct['Fourier_Embed'] = Fourier_Embedding(**four_stuff)
        for i in range(num_layers):
            dct[f"Bottle_{i}"] = Bottleneck(in_dim=in_dim, hid_dim=hid_dim, out_dim=in_dim,
                                            activ_func=activ_func, rnd_wght_fct=rnd_wght_fct)

        dct['OutLayer'] = DenseLayer(in_dim=in_dim, out_dim=out_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        self.net = torch.nn.Sequential(dct)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        The forward operation of the module. How to pass any 
        given input tensor through the module. 
        """
        return self.net(x)
    
class PiResNet(torch.nn.Module):
    """
    
    """
    def __init__(self, 
                 in_dim:int, num_layers:int, hid_dim:int, out_dim:int,
                 activ_func:Callable=torch.nn.Tanh(),
                 alpha:float=0.0,
                 prd_stuff:Union[None, Dict]=None,
                 four_stuff:Union[None, Dict]=None,
                 rnd_wght_fct:bool=True
                 ):
        super().__init__()
        # in the sequential layers do the embeddings then the dense layers
        dct = OrderedDict()
        if prd_stuff:
            # dct['Period_Embed'] = Periodic_Embedding(**prd_stuff)
            dct['Period_Embed'] = Periodic_EmbeddingVer2(**prd_stuff)
        if four_stuff:
            dct['Fourier_Embed'] = Fourier_Embedding(**four_stuff)
        for i in range(num_layers):
            dct[f"PiBottle_{i}"] = PIBottleneck(in_dim=in_dim, hid_dim=hid_dim, out_dim=in_dim,
                                            activ_func=activ_func, alpha=alpha, rnd_wght_fct=rnd_wght_fct)

        dct['OutLayer'] = DenseLayer(in_dim=in_dim, out_dim=out_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        self.net = torch.nn.Sequential(dct)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        The forward operation of the module. How to pass any 
        given input tensor through the module. 
        """
        return self.net(x)

class PirateNet(torch.nn.Module):
    """
    
    """
    def __init__(self, 
                 in_dim:int, num_layers:int, hid_dim:int, out_dim:int,
                 activ_func:Callable=torch.nn.Tanh(),
                 alpha:float=0.0,
                 prd_stuff:Union[None, Dict]=None,
                 four_stuff:Union[None, Dict]=None,
                 rnd_wght_fct:bool=True
                 ):
        # in the sequential layers do the embeddings then the dense layers
        super().__init__()
        self.num_layers = num_layers
        self.mduls = torch.nn.ModuleDict()
        if prd_stuff:
            # self.mduls['Period_Embed'] = Periodic_Embedding(**prd_stuff)
            self.mduls['Period_Embed'] = Periodic_EmbeddingVer2(**prd_stuff)
        if four_stuff:
            self.mduls['Fourier_Embed'] = Fourier_Embedding(**four_stuff)
        self.mduls[f"U"] = DenseLayer(in_dim=in_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        self.mduls['U_Act_Func'] = deepcopy(activ_func)
        self.mduls[f"V"] = DenseLayer(in_dim=in_dim, out_dim=hid_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
        self.mduls['V_Act_Func'] = deepcopy(activ_func)
        for i in range(num_layers):
            self.mduls[f"HidnLay{i}"] = PIModifiedBottleneck(in_dim=in_dim, hid_dim=hid_dim, out_dim=in_dim,
                                            activ_func=activ_func, alpha=alpha, rnd_wght_fct=rnd_wght_fct)
        
        self.mduls['OutLayer'] = DenseLayer(in_dim=in_dim, out_dim=out_dim, 
                                      weight_init_func=lambda z: torch.nn.init.xavier_normal_(z, gain=1.41),
                                     bias=True, bias_init_func=lambda y: torch.nn.init.zeros_(y),
                                     rnd_wght_fact=rnd_wght_fct, mean=1.0, std=0.1)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        The forward operation of the module. How to pass any 
        given input tensor through the module. 
        """
        if 'Period_Embed' in self.mduls.keys():
            x = self.mduls['Period_Embed'](x)
        if 'Fourier_Embed' in self.mduls.keys():
            x = self.mduls['Fourier_Embed'](x)
        U = self.mduls['U_Act_Func'](self.mduls[f"U"](x))
        V = self.mduls['V_Act_Func'](self.mduls[f"V"](x))
        for i in range(self.num_layers):
            x = self.mduls[f"HidnLay{i}"](x, U, V)

        y = self.mduls['OutLayer'](x)
        return y

    