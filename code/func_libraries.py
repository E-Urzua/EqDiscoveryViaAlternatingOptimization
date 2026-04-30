import abc
import warnings
from copy import deepcopy
from typing import List, Optional, Iterator, Tuple, Union
from itertools import product as iproduct
from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r

import numpy as np
import torch
from torch.autograd import grad
from PartialDerivFunctions import sptl_partials
import sparse_regress_algs as SpAlgs

from scipy.special import binom
from sklearn.preprocessing import PolynomialFeatures

def all_equal(iterator):
    """
    Function to determine if all the elements in an iterable object
    are the same (True) or not (False). Function comes from this
    stack overflow answer https://stackoverflow.com/a/3844832
    and slightly edited based off of a comment. 
    """
    iterator = iter(iterator)
    first = next(iterator, None)
    return all(first == x for x in iterator)

def monomialmulti_indices(numVar:int, maxDeg:int)->list:
    """
    Function that returns a list of lists such that the i-th list 
    contains as tuples all the combinations of the degree 1 
    monomials to form as products the ith-degree basis monomials
    for the polynomials of degree i 
    Ex: if numVar = 2 and maxDeg = 3 the output list would like 
    so:
        output[0] = []
        output[1] = [(1,), (2,)],
        output[2] = [(1, 1), (1, 2), (2, 2)],
        output[3] = [(1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2)]]
    """
    if not isinstance(numVar, int):
        raise TypeError(f"The numVar input arg must of an integer type argument")
    if not isinstance(maxDeg, int):
        raise TypeError(f"The maxDeg input arg must of an integer type argument")
    if numVar<1:
        raise ValueError(f"numVar input arg must be greater than 0 (at least 1")
    ls = [[]]
    ls.append([(i,) for i in range(1,numVar+1)])
    # now for degrees two and above
    for i in range(2, maxDeg+1):
        tmplst = []
        for tple in ls[i-1]:
            for j in range(tple[i-2], numVar+1):
                tmplst.append((*tple, j))
        if len(tmplst)!=NumDerivs(numVar, i):
            raise ValueError(f"The length of the tmplst is {len(tmplst)} when it should be {NumDerivs(numVar, i)}")
        ls.append(tmplst) 
    return ls

def NumDerivs(numVar:int, ord:int)->int:
    """The nummber partial derivative terms for numVar
    variables of deriv. order ord"""
    if not isinstance(numVar, int):
        raise TypeError(f"The numVar input arg must of an integer type argument")
    if not isinstance(ord, int):
        raise TypeError(f"The deg input arg must of an integer type argument")
    if numVar<1:
        raise ValueError(f"numVar input arg must be greater than 0 (at least 1")
    if ord<0:
        raise ValueError(f"deg input arg must be at least 0")
    return int(binom(ord+numVar-1,numVar-1))

def NumDerivsUpto(numVar:int, maxOrd:int)->int:
    """The nummber of partial derivs for numVar
    variables of order 0, 1,...maxOrd. The
    count includes the 0-th order deriv."""
    if not isinstance(numVar, int):
        raise TypeError(f"The numVar input arg must of an integer type argument")
    if not isinstance(maxOrd, int):
        raise TypeError(f"The maxDeg input arg must of an integer type argument")
    if numVar<1:
        raise ValueError(f"numVar input arg must be greater than 0 (at least 1")
    if maxOrd<0:
        raise ValueError(f"maxDeg input arg must be at least 0")
    
    return int(binom(maxOrd+numVar,numVar))
 
def combineBaseLibTerms(baseTerms:list, cmbnIDs:List[tuple])->list:

    """
    Function that given a list (baseTerms) that contains the string arguments
    produces a list of all the combinations of the terms in the list according
    to the cmbnIDs list which contains tuples of integers (1,2,3,4) that indicate
    which elements of the baseTerms list to combine (take the produce of). 
    Intended to be using in conjection with the out put of the above 
    monomialmulti_indices function
    """
    cmbdLib = []
    for tuple in cmbnIDs:
        trm = ''
        counts = np.zeros((len(baseTerms),), dtype=int)
        for j in range(len(baseTerms)):
            counts[j] = tuple.count(j+1)
        for k in range(len(counts)):
            if counts[k]>1:
                trm = trm+baseTerms[k]+'^'+str(counts[k])
            elif counts[k]==1:
                trm = trm+baseTerms[k]
        cmbdLib.append(trm)
    return cmbdLib

class BaseFuncLib():
    """
    Base class that all/any specific library function should be 
    a sub-class of. Not implemented at this time but when done
    the base class will handle most things and the specific 
    sub-classes will only need to define one main function, 
    "calcule" that calcules that library of functions using
    the nueral network and a 2D tensor of inputs as well as 
    any other necessary functions to do so. 
    """

    def __init__(self,
                 device:torch.device=torch.device('cpu'),
                 data_type:torch.dtype=None):
        
        if not isinstance(device, torch.device):
            raise TypeError(f"Input 'device' needs to be a torch.device type object")
        self.device = device
        if data_type==None:
            data_type=torch.get_default_dtype()
        if not isinstance(data_type, torch.dtype):
            raise TypeError(f"Data type argument needs to be a torch.dtype object ")
        self.data_type = data_type
    
    @abc.abstractmethod
    def get_library_names(self, net_out_func_names=None)->List[str]:
        """ Return library fucntion names. Input arguments are...

            * net_out_func_names : list of strings objects. Length needs
                to tbe the same as the number of neural network outputs. 
                Optional. By default library function names are
                "x0", "x1", ... "xn_features" is used.

            Return object is...
            
                lib_names : list of string objects, length num library 
                    functions
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def fit(self, network, inpts=torch.Tensor):
        """
            Determines/Calculates the number of functions there are in the
            library of candidate functions. Input arguments are...

                * network -  A neural network that can take the 
                    inpts argument as input to the network and 
                    return 2D tensor output. Most generally
                    though this argument can be any class/object 
                    that has as a .forward() methods which can 
                    take the inpts argument and return a 2D 
                    tensor. 

                * inpts - A 2D tensor that is passed through as
                    the input to the .forward() method of the 
                    network object. 

        There is no returned object. 
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def Calc(self, network, inpts:torch.Tensor)->torch.Tensor:
        """
            Function that calculates the entire funciton library over all
            the points found in the inpt argument. Input arguments are...

                * network -  A neural network that can take the 
                    inpts argument as input to the network and 
                    return 2D tensor output. Most generally
                    though this argument can be any class/object 
                    that has as a .forward() methods which can 
                    take the inpts argument and return a 2D 
                    tensor. 

                * inpts - A 2D tensor that is passed through as
                    the input to the .forward() method of the 
                    network object. The 'points' that all the 
                    library functions are evaluated at are the 
                    rows of this 2D tensor. 
            
            The returned object is a 2D torch tensor called lib

                * lib - A 2D tensor such that the (i,j) entry is 
                    the value of the j-th candidate function
                    evaluated at the i-th point (i.e point that
                    constitutes the ith row in inpts)

        """
        raise NotImplementedError
    
    def Get_Lib_Complexities(self):
        raise NotImplementedError

    @property
    def size(self):
        #  come up with a way of determining that the library 
        # has be fit. i.e, any and all set up functions/routines
        # have been run 
        # self.check_setup(self)
        return self.n_output_features_


class LibConCate(BaseFuncLib):

    """
    Concatenation of two function libraries. 

    Input prarmeters are the following...

        * libraries - list of libraries. Each library in the list
            should be a child of the BaseFunLib class and implement
            any and all required methods/functions needed to calc.
            the library functions. 

    """

    def __init__(self, libraries:List[BaseFuncLib],
                 device = torch.device('cpu'),
                 data_type = None):
        super().__init__(device, data_type)

        self.libraries = libraries

    def get_library_names(self, net_out_func_names=None):
        """
            
        """
        # the library names just the names found in each
        # of the concatenated library functions. 
        lib_names = []
        for lib in self.libraries:
            lib_names += lib.get_library_names(net_out_func_names=net_out_func_names)

        return lib_names
        

    def fit(self, network, inpts=torch.Tensor):
        """
            Determines/Calculates the number of functions there are in the
            library of candidate functions. Input arguments are...

                * network -  A neural network that can take the 
                    inpts argument as input to the network and 
                    return 2D tensor output. Most generally
                    though this argument can be any class/object 
                    that has as a .forward() methods which can 
                    take the inpts argument and return a 2D 
                    tensor. 

                * inpts - A 2D tensor that is passed through as
                    the input to the .forward() method of the 
                    network object. 

            There is no returned object. 
        """
    
        if inpts.device!=self.device or inpts.dtype!=self.data_type:
            raise TypeError(f"The input dtype or device is not what was expected")

        outputs = network.forward(inpts).detach()
        if outputs.ndim!=2:
            raise ValueError(f"The neural network output is expected to be a 2D tensor. Doesn't have 2 dims.")

        n_features = outputs.size(1)
        self.n_features_in_ = n_features

        # run the .fit() method for each of the libraries in the list
        post_fit = [lib.fit(network, inpts) for lib in self.libraries]
        # the number of lib funcs is the sum of all the libraries being concatenated. 
        self.n_output_features_ = sum([lib.n_output_features_ for lib in post_fit])

        self.libraries = post_fit
    

    def Calc(self, network, inpts:torch.Tensor):
        """

        """
        # again should implement a function that 
        # checks if the libraies have had any and all
        # necessary/applicable setup functions run
        # before doing this. 

        concat_lib_evals = torch.concat([lib.Calc(network, inpts) for lib in self.libraries], dim=1)

        return concat_lib_evals

class LibProducts(BaseFuncLib):

    def __init__(self, 
                libraries:List[BaseFuncLib],
                device = torch.device('cpu'), 
                data_type = None):
        super().__init__(device, data_type)


    def get_library_names(self, net_out_func_names=None):
        """
        
        """
        raise NotImplementedError
    def fit(self, network, inpts=torch.Tensor):
        """
        
        """
        raise NotImplementedError
    def Calc(self, network, inpts:torch.Tensor):
        """
        
        """
        raise NotImplementedError

class ODE_Poly_Deriv_Library(BaseFuncLib):
    """
    Class for the sole purpose of calculating a large library of
    candidate functions over a large amount of points given the 
    points as a 2D tensor and a nueral network. 
    """

    def __init__(self,
            poly_degree:int=2,
            include_poly_interaction:bool=True,
            poly_interaction_only:bool=False,
            derivative_order:int=0,
            include_bias:bool=False,
            deriv_exclude_inds:list=None,
            device:torch.device=torch.device('cpu'),
            data_type:torch.dtype=None
    ):
        super().__init__(device, data_type) # eventually when needed
        # check the inputs
        if not isinstance(poly_degree, int):
            raise TypeError(f"Input 'poly_degree' needs to be an integer type object")
        if poly_degree<0:
            raise ValueError(f"Input 'poly_degree' needs to be a postive integer.")
        self.poly_degree = poly_degree

        if not isinstance(include_poly_interaction, bool):
            raise TypeError(f"The 'include_poly_interaction' needs to be a boolean object")
        self.include_poly_interaction = include_poly_interaction

        if not isinstance(poly_interaction_only, bool):
            raise TypeError(f"The 'include_poly_interaction' needs to be a boolean object")
        self.poly_interaction_only = poly_interaction_only

        if not isinstance(derivative_order, int):
            raise TypeError(f"Input 'derivative_order' needs to be an integer type object")
        if derivative_order<0:
            raise ValueError(f"Input 'derivative_order' needs to be a postive integer.")
        self.derivative_order = derivative_order

        if not isinstance(deriv_exclude_inds, list) and deriv_exclude_inds!=None:
            raise TypeError(f"Input 'multi_indices' needs to be a list (of lists).")

        self.include_bias = include_bias

        if deriv_exclude_inds is None:
            self.deriv_inds = [torch.arange(0, self.derivative_order, device=self.device)]
            self.deriv_exclude_inds = deriv_exclude_inds
            self.num_derives = self.deriv_inds[0].size(0)
        else:
            excld_drvs = np.array(deriv_exclude_inds, dtype=int)

            # do some checks on the multi indices -  maybe make a function to do all of this later...
            if excld_drvs.ndim!=2:
                raise ValueError(f"The 'multi_indices' argument when converted so an np.array needs to be two dimensional")
            
            if np.any(excld_drvs<1):
                raise ValueError(f"One of the elements in one of the multi_indices is less than one. All elements needs to be integers and at least one. ")
            
            if np.any(excld_drvs>self.derivative_order):
                raise ValueError(f"A derivative order given within the \'deriv_exclude_inds\' argument is\n greater than the derivative_order value given...\n{excld_drvs}")
            
            deriv_inds = []
            self.num_derives = 0
            for row in excld_drvs:
                deriv_inds.append(torch.from_numpy(np.setdiff1d(np.arange(self.derivative_order, dtype=int), row-1)).to(device=self.device))
                self.num_derives += deriv_inds[-1].size(0)
            self.deriv_exclude_inds = deriv_exclude_inds

    @staticmethod
    def _combinations(
        n_features: int,
        degree: int,
        include_interaction: bool,
        interaction_only: bool,
        include_bias: bool,
    ) -> Iterator[Tuple[int, ...]]:
        """
        Create selection tuples of input indexes for each polynomail term

        Selection tuple iterates the input indexes present in a single term.
        For example, (x+y+1)^2 would be in iterator of the tuples:
        (), (0,), (1,), (0, 0), (0, 1), (1, 1)
        1    x     y      x^2     x*y     y^2

        Order of terms is preserved regardless of include_interation.
        """
        if not include_interaction:
            return chain(
                [()] if include_bias else [],
                (
                    exponent * (feat_idx,)
                    for exponent in range(1, degree + 1)
                    for feat_idx in range(n_features)
                ),
            )
        return PolynomialFeatures._combinations(
            n_features=n_features,
            min_degree=int(not include_bias),
            max_degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
        )

    @property
    def powers_(self):
        """
        The exponents of the polynomial as an array of shape
        (n_features_out, n_features_in), where each item is the exponent of the
        jth input variable in the ith polynomial term.
        """
        
        combinations = self._combinations(
            n_features=self.n_features_in_ + self.num_derives,
            degree=self.poly_degree,
            include_interaction=self.include_poly_interaction,
            interaction_only=self.poly_interaction_only,
            include_bias=self.include_bias,
        )
        lngth = self.n_features_in_ + self.num_derives

        return np.vstack(
            [np.bincount(c, minlength=lngth) for c in combinations]
        )

    
    def get_library_names(self, net_out_func_names=None)->List[str]:
        """ 
            Return library fucntion names. Input arguments are...

                * net_out_func_names : list of strings objects. Length needs
                    to tbe the same as the number of neural network outputs. 
                    Optional. By default library function names are
                    "x0", "x1", ... "xn_features" is used.

            Return object is...
            
                lib_names : list of string objects, length num library 
                    functions
        """
        if net_out_func_names==None:
            net_out_func_names = [f"f{i}(t)" for i in range(self.n_features_in_)]

        if self.n_features_in_!=len(net_out_func_names):
            raise ValueError(f"Then number of net_out_func_names is less than the number of network output outputs.")
        
        base_funcs = deepcopy(net_out_func_names)
        
        for i in range(len(base_funcs)):
            for row in self.deriv_inds:
                func = base_funcs[i]+"_"
                for val in row:
                    func += (val+1)*"t"
                base_funcs.append(func)

        powers = self.powers_
        lib_names = []
        # idx = 0 if self.include_bias
        for row in powers:
            inds = np.where(row)[0]
            if len(inds)==1:
                ind = inds[0]
                exp = row[ind]
                name = "(%s)^%d" % (base_funcs[ind], exp) if exp != 1 else base_funcs[ind] 
            elif len(inds)>=2:
                name = ""
                for ind, exp in zip(inds, row[inds]):
                    name += "(%s)^%d" % (base_funcs[ind], exp) if exp != 1 else "(%s)" % base_funcs[ind]
            else:
                name = "1"
            lib_names.append(name)

        return lib_names


    def fit(self, network, inpts=torch.Tensor):
        """
            Determines/Calculates the number of functions there are in the
            library of candidate functions. Input arguments are...

                * network -  A neural network that can take the 
                    inpts argument as input to the network and 
                    return 2D tensor output. Most generally
                    though this argument can be any class/object 
                    that has as a .forward() methods which can 
                    take the inpts argument and return a 2D 
                    tensor. 

                * inpts - A 2D tensor that is passed through as
                    the input to the .forward() method of the 
                    network object. 

        There is no returned object. 
        """
        if self.poly_degree<0 or not isinstance(self.poly_degree, int):
            raise ValueError("degree must be a nonnegative integer")
        if (not self.include_poly_interaction) and self.poly_interaction_only:
            raise ValueError(
                "Can't have include_interaction be False and interaction_only"
                " be True"
            )   
        
        if inpts.device!=self.device or inpts.dtype!=self.data_type:
            raise TypeError(f"The input dtype or device is not what was expected")

        outputs = network.forward(inpts).detach()
        if outputs.ndim!=2:
            raise ValueError(f"The neural network output is expected to be a 2D tensor. Doesn't have 2 dims.")

        n_features = outputs.size(1)
        self.n_features_in_ = n_features
        if self.deriv_exclude_inds is None:
            self.num_derives = self.deriv_inds[0].size(0)
            for _ in range(1, n_features):
                self.deriv_inds.append(torch.arange(0, self.derivative_order, device=self.device))
            self.num_derives = n_features * self.derivative_order

        # With the number of derivatves get the total number of out features
        combinations = self._combinations(
            n_features + self.num_derives,
            self.poly_degree,
            self.include_poly_interaction,
            self.poly_interaction_only,
            self.include_bias,
        )
        self.n_output_features_ = sum(1 for _ in combinations)

        # required to generate the function names
        self.get_library_names()

        return self
    
    
    def Calc(self, network, inpts:torch.Tensor)->torch.Tensor:
        """
            Function that calculates the entire funciton library over all
            the points found in the inpt argument. Input arguments are...

                * network -  A neural network that can take the 
                    inpts argument as input to the network and 
                    return 2D tensor output. Most generally
                    though this argument can be any class/object 
                    that has as a .forward() methods which can 
                    take the inpts argument and return a 2D 
                    tensor. 

                * inpts - A 2D tensor that is passed through as
                    the input to the .forward() method of the 
                    network object. The 'points' that all the 
                    library functions are evaluated at are the 
                    rows of this 2D tensor. 
            
            The returned object is a 2D torch tensor called lib

                * lib - A 2D tensor such that the (i,j) entry is 
                    the value of the j-th candidate function
                    evaluated at the i-th point (i.e point that
                    constitutes the ith row in inpts)

        """

        if inpts.device!=self.device or inpts.dtype!=self.data_type:
            raise TypeError(f"The input dtype or device is not what was expected")
        
        if not inpts.requires_grad:
            inpts.requires_grad_(True)

        idx = 1 
        if not self.include_bias:
            lib = torch.ones((inpts.size(0), 1+ self.n_output_features_),device=self.device, dtype=self.data_type)
        else:
            lib = torch.ones((inpts.size(0), self.n_output_features_),device=self.device, dtype=self.data_type)

        lib[:, idx:self.n_features_in_+idx] = network.forward(inpts)
        idx += self.n_features_in_
        if self.derivative_order>0:
            for i in range(self.n_features_in_):
                n_deriv = self.deriv_inds[i].size(0)
                lib[:, idx+i*n_deriv:idx+ (i+1)*n_deriv] = sptl_partials(values=network.forward(inpts)[:, i:i+1], pts=inpts, order=self.derivative_order)[:, self.deriv_inds[i]]
                idx += n_deriv
        # idx += self.num_prtls*self.n_features_in_ 
        
        combinations = self._combinations(
            self.num_derives + self.n_features_in_ ,
            self.poly_degree,
            self.include_poly_interaction,
            self.poly_interaction_only,
            self.include_bias,
        )
        idx = 0 if self.include_bias else 1
        for i, comb in enumerate(combinations):
            if len(comb)<=1:
                continue
            lib[:,i+idx] = lib[:, np.array(comb)+1].prod(-1)
        
        return lib[:,-self.n_output_features_:]

    def Get_Lib_Complexities(self)->np.ndarray:
        
        # now get the complexities for each libray 
        # functions. This is very easy to calc. for 
        # this library since its only polynomials 
        # combinations and we consider each "base"
        # term to have complexity 1. 
        pwrs = self.powers_
        complexities = 2*pwrs.sum(axis=1) - 1
        if self.include_bias:
            # the complexity of a constant is one
            # not zero. 
            complexities[0] = 1
        return complexities


class Poly_Deriv_Library(BaseFuncLib):
    """
    Class for the sole purpose of calculating a large library of
    candidate functions over a large amount of points given the 
    points as a 2D tensor and a nueral network. 
    """

    def __init__(self,
            poly_degree:int=2,
            include_poly_interaction:bool=True,
            poly_interaction_only:bool=False,
            derivative_order:int=0,
            sptl_dims:int=1,
            include_bias:bool=False,
            include_deriv_interaction:bool=True,
            multi_indices:list=None,
            device:torch.device=torch.device('cpu'),
            data_type:torch.dtype=None
    ):
        super().__init__(device, data_type) # eventually when needed
        # check the inputs
        if not isinstance(poly_degree, int):
            raise TypeError(f"Input 'poly_degree' needs to be an integer type object")
        if poly_degree<0:
            raise ValueError(f"Input 'poly_degree' needs to be a postive integer.")
        self.poly_degree = poly_degree

        if not isinstance(include_poly_interaction, bool):
            raise TypeError(f"The 'include_poly_interaction' needs to be a boolean object")
        self.include_poly_interaction = include_poly_interaction

        if not isinstance(poly_interaction_only, bool):
            raise TypeError(f"The 'include_poly_interaction' needs to be a boolean object")
        self.poly_interaction_only = poly_interaction_only

        if not isinstance(derivative_order, int):
            raise TypeError(f"Input 'derivative_order' needs to be an integer type object")
        if derivative_order<0:
            raise ValueError(f"Input 'derivative_order' needs to be a postive integer.")
        self.derivative_order = derivative_order

        # NOTE do something about if sptl_dims=0, (either new class of something else)
        if not isinstance(sptl_dims, int):
            raise TypeError(f"Input 'sptl_dims' needs to be an integer type object")
        if sptl_dims<0:
            raise ValueError(f"Input 'sptl_dims' needs to be a postive integer.")
        self.sptl_dims = sptl_dims

        if not isinstance(poly_interaction_only, bool):
            raise TypeError(f"The 'include_poly_interaction' needs to be a boolean object")
        self.include_bias = include_bias

        if not isinstance(include_deriv_interaction, bool):
            raise TypeError(f"The 'include_deriv_interaction' needs to be a boolean object")
        self.include_deriv_interaction = include_deriv_interaction
        
        if not isinstance(multi_indices, list) and multi_indices!=None:
            raise TypeError(f"Input 'multi_indices' needs to be a list (of lists).")

        # if not isinstance(device, torch.device):
        #     raise TypeError(f"Input 'device' needs to be a torch.device type object")
        # self.device = device
        # if not isinstance(data_type, torch.dtype):
        #     raise TypeError(f"Input 'data_type' needs to be a torch.dtype type object")
        # self.data_type = data_type
            
        if multi_indices==None:
            num_prtls = NumDerivsUpto(numVar=sptl_dims, maxOrd=derivative_order) - 1
            self.deriv_inds = torch.arange(start=0, end=num_prtls, device=self.device)
            self.multi_indices = self.prtls_multi_inds_
        else:
            gvn_inds = np.array(multi_indices, dtype=int)

            # do some checks on the multi indices -  maybe make a function to do all of this later...
            if gvn_inds.ndim!=2:
                raise ValueError(f"The 'multi_indices' argument when converted so an np.array needs to be two dimensional")
            
            if self.sptl_dims==0 and gvn_inds.shape[1]!=1:
                raise ValueError(f"In the case of zero (0) spatial dims, still needs to one element in each multi_index with the multi_indices ")
            elif gvn_inds.shape[1]!=sptl_dims:
                raise ValueError(f"The number of elements in each multi_index with the multi_indices needs to be the same as the number of spatial dims ")
            
            if np.any(gvn_inds<1):
                raise ValueError(f"One of the elements in one of the multi_indices is less than one. All elements needs to be integers and at least one. ")
            
            if np.any(np.logical_and(gvn_inds.sum(axis=1)>=1, gvn_inds.sum(axis=1)<derivative_order)):
                raise ValueError(f"The derivative order given within the 'multi_indices' argument is\n greater than the derivative_order value given...\n{gvn_inds}")
            
            all_mult_inds = self.prtls_multi_inds_
            deriv_inds = []
            tmp_inds = []
            for indx in gvn_inds:
                bool_vals = np.all(indx == all_mult_inds, axis=1)
                if np.any(bool_vals):
                    deriv_inds.append(np.nonzero(bool_vals)[0].item())
                    tmp_inds.append(indx)

            self.deriv_inds = torch.unique(torch.tensor(deriv_inds, device=self.device))
            self.multi_indices = np.array(tmp_inds)
            
        self.num_prtls = self.multi_indices.shape[0]        
        if self.num_prtls<self.deriv_inds.size(0):
            raise ValueError(f"Too many multi_indices. The number of multi_indices must be <= total number of derivatives (Possibly dublicate multi_inds)")


    @staticmethod
    def _combinations(
        n_features: int,
        degree: int,
        include_interaction: bool,
        interaction_only: bool,
        include_bias: bool,
    ) -> Iterator[Tuple[int, ...]]:
        """
        Create selection tuples of input indexes for each polynomail term

        Selection tuple iterates the input indexes present in a single term.
        For example, (x+y+1)^2 would be in iterator of the tuples:
        (), (0,), (1,), (0, 0), (0, 1), (1, 1)
        1    x     y      x^2     x*y     y^2

        Order of terms is preserved regardless of include_interation.
        """
        if not include_interaction:
            return chain(
                [()] if include_bias else [],
                (
                    exponent * (feat_idx,)
                    for exponent in range(1, degree + 1)
                    for feat_idx in range(n_features)
                ),
            )
        return PolynomialFeatures._combinations(
            n_features=n_features,
            min_degree=int(not include_bias),
            max_degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
        )

    @property
    def powers_(self):
        """
        The exponents of the polynomial as an array of shape
        (n_features_out, n_features_in), where each item is the exponent of the
        jth input variable in the ith polynomial term.
        """
        
        combinations = self._combinations(
            n_features=self.n_features_in_ + self.num_prtls*self.n_features_in_,
            degree=self.poly_degree,
            include_interaction=self.include_poly_interaction,
            interaction_only=self.poly_interaction_only,
            include_bias=self.include_bias,
        )
        lngth = self.n_features_in_ + self.n_features_in_ *  self.num_prtls

        return np.vstack(
            [np.bincount(c, minlength=lngth) for c in combinations]
        )
    
    @property
    def prtls_multi_inds_(self):

        if self.sptl_dims==0:
            return np.arange(1, self.derivative_order+1, dtype=int)[:, np.newaxis]


        combs = PolynomialFeatures._combinations(
            n_features=self.sptl_dims,
            min_degree=1,
            max_degree=self.derivative_order,
            interaction_only=False,
            include_bias=False,
        )
        
        return np.vstack(
            [np.bincount(c, minlength=self.sptl_dims) for c in combs]
        )

    
    def get_library_names(self, net_out_func_names=None)->List[str]:
        """ 
            Return library fucntion names. Input arguments are...

                * net_out_func_names : list of strings objects. Length needs
                    to tbe the same as the number of neural network outputs. 
                    Optional. By default library function names are
                    "x0", "x1", ... "xn_features" is used.

            Return object is...
            
                lib_names : list of string objects, length num library 
                    functions
        """
        if net_out_func_names==None:
            net_out_func_names = [f"f{i}(...)" for i in range(self.n_features_in_)]

        if self.n_features_in_!=len(net_out_func_names):
            raise ValueError(f"Then number of net_out_func_names is less than the number of network output outputs.")
        
        base_funcs = deepcopy(net_out_func_names)
        if self.sptl_dims>=1:
            sptl_vars = ["x%d"%i for i in range(self.sptl_dims)]
        else:
            sptl_vars = ["t"]
        
        for i in range(len(base_funcs)):
            for row in self.multi_indices:
                func = base_funcs[i]+"_"
                for j, val in enumerate(row):
                    func += val*("%s" %sptl_vars[j])
                base_funcs.append(func)

        powers = self.powers_
        lib_names = []
        # idx = 0 if self.include_bias
        for row in powers:
            inds = np.where(row)[0]
            if len(inds)==1:
                ind = inds[0]
                exp = row[ind]
                name = "(%s)^%d" % (base_funcs[ind], exp) if exp != 1 else base_funcs[ind] 
            elif len(inds)>=2:
                name = ""
                for ind, exp in zip(inds, row[inds]):
                    name += "(%s)^%d" % (base_funcs[ind], exp) if exp != 1 else "(%s)" % base_funcs[ind]
            else:
                name = "1"
            lib_names.append(name)

        return lib_names


    def fit(self, network, inpts=torch.Tensor):
        """
            Determines/Calculates the number of functions there are in the
            library of candidate functions. Input arguments are...

                * network -  A neural network that can take the 
                    inpts argument as input to the network and 
                    return 2D tensor output. Most generally
                    though this argument can be any class/object 
                    that has as a .forward() methods which can 
                    take the inpts argument and return a 2D 
                    tensor. 

                * inpts - A 2D tensor that is passed through as
                    the input to the .forward() method of the 
                    network object. 

        There is no returned object. 
        """
        if self.poly_degree<0 or not isinstance(self.poly_degree, int):
            raise ValueError("degree must be a nonnegative integer")
        if (not self.include_poly_interaction) and self.poly_interaction_only:
            raise ValueError(
                "Can't have include_interaction be False and interaction_only"
                " be True"
            )   
        
        if inpts.device!=self.device or inpts.dtype!=self.data_type:
            raise TypeError(f"The input dtype or device is not what was expected")

        outputs = network.forward(inpts).detach()
        if outputs.ndim!=2:
            raise ValueError(f"The neural network output is expected to be a 2D tensor. Doesn't have 2 dims.")

        n_features = outputs.size(1)
        self.n_features_in_ = n_features

        tot_n_derivs = n_features * self.num_prtls

        # With the number of derivatves get the total number of out features
        combinations = self._combinations(
            n_features + tot_n_derivs,
            self.poly_degree,
            self.include_poly_interaction,
            self.poly_interaction_only,
            self.include_bias,
        )
        self.n_output_features_ = sum(1 for _ in combinations)

        # required to generate the function names
        self.get_library_names()

        return self
    
    
    def Calc(self, network, inpts:torch.Tensor)->torch.Tensor:
        """
            Function that calculates the entire funciton library over all
            the points found in the inpt argument. Input arguments are...

                * network -  A neural network that can take the 
                    inpts argument as input to the network and 
                    return 2D tensor output. Most generally
                    though this argument can be any class/object 
                    that has as a .forward() methods which can 
                    take the inpts argument and return a 2D 
                    tensor. 

                * inpts - A 2D tensor that is passed through as
                    the input to the .forward() method of the 
                    network object. The 'points' that all the 
                    library functions are evaluated at are the 
                    rows of this 2D tensor. 
            
            The returned object is a 2D torch tensor called lib

                * lib - A 2D tensor such that the (i,j) entry is 
                    the value of the j-th candidate function
                    evaluated at the i-th point (i.e point that
                    constitutes the ith row in inpts)

        """

        if inpts.device!=self.device or inpts.dtype!=self.data_type:
            raise TypeError(f"The input dtype or device is not what was expected")
        
        if not inpts.requires_grad:
            inpts.requires_grad_(True)

        idx = 1 
        if not self.include_bias:
            lib = torch.ones((inpts.size(0), 1+ self.n_output_features_),device=self.device, dtype=self.data_type)
        else:
            lib = torch.ones((inpts.size(0), self.n_output_features_),device=self.device, dtype=self.data_type)

        lib[:, idx:self.n_features_in_+idx] = network.forward(inpts)
        idx += self.n_features_in_
        for i in range(self.n_features_in_):
            lib[:, idx+i*self.num_prtls:idx+ (i+1)*self.num_prtls] = sptl_partials(values=network.forward(inpts)[:, i:i+1], pts=inpts, order=self.derivative_order)[:, self.deriv_inds]
        # idx += self.num_prtls*self.n_features_in_ 
        
        combinations = self._combinations(
            self.num_prtls*self.n_features_in_ + self.n_features_in_ ,
            self.poly_degree,
            self.include_poly_interaction,
            self.poly_interaction_only,
            self.include_bias,
        )
        idx = 0 if self.include_bias else 1
        for i, comb in enumerate(combinations):
            if len(comb)<=1:
                continue
            lib[:,i+idx] = lib[:, np.array(comb)+1].prod(-1)
        
        return lib[:,-self.n_output_features_:]

    def Get_Lib_Complexities(self)->np.ndarray:
        
        # now get the complexities for each libray 
        # functions. This is very easy to calc. for 
        # this library since its only polynomials 
        # combinations and we consider each "base"
        # term to have complexity 1. 
        pwrs = self.powers_
        complexities = 2*pwrs.sum(axis=1) - 1
        if self.include_bias:
            # the complexity of a constant is one
            # not zero. 
            complexities[0] = 1
        return complexities

class Custom_Library(BaseFuncLib):
    """
    
    """

    def __init__(self,
                 functions:List,
                 func_complexities:Union[List[int], np.ndarray],
                 func_names:List[str]=None,
                 funct_interacts:bool=False,
                 include_bias:bool=True,
                 device:torch.device=torch.device('cpu'),
                 data_type:torch.dtype=None):
        """
        
        """
        super().__init__(device, data_type)
        # check the input arguments (later)
        self.functions = functions
        if not isinstance(func_complexities, (list, np.ndarray)):
            raise TypeError(f"The func_complexities argument needs to be a list of np.ndarray object")
        if isinstance(func_complexities, list):
            if any([not isinstance(c, int) for c in func_complexities]):
                raise TypeError("One of the given function complexity values was not an integer")
        elif isinstance(func_complexities, np.ndarray):
            if not np.issubdtype(func_complexities.dtype, np.integer):
                raise TypeError(f"The given func_complexities argument is a np.ndarray but not an array of integers.")
        self.func_complexitites = np.asarray(func_complexities)
        if func_names and (not isinstance(func_names, list)):
            raise TypeError(f"The func_names argument needs to be a list object or just None")
        if isinstance(func_names, list):
            if any([not isinstance(name, str) for name in func_names]):
                raise TypeError(f"One of the elements in the func_names list is not a string object")
        self.func_names = func_names

        self.funct_interacts = funct_interacts
        self.include_bias = include_bias
        
    def get_library_names(self, net_out_func_names):
        """

        """

        if self.func_names==None:
            func_names = [f"g{i}(...)" for i in range(self.n_output_features_)]
        else:
            func_names = deepcopy(self.func_names)

        # Check if the number of functions names is the same as the number of output function
        if len(func_names)!=(self.n_output_features_ - int(self.include_bias)):
            raise ValueError(f"The number of function names differs from the number of output library functions")
        
        if self.include_bias:
            func_names.insert(0, "1")
        
        return func_names

    def fit(self, network, inpts:torch.Tensor):
        """

        """
        n_out_funcs = 0
        self.funcs_out_sizes = []
        for i,f in enumerate(self.functions):
            f_out = f(inpts)
            if f_out.ndim !=2:
                raise ValueError(f"The {i}-th function in the given functions list did not return a 2D tensor")
            self.funcs_out_sizes.append(f_out.size(1))
            n_out_funcs += f_out.size(1)

        self.n_output_features_ = n_out_funcs + int(self.include_bias)
        self.get_library_names(net_out_func_names=None)

    def Calc(self, network, inpts:torch.Tensor):
        """

        """

        lib = torch.ones(size=(inpts.shape[0], self.n_output_features_ ), device=self.device, dtype=self.data_type)
        idx = 1 if self.include_bias else 0

        for i, f in enumerate(self.functions):
            lib[:, idx:idx + self.funcs_out_sizes[i]] = f(inpts)
            idx += self.funcs_out_sizes[i]

        if lib.shape[1] != (self.n_output_features_ ):
            raise ValueError(f"The number of calculated outputs functions differs from the number of function names")

        return lib

    def Get_Lib_Complexities(self)->np.ndarray:
        """
            Need to define so that it works well with the other library stuff. 
        """
        # remember the complexity of the constant needs to be included
        # and that its value is 1
        if self.include_bias:
            return np.concat(([1], self.func_complexitites),)
        else:
            return self.func_complexitites

class Expanded_Poly_Deriv_Library(BaseFuncLib):
    """
    Class for the sole purpose of calculating a large library of
    candidate functions over a large amount of points given the 
    points as a 2D tensor and a nueral network. 
    """

    def __init__(self,
            xtra_funcs:BaseFuncLib=None,
            xtra_func_complex:Union[List[int], np.ndarray]=None,
            poly_degree:int=2,
            include_poly_interaction:bool=True,
            poly_interaction_only:bool=False,
            derivative_order:int=0,
            sptl_dims:int=1,
            include_deriv_interaction:bool=True,
            include_bias:bool=False,
            multi_indices:list=None,
            device:torch.device=torch.device('cpu'),
            data_type:torch.dtype=None
    ):
        super().__init__(device, data_type) # eventually when needed
        # check the inputs
        self.xtra_funcs = xtra_funcs
        if not isinstance(poly_degree, int):
            raise TypeError(f"Input 'poly_degree' needs to be an integer type object")
        
        if xtra_func_complex is None:
            raise TypeError
        if not isinstance(xtra_func_complex, (list, np.ndarray)):
            raise TypeError(f"The xtra_func_complex argument needs to be a list of np.ndarray object")
        if isinstance(xtra_func_complex, list):
            if any([not isinstance(c, int) for c in xtra_func_complex]):
                raise TypeError("One of the given function complexity values was not an integer")
        elif isinstance(xtra_func_complex, np.ndarray):
            if not np.issubdtype(xtra_func_complex.dtype, np.integer):
                raise TypeError(f"The given xtra_func_complex argument is a np.ndarray but not an array of integers.")
        
        if poly_degree<0:
            raise ValueError(f"Input 'poly_degree' needs to be a postive integer.")
        self.poly_degree = poly_degree

        if not isinstance(include_poly_interaction, bool):
            raise TypeError(f"The 'include_poly_interaction' needs to be a boolean object")
        self.include_poly_interaction = include_poly_interaction

        if not isinstance(poly_interaction_only, bool):
            raise TypeError(f"The 'include_poly_interaction' needs to be a boolean object")
        self.poly_interaction_only = poly_interaction_only

        if not isinstance(derivative_order, int):
            raise TypeError(f"Input 'derivative_order' needs to be an integer type object")
        if derivative_order<0:
            raise ValueError(f"Input 'derivative_order' needs to be a postive integer.")
        self.derivative_order = derivative_order

        # NOTE do something about if sptl_dims=0, (either new class of something else)
        if not isinstance(sptl_dims, int):
            raise TypeError(f"Input 'sptl_dims' needs to be an integer type object")
        if sptl_dims<0:
            raise ValueError(f"Input 'sptl_dims' needs to be a postive integer.")
        self.sptl_dims = sptl_dims

        if not isinstance(poly_interaction_only, bool):
            raise TypeError(f"The 'include_poly_interaction' needs to be a boolean object")
        self.include_bias = include_bias

        if not isinstance(include_deriv_interaction, bool):
            raise TypeError(f"The 'include_deriv_interaction' needs to be a boolean object")
        self.include_deriv_interaction = include_deriv_interaction
        
        if not isinstance(multi_indices, list) and multi_indices!=None:
            raise TypeError(f"Input 'multi_indices' needs to be a list (of lists).")

        if not isinstance(device, torch.device):
            raise TypeError(f"Input 'device' needs to be a torch.device type object")
        self.device = device
        if not isinstance(data_type, torch.dtype):
            raise TypeError(f"Input 'data_type' needs to be a torch.dtype type object")
        self.data_type = data_type
            
        if multi_indices==None:
            num_prtls = NumDerivsUpto(numVar=sptl_dims, maxOrd=derivative_order) - 1
            self.deriv_inds = torch.arange(start=0, end=num_prtls, device=self.device)
            self.multi_indices = self.prtls_multi_inds_
        else:
            gvn_inds = np.array(multi_indices, dtype=int)

            # do some checks on the multi indices -  maybe make a function to do all of this later...
            if gvn_inds.ndim!=2:
                raise ValueError(f"The 'multi_indices' argument when converted so an np.array needs to be two dimensional")
            
            if self.sptl_dims==0 and gvn_inds.shape[1]!=1:
                raise ValueError(f"In the case of zero (0) spatial dims, still needs to one element in each multi_index with the multi_indices ")
            elif gvn_inds.shape[1]!=sptl_dims:
                raise ValueError(f"The number of elements in each multi_index with the multi_indices needs to be the same as the number of spatial dims ")
            
            if np.any(gvn_inds<1):
                raise ValueError(f"One of the elements in one of the multi_indices is less than one. All elements needs to be integers and at least one. ")
            
            if np.any(np.logical_and(gvn_inds.sum(axis=1)>=1, gvn_inds.sum(axis=1)<derivative_order)):
                raise ValueError(f"The derivative order given within the 'multi_indices' argument is\n greater than the derivative_order value given...\n{gvn_inds}")
            
            all_mult_inds = self.prtls_multi_inds_
            deriv_inds = []
            tmp_inds = []
            for indx in gvn_inds:
                bool_vals = np.all(indx == all_mult_inds, axis=1)
                if np.any(bool_vals):
                    deriv_inds.append(np.nonzero(bool_vals)[0].item())
                    tmp_inds.append(indx)

            self.deriv_inds = torch.unique(torch.tensor(deriv_inds, device=self.device))
            self.multi_indices = np.array(tmp_inds)
            
        self.num_prtls = self.multi_indices.shape[0]        
        if self.num_prtls<self.deriv_inds.size(0):
            raise ValueError(f"Too many multi_indices. The number of multi_indices must be <= total number of derivatives (Possibly dublicate multi_inds)")


    @staticmethod
    def _combinations(
        n_features: int,
        degree: int,
        include_interaction: bool,
        interaction_only: bool,
        include_bias: bool,
    ) -> Iterator[Tuple[int, ...]]:
        """
        Create selection tuples of input indexes for each polynomail term

        Selection tuple iterates the input indexes present in a single term.
        For example, (x+y+1)^2 would be in iterator of the tuples:
        (), (0,), (1,), (0, 0), (0, 1), (1, 1)
        1    x     y      x^2     x*y     y^2

        Order of terms is preserved regardless of include_interation.
        """
        if not include_interaction:
            return chain(
                [()] if include_bias else [],
                (
                    exponent * (feat_idx,)
                    for exponent in range(1, degree + 1)
                    for feat_idx in range(n_features)
                ),
            )
        return PolynomialFeatures._combinations(
            n_features=n_features,
            min_degree=int(not include_bias),
            max_degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
        )

    @property
    def powers_(self):
        """
        The exponents of the polynomial as an array of shape
        (n_features_out, n_features_in), where each item is the exponent of the
        jth input variable in the ith polynomial term.
        """
        
        combinations = self._combinations(
            n_features=self.n_features_in_ + self.num_prtls*self.n_features_in_ + self.func_lib_n_out_feats,
            degree=self.poly_degree,
            include_interaction=self.include_poly_interaction,
            interaction_only=self.poly_interaction_only,
            include_bias=self.include_bias,
        )
        lngth = self.n_features_in_ + self.n_features_in_*self.num_prtls + self.func_lib_n_out_feats

        return np.vstack(
            [np.bincount(c, minlength=lngth) for c in combinations]
        )
    
    @property
    def prtls_multi_inds_(self):

        if self.sptl_dims==0:
            return np.arange(1, self.derivative_order+1, dtype=int)[:, np.newaxis]


        combs = PolynomialFeatures._combinations(
            n_features=self.sptl_dims,
            min_degree=1,
            max_degree=self.derivative_order,
            interaction_only=False,
            include_bias=False,
        )
        
        return np.vstack(
            [np.bincount(c, minlength=self.sptl_dims) for c in combs]
        )

    
    def get_library_names(self, net_out_func_names=None)->List[str]:
        """ 
            Return library fucntion names. Input arguments are...

                * net_out_func_names : list of strings objects. Length needs
                    to tbe the same as the number of neural network outputs. 
                    Optional. By default library function names are
                    "x0", "x1", ... "xn_features" is used.

            Return object is...
            
                lib_names : list of string objects, length num library 
                    functions
        """
        if net_out_func_names==None:
            net_out_func_names = [f"f{i}(...)" for i in range(self.n_features_in_)]


        if self.n_features_in_!=len(net_out_func_names):
            raise ValueError(f"Then number of net_out_func_names is less than the number of network output outputs.")
        
        base_funcs = deepcopy(net_out_func_names)
        if self.sptl_dims>=1:
            sptl_vars = ["x%d"%i for i in range(self.sptl_dims)]
        else:
            sptl_vars = ["t"]
        
        for i in range(len(base_funcs)):
            for row in self.multi_indices:
                func = base_funcs[i]+"_"
                for j, val in enumerate(row):
                    func += val*("%s" %sptl_vars[j])
                base_funcs.append(func)
        
        if self.xtra_funcs:
            base_funcs += self.xtra_funcs.get_library_names(net_out_func_names)
        

        powers = self.powers_
        lib_names = []
        # idx = 0 if self.include_bias
        for row in powers:
            inds = np.where(row)[0]
            if len(inds)==1:
                ind = inds[0]
                exp = row[ind]
                name = "(%s)^%d" % (base_funcs[ind], exp) if exp != 1 else base_funcs[ind] 
            elif len(inds)>=2:
                name = ""
                for ind, exp in zip(inds, row[inds]):
                    name += "(%s)^%d" % (base_funcs[ind], exp) if exp != 1 else "(%s)" % base_funcs[ind]
            else:
                name = "1"
            lib_names.append(name)

        return lib_names


    def fit(self, network, inpts=torch.Tensor):
        """
            Determines/Calculates the number of functions there are in the
            library of candidate functions. Input arguments are...

                * network -  A neural network that can take the 
                    inpts argument as input to the network and 
                    return 2D tensor output. Most generally
                    though this argument can be any class/object 
                    that has as a .forward() methods which can 
                    take the inpts argument and return a 2D 
                    tensor. 

                * inpts - A 2D tensor that is passed through as
                    the input to the .forward() method of the 
                    network object. 

        There is no returned object. 
        """
        if self.poly_degree<0 or not isinstance(self.poly_degree, int):
            raise ValueError("degree must be a nonnegative integer")
        if (not self.include_poly_interaction) and self.poly_interaction_only:
            raise ValueError(
                "Can't have include_interaction be False and interaction_only"
                " be True"
            )   
        
        if inpts.device!=self.device or inpts.dtype!=self.data_type:
            raise TypeError(f"The input dtype or device is not what was expected")

        outputs = network.forward(inpts).detach()
        if outputs.ndim!=2:
            raise ValueError(f"The neural network output is expected to be a 2D tensor. Doesn't have 2 dims.")

        n_features = outputs.size(1)
        self.n_features_in_ = n_features

        tot_n_derivs = n_features * self.num_prtls

        if self.xtra_funcs:
            self.xtra_funcs.fit(network, inpts)
            self.func_lib_n_out_feats = self.xtra_funcs.n_output_features_
        else:
            self.func_lib_n_out_feats = 0

        # With the number of derivatves get the total number of out features
        combinations = self._combinations(
            n_features + tot_n_derivs + self.func_lib_n_out_feats,
            self.poly_degree,
            self.include_poly_interaction,
            self.poly_interaction_only,
            self.include_bias,
        )
        self.n_output_features_ = sum(1 for _ in combinations)

        # required to generate the function names
        self.get_library_names()

        return self
    
    
    def Calc(self, network, inpts:torch.Tensor)->torch.Tensor:
        """
            Function that calculates the entire funciton library over all
            the points found in the inpt argument. Input arguments are...

                * network -  A neural network that can take the 
                    inpts argument as input to the network and 
                    return 2D tensor output. Most generally
                    though this argument can be any class/object 
                    that has as a .forward() methods which can 
                    take the inpts argument and return a 2D 
                    tensor. 

                * inpts - A 2D tensor that is passed through as
                    the input to the .forward() method of the 
                    network object. The 'points' that all the 
                    library functions are evaluated at are the 
                    rows of this 2D tensor. 
            
            The returned object is a 2D torch tensor called lib

                * lib - A 2D tensor such that the (i,j) entry is 
                    the value of the j-th candidate function
                    evaluated at the i-th point (i.e point that
                    constitutes the ith row in inpts)

        """

        if inpts.device!=self.device or inpts.dtype!=self.data_type:
            raise TypeError(f"The input dtype or device is not what was expected")
        
        if not inpts.requires_grad:
            inpts.requires_grad_(True)

        idx = 1 
        if not self.include_bias:
            lib = torch.ones((inpts.size(0), 1+ self.n_output_features_),device=self.device, dtype=self.data_type)
        else:
            lib = torch.ones((inpts.size(0), self.n_output_features_),device=self.device, dtype=self.data_type)

        lib[:, idx:self.n_features_in_+idx] = network.forward(inpts)
        idx += self.n_features_in_
        for i in range(self.n_features_in_):
            lib[:, idx+i*self.num_prtls:idx+ (i+1)*self.num_prtls] = sptl_partials(values=network.forward(inpts)[:, i:i+1], pts=inpts, order=self.derivative_order)[:, self.deriv_inds]
        
        if self.xtra_funcs:
            idx += self.num_prtls*self.n_features_in_ 
            lib[:, idx:idx + self.func_lib_n_out_feats] = self.xtra_funcs.Calc(network, inpts)
        
        combinations = self._combinations(
            self.num_prtls*self.n_features_in_ + self.n_features_in_ ,
            self.poly_degree,
            self.include_poly_interaction,
            self.poly_interaction_only,
            self.include_bias,
        )
        idx = 0 if self.include_bias else 1
        for i, comb in enumerate(combinations):
            if len(comb)<=1:
                continue
            lib[:,i+idx] = lib[..., np.array(comb)+1].prod(-1)
        
        return lib[:,-self.n_output_features_:]
    
    def Get_Lib_Complexities(self):
        """
        
        """
        pwrs = self.powers_
        complexities = np.zeros(shape=(self.n_output_features_,))
        base_complexities = np.concat(
            (
                np.ones((self.n_features_in_ + self.num_prtls*self.n_features_in_), ), 
                self.xtra_funcs.Get_Lib_Complexities()
            ),
            axis=0)
        
        for i in range(self.n_output_features_):
            cmplxty = 0
            row = pwrs[i]
            nz_expons = np.nonzero(row)[0]
            n_nonzeros = nz_expons.size

            if n_nonzeros!=0:
                for id in nz_expons:
                    if row[id]==1:
                        cmplxty = base_complexities[id]
                    else:
                        cmplxty = cmplxty + row[id]*base_complexities[id] + (row[id]-1)
            if n_nonzeros>=2:
                cmplxty += (n_nonzeros -1)

            complexities[i] = cmplxty

        if self.include_bias: 
            # since the row for the constant (bias) is a row 
            # of all zeros based off code above the value 
            # would be zero but it needs to be 1. So just 
            # set the array value to one here. 
            complexities[0]=1
        return complexities

