"""
Pythonn files that contains custom made activations function that will/can/may be used in the PINNS models
that are able to be made using the others files in this library in addition to all the activation funcs
available natively through the Pytorch library. Note that in comparision to the native Pytorch act. funcs
the more complicated act. funcs that requite learnable parameters and cannot be expressed as a combination
of existing Pytorch functions will run slower than the Pytorch act funcs since these functions are written
in Python whereas the Pytorch ones are all written in C/C++ and just called in python here with wrappers
and so will be faster. Just wanted to note that. 
"""
from typing import Union, Tuple, List
import numpy as np
import torch
import torch.nn as nn
# from .module import Module

class SinAct(nn.Module):
    r""" Custom Activation function that Applies the Sine (sin) function element-wise.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SinAct()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def __init__(self):
        super(SinAct, self).__init__()

    def forward(self, X):
        return torch.sin(X)

class CosAct(nn.Module):
    r""" Custom Activation function that Applies the Cosine (cos) function element-wise.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = CosAct()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    # def __init__(self):
    #     super(CosAct).__init__()

    def forward(self, X):
        return torch.cos(X)
    
class ArcTanAct(nn.Module):
    r""" Custom Activation function that Applies the Inverse Tangent (arctan) function 
         element-wise.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ArcTanAct()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    # def __init__(self):
    #     super(ArcTanAct).__init__()

    def forward(self, X):
        return torch.atan(X)
    
class ArcSinhAct(nn.Module):
    r""" Custom Activation function that Applies the Inverse hyperbolic sine (arsinh) function 
         element-wise.
    Tanh is defined as:

    .. math::
        \text{ArcSinhAct}(x) = \text{arcsinh}(x) = \ln\left( x +\sqrt{x^{2} + 1} \right)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = ArcSinhAct()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    # def __init__(self):
    #     super(ArcSinhAct).__init__()

    def forward(self, X):
        return torch.asinh(X)
# Copied over from the PDE-READ Paper
class Rational(nn.Module):
    def __init__(self,
                 Data_Type = torch.float32,
                 Device    = torch.device('cpu')):
        # This activation function is based on the following paper:
        # Boulle, Nicolas, Yuji Nakatsukasa, and Alex Townsend. "Rational neural
        # networks." arXiv preprint arXiv:2004.01902 (2020).

        super(Rational, self).__init__();

        # Initialize numerator and denominator coefficients to the best
        # rational function approximation to ReLU. These coefficients are listed
        # in appendix A of the paper.
        self.a = torch.nn.parameter.Parameter(
                        torch.tensor((1.1915, 1.5957, 0.5, .0218),
                                     dtype = Data_Type,
                                     device = Device));
        self.a.requires_grad_(True);

        self.b = torch.nn.parameter.Parameter(
                        torch.tensor((2.3830, 0.0, 1.0),
                                     dtype = Data_Type,
                                     device = Device));
        self.b.requires_grad_(True);

    def forward(self, X : torch.tensor):
        """ This function applies a rational function to each element of X.

        ------------------------------------------------------------------------
        Arguments:

        X: A tensor. We apply the rational function to every element of X.

        ------------------------------------------------------------------------
        Returns:

        Let N(x) = sum_{i = 0}^{3} a_i x^i and D(x) = sum_{i = 0}^{2} b_i x^i.
        Let R = N/D (ignoring points where D(x) = 0). This function applies R
        to each element of X and returns the resulting tensor. """

        # Create aliases for self.a and self.b. This makes the code cleaner.
        a = self.a;
        b = self.b;

        # Evaluate the numerator and denominator. Because of how the * and +
        # operators work, this gets applied element-wise.
        N_X = a[0] + X*(a[1] + X*(a[2] + a[3]*X));
        D_X = b[0] + X*(b[1] + b[2]*X);

        # Return R = N_X/D_X. This is also applied element-wise.
        return N_X/D_X;

class RationalAct(nn.Module):
    """
    torch.nn.Module subclass for a Rational Activation layer. Details about Rational Nueral Networks
    and rational activation functions can be found in the paper "Rational neural networks." by 
    Boulle, Nicolas, Yuji Nakatsukasa, and Alex Townsend at the following URL: 
    https://arxiv.org/pdf/2004.01902 (arXiv preprint arXiv:2004.01902 (2020))
    The input parameters are as follows:
        a = 1D numpy array or torch tensor that contains the coefficients of the
            numerator polynomial
        b = 1D numpy array or torch tensor that contains the coefficients of the
            denominator polynomial
        dtype = The data type (torch.float32 or float64) that the model that this 
            activation function is an activation function for uses as it parameter
            value typing. Needs to tbe the same as type of the model's typing
        dvc = device of that the model will be run on (i.e cpu of cuda device).
            Needs to be the same as the device that the model is on. 
    """
    def __init__(self, a:Union[np.ndarray, torch.Tensor]=torch.tensor((1.1915, 1.5957, 0.5, .0218),),
                 b:Union[np.ndarray, torch.Tensor]=torch.tensor((2.3830, 0.0, 1.0),),
                 dtyp = torch.float32,
                 dvc  = torch.device('cuda')):
        super(RationalAct, self).__init__()
        # check the a and b inputs are 1D and that a is longer than b. 
        if not(isinstance(a, (np.ndarray, torch.Tensor)) and isinstance(b, (np.ndarray, torch.Tensor))):
            raise TypeError(f"One or both of the a an b input args is not a numpy array or torch tensor.")
        if a.ndim!=1 or b.ndim !=1:
            raise ValueError(f"The input arguments a and b are to be 1D arrays/tensors. Have that a is {a.ndim} and b is {b.ndim}")
        if a.shape[0]<=b.shape[0]:
            raise ValueError("The length of the a input arg must be bigger than the length of the b input arg.\n"
                             f"Have a.shape[0]={a.shape[0]} vs b.shape[0]={b.shape[0]}")
        # Check that b is not filled with only zero values. 
        if (b==0).sum()==b.size:
            raise ZeroDivisionError(f"All the value in the b input argument are zero and so will divide by zero eventually")
        if not isinstance(dtyp, torch.dtype):
            raise TypeError(f"dtyp input arg needs to be a torch.dtype object")
        # since the torch device can be specified many ways (strings, torch.cuda(0) etc) not going to check at this moment.
        if isinstance(a, torch.Tensor):
            self.nooms = nn.parameter.Parameter(data=a.to(dtype=dtyp, device=dvc), requires_grad=True)
        else:
            self.nooms = nn.parameter.Parameter(data=torch.tensor(a.tolist,dtype=dtyp, device=dvc), requires_grad=True)
        if isinstance(b, torch.Tensor):
            self.denoms = nn.parameter.Parameter(data=b.to(dtype=dtyp, device=dvc), requires_grad=True)
        else:
            self.denoms = nn.parameter.Parameter(data=torch.tensor(b.tolist,dtype=dtyp, device=dvc), requires_grad=True)
        # so that we do not divide by zero later, get a really small number. this case machine precision/epsilon from numpy
        self.eps = np.finfo(float).eps

    def forward(self, X:torch.tensor):
        top = torch.zeros_like(X)
        bot = torch.zeros_like(X)
        for i in range(self.nooms.shape[0]):
            top += self.nooms[i]*X.pow(i)
        for i in range(self.denoms.shape[0]):
            bot += self.denoms[i]*X.pow(i)

        return top/(bot + self.eps)
    

    