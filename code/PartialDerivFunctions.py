from typing import List
import torch
import numpy as np
from torch.autograd import grad
from scipy.special import binom

def sptl_partials(values:torch.Tensor, pts:torch.Tensor, order:int=1)->torch.Tensor:
    """
    Given a function (fnc) return all the spatial partial derivatives of order 0, 1,..., order
    evaluated at the given points. Input parameters are the following:
        * func - The function for with the partials will detemined at the points given in pts
        * pts - Tensor containing the pts at which the prediction were made and the points at which
            the partials will be evaluated. The (x, y, z) coordinates of the points should be stored
            row-wise in the tensor. So pts[:, 0:1] will give all the x points and so forth.
        * order - The order up to which the partials will be determined and evaluated at
    TODO: 
        (1) INPUT Arguments checking
        (2) The order value must be between 1 and 4 currently, currently cannont do 
            5th order derivatives for more than 2 spatial dimensions (May not have to go that high - Make operators?)
    """
    # TODO - Do input args checking here for the correct types
    
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None')
        return None
    device = pts.device
    if order<=0:
        # an obvious fuck up in that we cannot take non-positive derivatieves so just return the pts
        print('ERROR: Order input arg was less than 0!\nReturning the given points')
        return pts
    # the number of dimensions is the number of columns 1D is for only none or 1 column
    try:
        dims = pts.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        dims = 0  # the given points are 0 dimensional (basically just a list)
    temp = grad(values.sum(), pts, create_graph=True,)[0]
    # NOTE: Probably can get rid of the if dims==0 block. We expect there to be 1,2 or 3 spatial and 1 time dims
    if dims == 0:
        # in this case of no columns (just a single row) store the partials row-wise
        derivs = torch.empty(size=(int(order),pts.shape[0],), device=device, dtype=pts.dtype)
        derivs[0] = temp
        f = temp.sum()
        for i in range(1, order):
            temp = grad(outputs=f, inputs=pts, create_graph=True)[0]
            derivs[i] = temp
            f = temp.sum()
        return derivs
    elif dims==1:
        derivs = torch.empty(size=(pts.shape[0], dims*int(order)), device=device, dtype=pts.dtype)
        derivs[:, 0:1] = temp
        for i in range(1, order):
            temp = grad(outputs=temp.sum(), inputs=pts, create_graph=True)[0]
            derivs[:, dims*i:dims*(i+1)] = temp
        return derivs
    elif dims==2:
        # so only one spatial dimension and one temporal dimension
        derivs = torch.empty(size=(pts.shape[0], int(order)), device=device, dtype=pts.dtype)
        derivs[:, 0:1] = temp[:, 0:1]
        for i in range(1, order):
            temp = grad(outputs=temp[:, 0:1].sum(), inputs=pts, grad_outputs=None, create_graph=True)[0]#[:,0:1]
            derivs[:, i:(i+1)] = temp[:, 0:1]
        return derivs
    # now if there are 3 or more columns is where the real messy stuff is; 
    # NOTE we know there is at least the first order
    prtl_tots = 0   # the total number of all the partials (sum of all 1st, 2nd, 3rd, and or 4th order partials)
    for i in range(1, order+1):
        prtl_tots += int(binom(i+dims-2,dims-2))
    # prtls = [int(binom(i+dims-2,dims-2)) for i in range(1, order+1)] # the number of all the partials
    derivs = torch.empty(size=(pts.shape[0], prtl_tots), device=device, dtype=pts.dtype)
    # The idea is to do the partials in a set order, the First orders then the second then the third so forth as needed
    if order>=1:
        derivs[:, 0:int(binom(1+dims-2,dims-2))] = temp[:,0:-1]
    if order>=2: 
        # to enter this block means that we have also entered the order>=1 block as an order value that is >=2 is also >=1
        indx = [2,4,5] if dims==3 else [3, 6, 8, 9] # dim == 3 means 2 spatial (x,y) dims and 1 temporal (t)
        for i in range(dims-1):
            derivs[:,indx[i]:indx[i+1]] = grad(outputs=temp[:,i:i+1], inputs=pts, grad_outputs=torch.ones_like(temp[:,i:i+1]), create_graph=True)[0][:,i:-1]
    if order>=3: 
        # to enter this block means that we have also entered the order>=2 block as a order value that is >=3 is also >=2
        indx = [5,7,9] if dims==3 else [9, 12, 15, 18, 19]
        if dims==3:
            cols = [[5,6], [7,8]]
        else:
            cols = [[9, 10, 11], [12, 15, 16], [14, 17, 18], [13]]
        func_map = lambda x: 2*x +2 if dims==3 else  int((-5/6)*x**3 + 2*x**2 + (11/6)*x + 3)
        for i in range(len(cols)):
            temp = grad(outputs=derivs[:, func_map(i):func_map(i) + 1], inputs=pts, grad_outputs=torch.ones_like(derivs[:,func_map(i):func_map(i)+1]), create_graph=True)[0][:,:-1]
            if temp==None:
                temp = torch.zeros_like(pts, device=device)
            if i == 3:
                derivs[:,cols[i]] = temp[:, 2:]
            else:
                derivs[:,cols[i]] = temp
    if order>=4:
        # to enter this block means that we have also entered the order>=3 block as a order value that is >=4 is also >=3
        indx = [9,11,12,14] if dims==3 else [19,22,24,27,29,31,34]
        if dims==3:
            cols = [[9,10], [11],[12,13]]
        else:
            cols = [[19,20,21], [22,23], [24,27], [25,29,30], [26,31], [28, 32, 33]]
        func_map = lambda x: int(0.5*x**2 + 0.5*x + 5) if dims==3 else  int((-11/120)*x**5 + (31/24)*x**4 - (155/24)*x**3 + (317/24)*x**2 - (139/20)*x + 9.05)
        for i in range(len(cols)):
            temp = grad(outputs=derivs[:, func_map(i):func_map(i) + 1], inputs=pts, grad_outputs=torch.ones_like(derivs[:,func_map(i):func_map(i)+1]), create_graph=True)[0][:,:-1]
            if temp==None:
                temp = torch.zeros_like(pts, device=device)
            if i==1:
                derivs[:,cols[i]] = temp[:, 1:]
            elif i==2 or i==4:
                derivs[:,cols[i]] = temp[:, :2]
            else:
                derivs[:,cols[i]] = temp
    return derivs

def time_prtls(values:torch.Tensor, pts:torch.Tensor, order:int=1)->torch.Tensor:
    """
    Along the line for the spatial_partial_derivatives function, deterine the temportial partials up to the given 
    order. Expect the time coordinate values to be in the last/final column of the pts matrix
    
    TODO: 
        (1) INPUT Arguments checking
    """
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None')
        return None
    device = pts.device
    if order<=0:
        # an obvious fuck in that we cannot take non-positive derivatieves so just return the predicsions
        return None
    # the number of dimensions is the number of columns 1D is for only none or 1 column
    try:
        dims = pts.shape[1] # the number of spatial dimension (i.e 1 for just x ,2 for (x,y), 3 for (x,y,z) )
    except IndexError:
        dims = 0  # the given points are 0 dimensional (basically just a list)
    if dims <2 :
        # we expect there to be at least two dimensions as there should be at least one spatial and one temporal 
        # dimension
        return None
    # so the tensor of points has two or more dimensions
    temp = grad(outputs=values.sum(), inputs=pts, create_graph=True)[0][:, -1:dims]
    derivs = torch.empty(size=(pts.shape[0], int(order)), device=device, dtype=pts.dtype)
    derivs[:, 0:1] = temp
    for i in range(1, order):
        temp = torch.autograd.grad(outputs=temp, inputs=pts, grad_outputs=torch.ones_like(temp), create_graph=True)[0][:, -1:dims]
        derivs[:, i:i+1] = temp
    return derivs

def Nth_temporal_prtl(values:torch.Tensor, pts:torch.Tensor, order_n:int=1)->torch.Tensor:
    """
    Along the line for the spatial_partial_derivatives function, deterine the temportial partials up to the given 
    order_n. Expect the time coordinate values to be in the last/final column of the pts matrix
    
    TODO: 
        (1) INPUT Arguments checking
    """
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None')
        return None
    # device = pts.device
    if order_n<=0:
        # an obvious fuck in that we cannot take non-positive derivatieves so just return the predicsions
        return None
    # the number of dimensions is the number of columns 1D is for only none or 1 column
    try:
        dims = pts.shape[1] # the number of spatial dimension (i.e 1 for just x ,2 for (x,y), 3 for (x,y,z) )
    except IndexError:
        dims = 0  # the given points are 0 dimensional (basically just a list)
    if dims <2 :
        # we expect there to be at least two dimensions as there should be at least one spatial and one temporal 
        # dimension
        return None
    # so the tensor of points has two or more dimensions
    prtl = grad(outputs=values.sum(), inputs=pts, create_graph=True)[0][:, -1:dims]
    for i in range(1, order_n):
        prtl = torch.autograd.grad(outputs=prtl, inputs=pts, grad_outputs=torch.ones_like(prtl), create_graph=True)[0][:, -1:dims]
    return prtl

def Nth_temporal_prtls(values:torch.Tensor, pts:torch.Tensor, orders:List[int])->torch.Tensor:
    """
    Along the line for the spatial_partial_derivatives function, deterine the temportial partials up to the given 
    order_n. Expect the time coordinate values to be in the last/final column of the pts matrix
    
    TODO: 
        (1) INPUT Arguments checking
    """
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None')
        return None
    # device = pts.device
    if any([ordr<0 for ordr in orders]):
        # an obvious fuck in that we cannot take non-positive derivatieves so just return the predicsions
        raise ValueError(f"One of the temporal orders is less than 0")
    # the number of dimensions is the number of columns 1D is for only none or 1 column
    if pts.ndim!=2:
        raise ValueError(f"the number of dimenions for the pts tensor argument needs to be 2")
    dims = pts.size(-1)
    if values.ndim!=2:
        raise ValueError(f"the number of dimenions for the values tensor argument needs to be 2")
    n_trgts = values.size(-1)
    prtls = torch.zeros_like(values)
    for i in  range(n_trgts):
        if orders[i]==0:
            prtls[:,i:i+1] = values[:,i:i+1]
            continue
        prtls[:,i:i+1] = grad(outputs=values[:,i:i+1].sum(), inputs=pts, create_graph=True)[0][:, -1:dims]
        for _ in range(1,orders[i]):
            prtls[:,i:i+1] = torch.autograd.grad(outputs=prtls[:,i:i+1], inputs=pts, grad_outputs=torch.ones_like(prtls[:,i:i+1]), create_graph=True)[0][:, -1:dims]
    
    return prtls


def VecTimePrtls(values:torch.Tensor, pts:torch.Tensor, orders:list)->torch.Tensor:
    """
    Along the line for the spatial_partial_derivatives function, deterime the temportial partials up to the given 
    order of a vector valued function. The order of the temporal derivative do not have to be the same for each
    component function. NOTE that we expect the time coordinate values to be in the last/final column of the 
    pts matrix/tensor and that length of orders should be the same as the number of columns in values. 
    
    TODO: 
        (1) INPUT Arguments checking
    """
    # Input args checking goes here
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None')
        return None
    try:
        pts_dims = pts.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        pts_dims = 0  # the given points are 0 dimensional (basically just a list)
    if pts_dims==1 or pts_dims==0:
        print('ERROR - YO Idiot the points array/tensor should have at least 2 columns where last is for time t')
        print('Rerun this shit again properly. For now returning None')
        return None
    try:
        val_dims = values.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        val_dims = 0  # the given points are 0 dimensional (basically just a list)
    if val_dims<1:
        print('ERROR - HEY IDIOT, the columns  in the pts or values input tensor is not correct. Values tensor')
        print('Should have as many columns as there are spatial variables. So the number of columsn that the values')
        print('tensor should have is 1 less than the number of columns in the pts input args as the input args ')
        print('tensor has one column of the temporal (t) variable \nReturning None to handle this Dumb Dumb')
        return None
    out = torch.empty_like(values)
    
    for i in range(val_dims):
        out[:, i:i+1] = grad(outputs=values[:,i:i+1].sum(),inputs=pts, create_graph=True)[0][:,-1:]
    # already done one temporal derivative to each component function now possible need to do more to them
    for i in len(orders):
        # only if the ith order value is greater then 1 do we need more temporal partials
        if orders[i]>1:
            # since we have alreay done one temporal derivative for loop is order[0]-1
            for _ in range(orders[i]-1):
                out[:, i:i+1] = grad(outputs=out[:,i:i+1].sum(),inputs=pts, create_graph=True)[0][:,-1:]
    # u1_grad = grad(outputs=values[:,0:1].sum(),inputs=pts, create_graph=True)[0]
    # u2_grad = grad(outputs=values[:,1:2].sum(),inputs=pts, create_graph=True)[0]
    # u3_grad = grad(outputs=values[:,2:3].sum(),inputs=pts, create_graph=True)[0]
    # return torch.cat((u1_grad[:,-1:], u2_grad[:,-1:], u3_grad[:,-1:]), dim=1)
    return out

def ScalarLaplacian1(values:torch.Tensor, pts:torch.Tensor)->torch.Tensor:
    """
    Return the spatial Laplacian of a function u of at most 3 spatial variables and one temporal
    variable given the values of u determined at the points (x_{i},t_{i}), (x_{i},y_{i},t_{i}), 
    or (x_{i},y_{i},z_{i},t_{i}) for 0<=i<=M and the points both stored as Mx1 and Mx2, Mx3, and 
    Mx4 tensors respectively. The returned tensor will be located on the device that both the
    points and the values are on and the tensor will have a shape of Mx1. Input args are as 
    follows:
        * values(tensor): the values of u stored as a column vector (2D tensor) evaluated as 
            the points in found in the points matrix.
        * pts(tensor): The cooridinates points where the u was evaluated at and where the 
            laplacian will be determined at
    NOTE - This function utilizes the sptl_partials functions to determined the 2nd order
    partials than then sum up the correct columns form the 1D, 2D and 3D spatial laplacian
    """
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None to handle this run time error')
        return None
    # device = pts.device
    try:
        dims = pts.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        dims = 0  # the given points are 0 dimensional (basically just a list)
    if dims == 2:
        # 1 spatial dimensions (temporal is the last column in pts)
        return sptl_partials(values=values, pts=pts, order=2)[:, -1:]
    elif dims==3:
        # 2 spatial dimensions (temporal is the last column in pts)
        return sptl_partials(values=values, pts=pts, order=2)[:, [2,4]].sum(axis=1, keepdim=True)
    elif dims==4:
        # 3 spatial dimensions (temporal is the last column in pts)
        return sptl_partials(values=values, pts=pts, order=2)[:, [3,6,8]].sum(axis=1, keepdim=True)
    else:
        print("ERROR - The pts tensor needs to be a 2D tensor where 2-4 columns with the last column for the temporal points")
        print("Returning None to handle this runtime error")
        return None
    
def ScalarLaplacian2(values:torch.Tensor, pts:torch.Tensor)->torch.Tensor:
    """
    Return the spatial Laplacian of a function u of at most 3 spatial variables and one temporal
    variable given the values of u determined at the points (x_{i},t_{i}), (x_{i},y_{i},t_{i}), 
    or (x_{i},y_{i},z_{i},t_{i}) for 0<=i<=M and the points both stored as Mx1 and Mx2, Mx3, and 
    Mx4 tensors respectively. The returned tensor will be located on the device that both the
    points and the values are on and the tensor will have a shape of Mx1. Input args are as 
    follows:
        * values(tensor): the values of u stored as a column vector (2D tensor) evaluated as 
            the points in found in the points matrix.
        * pts(tensor): The cooridinates points where the u was evaluated at and where the 
            laplacian will be determined at
    NOTE - This function utilizes the the nearly the same code as that makes the sptl_partials 
    function but should be a little bit faster as it does not have to call that function and is
    quicker for 1D spatial dimension 
    """
    # TODO - Do input args checking here for the correct types
    
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None to handle this run time error')
        return None
    device = pts.device
    order=2
    try:
        dims = pts.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        dims = 0  # the given points are 0 dimensional (basically just a list)
    temp = grad(values.sum(), pts, create_graph=True,)[0]
    # NOTE: Probably can get rid of the if dims==0 block. We expect there to be 1,2 or 3 spatial and 1 time dims
    if dims == 0:
        # in this case of no columns (just a single row) store the partials row-wise
        derivs = torch.empty(size=(int(order),pts.shape[0],), device=device, dtype=pts.dtype)
        derivs[0] = temp
        f = temp.sum()
        for i in range(1, order):
            temp = grad(outputs=f, inputs=pts, create_graph=True)[0]
            derivs[i] = temp
            f = temp.sum()
        return derivs
    elif dims==1:
        derivs = torch.empty(size=(pts.shape[0], dims*int(order)), device=device, dtype=pts.dtype)
        derivs[:, 0:1] = temp
        for i in range(1, order):
            temp = grad(outputs=temp.sum(), inputs=pts, create_graph=True)[0]
            derivs[:, dims*i:dims*(i+1)] = temp
        return derivs[:,-1:]
    elif dims==2:
        # so only one spatial dimension and one temporal dimension
        return grad(outputs=temp[:, 0:1].sum(), inputs=pts, grad_outputs=None, create_graph=True)[0][:,0:1]
    # now if there are 3 or more columns is where the real messy stuff is; 
    # NOTE we know there are at least first order
    prtl_tots = 0   # the total number of all the partials (sum of all 1st, 2nd, 3rd, and or 4th order partials)
    for i in range(1, order+1):
        prtl_tots += int(binom(i+dims-2,dims-2))
    # prtls = [int(binom(i+dims-2,dims-2)) for i in range(1, order+1)] # the number of all the partials
    derivs = torch.empty(size=(pts.shape[0], prtl_tots), device=device, dtype=pts.dtype)
    # The idea is to do the partials in a set order, the First orders then the second then the third so forth as needed
    if order>=1:
        derivs[:, 0:int(binom(1+dims-2,dims-2))] = temp[:,0:-1]
    if order>=2: 
        # to enter this block means that we have also entered the order>=1 block as an order value that is >=2 is also >=1
        indx = [2,4,5] if dims==3 else [3, 6, 8, 9] # dim == 3 means 2 spatial (x,y) dims and 1 temporal (t)
        for i in range(dims-1):
            derivs[:,indx[i]:indx[i+1]] = grad(outputs=temp[:,i:i+1], inputs=pts, grad_outputs=torch.ones_like(temp[:,i:i+1]), create_graph=True)[0][:,i:-1]
    indx.pop()
    return derivs[:, indx].sum(axis=1, keepdim=True)

def ScalarGrad(values:torch.Tensor, pts:torch.Tensor)->torch.Tensor:
    """
    Return the gradient of a function u of at most 3 spatial variables and one temporal
    variable given the values of u determined at the points (x_{i},t_{i}), (x_{i},y_{i},t_{i}), 
    or (x_{i},y_{i},z_{i},t_{i}) for 0<=i<=M and the points both stored as Mx1 and Mx2, Mx3, and 
    Mx4 tensors respectively. The returned tensor will be located on the device that both the
    points and the values are on and the tensor will of the form [u_x | u_y | u_z]. The input 
    args are as 
    follows:
        * values(tensor) - The values of u for which have been determined at the coordinates 
            found in the pts input tensor. The partials will be detemined with these values
            and need to be on the device that the points are currently on. 
        * pts - Tensor containing the coordinate points at which the evaluations of u were 
            made and the points at which the partials will be evaluated/determined at. The 
            (x, y, z) coordinates of the points should be stored row-wise in the tensor. 
            So pts[:, 0:1] will give all the x points and so forth.
    TODO: 
        (1) INPUT Arguments checking -  like cases for when pts has more than 4 dimensions and 
            other shit such as not having the temporal variable (this much later)
    """
    # TODO - Do input args checking here for the correct types
    
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None')
        return None
    # the number of dimensions is the number of columns 1D is for only none or 1 column
    try:
        dims = pts.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        dims = 0  # the given points are 0 dimensional (basically just a list)
    return grad(values.sum(), pts, create_graph=True, allow_unused=True)[0][:,0:-1]

def VectDivergance1(values:torch.Tensor, pts:torch.Tensor)->torch.Tensor:
    """
    Given a vector field u with component functions [u_{1}, u_{2}, u{3}] evaluated (values) at the points (pts)
    (x_{i}, y_{i}, z_{i}, t_{i}) for 0<=i<=N, for some natural number N, this functions returns the spatial 
    divergance of u at all the points as a column vector (N by 1 2d tensor/matrix) [u_{1}_x + u_{2}_y + u_{3}_z].
    NOTE: This functions can for any number of spatial dimenions but currently (08/110/2022) must have the 
        temporal dimension. Should be a tiny/little bit slower than the other function seen below (VectDivergence2)
    Input arguments are as follows:
        * values (tensor) - 2D N by 1 2 or 3 tensor that contains the values of the component functions of u,
            u_{1}, u_{2} and u_{3} evaluated aat all the coordinates found in the pts tensor as columns the of 
            the values tensors (I.E 
            [u_{1}(x_{i}, y_{i},z_{i}, t_{i})| u_{2}(x_{i}, y_{i},z_{i}, t_{i})| u_{3}(x_{i}, y_{i},z_{i}, t_{i})])
        * pts (tensor) - 2D N by 2, 3, or 4 tensor that contains the coordinate points (x_{i}, y_{i}, z_{i}, t_{i})
            (in this order - temporal variable always is the last column) where the vector field uhas been 
            determined/evaluated at and whose values can be found in the values tensors. The points should look 
            something like this: [x_{i} | y_{i} | z_{i} | t_{i}] and similarly so for less spatial dimensions
    TODO (08/11/2022):
            (1) Input arguments checking as always.
            (2) If possible handle the case where some dumbass fucks up and does not give the temporal variable
    """
    # Input args checking goes here
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None')
        return None
    try:
        pts_dims = pts.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        pts_dims = 0  # the given points are 0 dimensional (basically just a list)
    if pts_dims==1 or pts_dims==0:
        print('ERROR - YO Idiot vector divergence needs the pts input argument to have at least 2 columns (1st x then t)')
        print('Rerun this shit again properly. For now returning None')
        return None
    try:
        val_dims = values.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        val_dims = 0  # the given points are 0 dimensional (basically just a list)
    if val_dims+1!=pts_dims:
        print('ERROR - HEY IDIOT, the columns are in the pts or values input tensor is not correct. Values tensor')
        print('Should have as many columns as there are spatial variables. So the number of columsn that the values')
        print('tensor should have is 1 less than the number of columns in the pts input args as the input args ')
        print('tensor has one column of the temporal (t) variable \nReturning None to handle this Dumb Dumb')
        return None
    # Alright so we should have be find taking the divergence now
    prtls = torch.empty_like(values)
    for i in range(val_dims):
        prtls[:,i:i+1] = grad(outputs=values[:,i:i+1].sum(), inputs=pts, create_graph=True, allow_unused=True)[0][:,i:i+1]
    return prtls.sum(axis=1, keepdim=True)

def VectDivergance2(values:torch.Tensor, pts:torch.Tensor)->torch.Tensor:
    """
    Given a vector field u with component functions [u_{1}, u_{2}, u{3}] evaluated (values) at the points (pts)
    (x_{i}, y_{i}, z_{i}, t_{i}) for 0<=i<=N, for some natural number N, this functions returns the spatial 
    divergance of u at all the points as a column vector (N by 1 2d tensor/matrix) [u_{1}_x + u_{2}_y + u_{3}_z].
    NOTE: This functions can only work for up to 3 spatial dimenions with one temporal dimension but should be a 
        tiny/little bit faster than the other function seen above (VectDivergence1)
    Input arguments are as follows:
        * values (tensor) - 2D N by 1 2 or 3 tensor that contains the values of the component functions of u,
            u_{1}, u_{2} and u_{3} evaluated aat all the coordinates found in the pts tensor as columns the of 
            the values tensors (I.E 
            [u_{1}(x_{i}, y_{i},z_{i}, t_{i})| u_{2}(x_{i}, y_{i},z_{i}, t_{i})| u_{3}(x_{i}, y_{i},z_{i}, t_{i})])
        * pts (tensor) - 2D N by 2, 3, or 4 tensor that contains the coordinate points (x_{i}, y_{i}, z_{i}, t_{i})
            (in this order - temporal variable always is the last column) where the vector field uhas been 
            determined/evaluated at and whose values can be found in the values tensors. The points should look 
            something like this: [x_{i} | y_{i} | z_{i} | t_{i}] and similarly so for less spatial dimensions
    TODO (08/11/2022):
            (1) Input arguments checking as always.
            (2) If possible handle the case where some dumbass fucks up and does not give the temporal variable
    """
    # Input args checking goes here
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None')
        return None
    try:
        pts_dims = pts.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        pts_dims = 0  # the given points are 0 dimensional (basically just a list)
    if pts_dims==1 or pts_dims==0:
        print('ERROR - YO Idiot vector divergence needs the pts input argument to have at least 2 columns (1st x then t)')
        print('Rerun this shit again properly. For now returning None')
        return None
    try:
        val_dims = values.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        val_dims = 0  # the given points are 0 dimensional (basically just a list)
    if val_dims+1!=pts_dims:
        print('ERROR - HEY IDIOT, the columns are in the pts or values input tensor is not correct. Values tensor')
        print('Should have as many columns as there are spatial variables. So the number of columsn that the values')
        print('tensor should have is 1 less than the number of columns in the pts input args as the input args ')
        print('tensor has one column of the temporal (t) variable \nReturning None to handle this Dumb Dumb')
        return None
    # Alright so we should have be find taking the divergence now
    prtls = torch.empty_like(values)
    if val_dims==1:
        prtls[:,0:1] = grad(outputs=values[:,0:1].sum(), inputs=pts, create_graph=True, allow_unused=True)[0][:,0:1]
    elif val_dims==2:
        prtls[:,0:1] = grad(outputs=values[:,0:1].sum(), inputs=pts, create_graph=True, allow_unused=True)[0][:,0:1]
        prtls[:,1:2] = grad(outputs=values[:,1:2].sum(), inputs=pts, create_graph=True, allow_unused=True)[0][:,1:2]
    elif val_dims==3:
        prtls[:,0:1] = grad(outputs=values[:,0:1].sum(), inputs=pts, create_graph=True, allow_unused=True)[0][:,0:1]
        prtls[:,1:2] = grad(outputs=values[:,1:2].sum(), inputs=pts, create_graph=True, allow_unused=True)[0][:,1:2]
        prtls[:,2:3] = grad(outputs=values[:,2:3].sum(), inputs=pts, create_graph=True, allow_unused=True)[0][:,2:3]
    else:
        print('ERROR -  This function is only supposed to be used for up to and included 3 spatial dimension not what you gave')
        print('To handle this run time error will just return a None')
        return None
    return prtls.sum(axis=1, keepdim=True)

def VectCurl2D1(values:torch.Tensor, pts:torch.Tensor)->torch.Tensor:
    """
    Given a 2D vector field u with component functions [u_{1} | u_{2}] evaluated (values) at the points (pts)
    (x_{i}, y_{i}, t_{i}) for 0<=i<=N, for some natural number N, this functions returns the spatial 2D curl
    of u at all the points as a single column vector (N by 1 2d tensor/matrix) [u_{2}_x + u_{1}_x]. This is 
    simply done by taking the gradient of each component function returning difference between the two correct
    partials.
    NOTE: This functions can for any number of spatial dimenions but currently (08/110/2022) must have the 
        temporal dimension. Should be a tiny/little bit slower than the other function seen below (VectCurl2D2)
    Input arguments are as follows:
        * values (tensor) - 2D N by 2 tensor that contains the values of the component functions of u,
            u_{1} and u_{2} evaluated at all the coordinates found in the pts tensor as a column the of 
            the values tensors (I.E 
            [u_{1}(x_{i}, y_{i},t_{i})| u_{2}(x_{i}, y_{i},t_{i})])
        * pts (tensor) - 2D N by 3 tensor that contains the coordinate points (x_{i}, y_{i},t_{i})
            (in this order - temporal variable always is the last column) where the vector field u 
            has been determined/evaluated at and whose values can be found in the values tensors. 
            The points should look like this: [x_{i} | y_{i} | t_{i}].
    TODO (09/11/2022):
            (1) Input arguments checking as always.
            (2) If possible handle the case where some dumbass fucks up and does not give the temporal variable
    """
    # Input args checking goes here
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None')
        return None
    try:
        pts_dims = pts.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        pts_dims = 0  # the given points are 0 dimensional (basically just a list)
    if pts_dims==1 or pts_dims==0:
        print('ERROR - YO Idiot 2d vector curl needs the pts input argument to have at least 3 columns (x, y then t)')
        print('Rerun this shit again properly. For now returning None')
        return None
    try:
        val_dims = values.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        val_dims = 0  # the given points are 0 dimensional (basically just a list) - could also just error and return None
    if val_dims+1!=pts_dims:
        print('ERROR - HEY IDIOT, the columns  in the pts or values input tensor is not correct. Values tensor')
        print('Should have as many columns as there are spatial variables. So the number of columsn that the values')
        print('tensor should have is 1 less than the number of columns in the pts input args as the input args ')
        print('tensor has one column of the temporal (t) variable \nReturning None to handle this Dumb Dumb')
        return None
    # u1_grad = ScalarGrad(values[:,0:1], pts)
    # u2_grad = ScalarGrad(values[:,1:2], pts)
    # return u2_grad[:,1:2] - u1_grad[:, 0:1]
    return ScalarGrad(values[:,1:2], pts)[:,1:2] - ScalarGrad(values[:,0:1], pts)[:, 0:1]

def VectCurl2D2(values:torch.Tensor, pts:torch.Tensor)->torch.Tensor:
    """
    Given a 2D vector field u with component functions [u_{1} | u_{2}] evaluated (values) at the points (pts)
    (x_{i}, y_{i}, t_{i}) for 0<=i<=N, for some natural number N, this functions returns the spatial 2D curl
    of u at all the points as a single column vector (N by 1 2d tensor/matrix) [u_{2}_x + u_{1}_x]. This is 
    simply done by taking the gradient of each component function returning difference between the two correct
    partials.
    NOTE: This functions can only work for up to 3 spatial dimenions with one temporal dimension but should be a 
        tiny/little bit faster than the other function seen above (VectCurl2D1)
    Input arguments are as follows:
        * values (tensor) - 2D N by 2 tensor that contains the values of the component functions of u,
            u_{1} and u_{2} evaluated at all the coordinates found in the pts tensor as a column the of 
            the values tensors (I.E 
            [u_{1}(x_{i}, y_{i},t_{i})| u_{2}(x_{i}, y_{i},t_{i})])
        * pts (tensor) - 2D N by 3 tensor that contains the coordinate points (x_{i}, y_{i},t_{i})
            (in this order - temporal variable always is the last column) where the vector field u 
            has been determined/evaluated at and whose values can be found in the values tensors. 
            The points should look like this: [x_{i} | y_{i} | t_{i}].
    TODO (09/11/2022):
            (1) Input arguments checking as always.
            (2) If possible handle the case where some dumbass fucks up and does not give the temporal variable
    """
    # Input args checking goes here
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None')
        return None
    try:
        pts_dims = pts.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        pts_dims = 0  # the given points are 0 dimensional (basically just a list)
    if pts_dims==1 or pts_dims==0:
        print('ERROR - YO Id2d vector cudlhneeds e pts input argument to have at least 2 3olumns (1st, yhen t)')
        print('Rerun this shit again properly. For now returning None')
        return None
    try:
        val_dims = values.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        val_dims = 0  # the given points are 0 dimensional (basically just a list) - could also just error and return None
    if val_dims+1!=pts_dims:
        print('ERROR - HEY IDIOT, the columns  in the pts or values input tensor is not correct. Values tensor')
        print('Should have as many columns as there are spatial variables. So the number of columsn that the values')
        print('tensor should have is 1 less than the number of columns in the pts input args as the input args ')
        print('tensor has one column of the temporal (t) variable \nReturning None to handle this Dumb Dumb')
        return None
    return grad(outputs=values[:,1:2].sum(), inputs=pts, create_graph=True)[0][:,0:1] - grad(outputs=values[:,0:1].sum(), inputs=pts, create_graph=True)[0][:,1:2]

def DxF1_M_DyF2A(values:torch.Tensor, pts:torch.Tensor)->torch.Tensor:
    """
    Given a 2D vector field u with component functions [u_{1}, u_{2}] evaluated (values) at the points (pts)
    (x_{i}, y_{i}, t_{i}) for 0<=i<=N, for some natural number N, this functions returns  the difference 
    between u_{1}_x and u_{2}_y at all the points as a column vector (N by 1 2d tensor/matrix) [u_{1}_x - u_{2}_y ]
    NOTE: This functions can only work for up to 2 spatial dimenions with one temporal dimension but should be a 
        tiny/little bit slower than the other function seen above (DxF1_M_DyF2B)
    Input arguments are as follows:
        * values (tensor) - 2D N by 2 tensor that contains the values of the component functions of u,
            u_{1} and u_{2} evaluated at all the coordinates found in the pts tensor as columns the of 
            the values tensors (I.E [u_{1}(x_{i}, y_{i}, t_{i})| u_{2}(x_{i}, y_{i}, t_{i})] )
        * pts (tensor) - 2D N by 3 tensor that contains the coordinate points (x_{i}, y_{i}, t_{i})
            (in this order - temporal variable always is the last column) where the vector field u has 
            been determined/evaluated at and whose values can be found in the values tensors. The pts 
            tensor should look something like this: [x_{i} | y_{i} | t_{i}] 
    TODO (09/11/2022):
            (1) Input arguments checking as always.
            (2) If possible handle the case where some dumbass fucks up and does not give the temporal variable
    """
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None')
        return None
    try:
        pts_dims = pts.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        pts_dims = 0  # the given points are 0 dimensional (basically just a list)
    if pts_dims!=3:
        print('ERROR - YO Idiot this function needs the pts input argument to have at least 3 columns (x, y then t in this order)')
        print('Rerun this shit again properly. For now returning None')
        return None
    try:
        val_dims = values.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        val_dims = 0  # the given points are 0 dimensional (basically just a list)
    if val_dims+1!=pts_dims:
        print('ERROR - HEY IDIOT, the columns in the pts or values input tensor is not correct. Values tensor')
        print('Should have as many columns as there are spatial variables. So the number of columsn that the values')
        print('tensor should have is 1 less than the number of columns in the pts input args as the input args ')
        print('tensor has one column of the temporal (t) variable \nReturning None to handle this Dumb Dumb')
        return None
    return ScalarGrad(values[:,0:1], pts)[:,0:1] - ScalarGrad(values[:,1:2], pts)[:,1:2]
    
def DxF1_M_DyF2B(values:torch.Tensor, pts:torch.Tensor)->torch.Tensor:
    """
    Given a 2D vector field u with component functions [u_{1}, u_{2}] evaluated (values) at the points (pts)
    (x_{i}, y_{i}, t_{i}) for 0<=i<=N, for some natural number N, this functions returns  the difference 
    between u_{1}_x and u_{2}_y at all the points as a column vector (N by 1 2d tensor/matrix) [u_{1}_x - u_{2}_y ]
    NOTE: This functions can only work for up to 2 spatial dimenions with one temporal dimension but should be a 
        tiny/little bit faster than the other function seen above (DxF1_M_DyF2A)
    Input arguments are as follows:
        * values (tensor) - 2D N by 2 tensor that contains the values of the component functions of u,
            u_{1} and u_{2} evaluated at all the coordinates found in the pts tensor as columns the of 
            the values tensors (I.E [u_{1}(x_{i}, y_{i}, t_{i})| u_{2}(x_{i}, y_{i}, t_{i})] )
        * pts (tensor) - 2D N by 3 tensor that contains the coordinate points (x_{i}, y_{i}, t_{i})
            (in this order - temporal variable always is the last column) where the vector field u has 
            been determined/evaluated at and whose values can be found in the values tensors. The pts 
            tensor should look something like this: [x_{i} | y_{i} | t_{i}] 
    TODO (09/11/2022):
            (1) Input arguments checking as always.
            (2) If possible handle the case where some dumbass fucks up and does not give the temporal variable
    """
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None')
        return None
    try:
        pts_dims = pts.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        pts_dims = 0  # the given points are 0 dimensional (basically just a list)
    if pts_dims!=3:
        print('ERROR - YO Idiot this function needs the pts input argument to have at least 3 columns (x, y then t in this order)')
        print('Rerun this shit again properly. For now returning None')
        return None
    try:
        val_dims = values.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        val_dims = 0  # the given points are 0 dimensional (basically just a list)
    if val_dims+1!=pts_dims:
        print('ERROR - HEY IDIOT, the columns in the pts or values input tensor is not correct. Values tensor')
        print('Should have as many columns as there are spatial variables. So the number of columsn that the values')
        print('tensor should have is 1 less than the number of columns in the pts input args as the input args ')
        print('tensor has one column of the temporal (t) variable \nReturning None to handle this Dumb Dumb')
        return None
    return grad(outputs=values[:,0:1].sum(), inputs=pts, create_graph=True)[0][:,0:1] - grad(outputs=values[:,1:2].sum(), inputs=pts, create_graph=True)[0][:,1:2]

def DyF1_P_DxF2A(values:torch.Tensor, pts:torch.Tensor)->torch.Tensor:
    """
    Given a 2D vector field u with component functions [u_{1}, u_{2}] evaluated (values) at the points (pts)
    (x_{i}, y_{i}, t_{i}) for 0<=i<=N, for some natural number N, this functions returns  the difference 
    between u_{1}_y and u_{2}_x at all the points as a column vector (N by 1 2d tensor/matrix) [u_{1}_y - u_{2}_x]
    NOTE: This functions can only work for up to 2 spatial dimenions with one temporal dimension but should be a 
        tiny/little bit faster than the other function seen above (DyF1_P_DxF2B)
    Input arguments are as follows:
        * values (tensor) - 2D N by 2 tensor that contains the values of the component functions of u,
            u_{1} and u_{2} evaluated at all the coordinates found in the pts tensor as columns the of 
            the values tensors (I.E [u_{1}(x_{i}, y_{i}, t_{i})| u_{2}(x_{i}, y_{i}, t_{i})] )
        * pts (tensor) - 2D N by 3 tensor that contains the coordinate points (x_{i}, y_{i}, t_{i})
            (in this order - temporal variable always is the last column) where the vector field u has 
            been determined/evaluated at and whose values can be found in the values tensors. The pts 
            tensor should look something like this: [x_{i} | y_{i} | t_{i}] 
    TODO (09/11/2022):
            (1) Input arguments checking as always.
            (2) If possible handle the case where some dumbass fucks up and does not give the temporal variable
    """
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None')
        return None
    try:
        pts_dims = pts.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        pts_dims = 0  # the given points are 0 dimensional (basically just a list)
    if pts_dims!=3:
        print('ERROR - YO Idiot this function needs the pts input argument to have at least 3 columns (x, y then t in this order)')
        print('Rerun this shit again properly. For now returning None')
        return None
    try:
        val_dims = values.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        val_dims = 0  # the given points are 0 dimensional (basically just a list)
    if val_dims+1!=pts_dims:
        print('ERROR - HEY IDIOT, the columns in the pts or values input tensor is not correct. Values tensor')
        print('Should have as many columns as there are spatial variables. So the number of columsn that the values')
        print('tensor should have is 1 less than the number of columns in the pts input args as the input args ')
        print('tensor has one column of the temporal (t) variable \nReturning None to handle this Dumb Dumb')
        return None
    return ScalarGrad(values[:,0:1], pts)[:,1:2] + ScalarGrad(values[:,1:2], pts)[:,0:1]
    
def DyF1_P_DxF2B(values:torch.Tensor, pts:torch.Tensor)->torch.Tensor:
    """
    Given a 2D vector field u with component functions [u_{1}, u_{2}] evaluated (values) at the points (pts)
    (x_{i}, y_{i}, t_{i}) for 0<=i<=N, for some natural number N, this functions returns  the difference 
    between u_{1}_y and u_{2}_x at all the points as a column vector (N by 1 2d tensor/matrix) [u_{1}_y - u_{2}_x]
    NOTE: This functions can only work for up to 2 spatial dimenions with one temporal dimension but should be a 
        tiny/little bit faster than the other function seen above (DyF1_P_DxF2A)
    Input arguments are as follows:
        * values (tensor) - 2D N by 2 tensor that contains the values of the component functions of u,
            u_{1} and u_{2} evaluated at all the coordinates found in the pts tensor as columns the of 
            the values tensors (I.E [u_{1}(x_{i}, y_{i}, t_{i})| u_{2}(x_{i}, y_{i}, t_{i})] )
        * pts (tensor) - 2D N by 3 tensor that contains the coordinate points (x_{i}, y_{i}, t_{i})
            (in this order - temporal variable always is the last column) where the vector field u has 
            been determined/evaluated at and whose values can be found in the values tensors. The pts 
            tensor should look something like this: [x_{i} | y_{i} | t_{i}] 
    TODO (09/11/2022):
            (1) Input arguments checking as always.
            (2) If possible handle the case where some dumbass fucks up and does not give the temporal variable
    """
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None')
        return None
    try:
        pts_dims = pts.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        pts_dims = 0  # the given points are 0 dimensional (basically just a list)
    if pts_dims!=3:
        print('ERROR - YO Idiot this function needs the pts input argument to have at least 3 columns (x, y then t in this order)')
        print('Rerun this shit again properly. For now returning None')
        return None
    try:
        val_dims = values.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        val_dims = 0  # the given points are 0 dimensional (basically just a list)
    if val_dims+1!=pts_dims:
        print('ERROR - HEY IDIOT, the columns in the pts or values input tensor is not correct. Values tensor')
        print('Should have as many columns as there are spatial variables. So the number of columsn that the values')
        print('tensor should have is 1 less than the number of columns in the pts input args as the input args ')
        print('tensor has one column of the temporal (t) variable \nReturning None to handle this Dumb Dumb')
        return None
    return grad(outputs=values[:,0:1].sum(), inputs=pts, create_graph=True)[0][:,1:2] - grad(outputs=values[:,1:2].sum(), inputs=pts, create_graph=True)[0][:,0:1]

def VectCurl3D1(values:torch.Tensor, pts:torch.Tensor)->torch.Tensor:
    """
    Given a 3D vector field u with component functions [u_{1} | u_{2} | u_{3}] evaluated (values) at the points (pts)
    (x_{i},y_{i},z_{i},t_{i}) for 0<=i<=N, for some natural number N, this functions returns the spatial 3D curl
    of u at all the points as a N by 3 2D tensor/matrix [u_{3}_y - u_{2}_z, u_{1}_z - u_{3}_x, u_{2}_x + u_{1}_x]. 
    This is simply done by taking the gradient of each component function and concatenating the difference between 
    the two correct partials of the u_{1}, u_{2} and u_{3} into a N by 3 tensor
    NOTE: This functions can wok for 3 spatial dimenions but currently (10/110/2022) must have the 
        temporal dimension. Should be a tiny/little bit slower than the other function seen below (VectCurl3D2)
    Input arguments are as follows:
        * values (tensor) - 2D N by 3 tensor that contains the values of the component functions of u,
            u_{1}, u_{2} and u_{3} evaluated at all the coordinates found in the pts tensor as a column 
            the of the values tensors (I.E 
            [u_{1}(x_{i},y_{i},z_{i},t_{i}) | u_{2}(x_{i},y_{i},z_{i},t_{i})| u_{3}(x_{i},y_{i},z_{i},t_{i})])
        * pts (tensor) - 2D N by 4 tensor that contains the coordinate points (x_{i}, y_{i}, z_{i}, t_{i})
            (in this order - temporal variable always is the last column) where the vector field u 
            has been determined/evaluated at and whose values can be found in the values tensors. 
            The points should look like this: [x_{i} | y_{i} | z_{i} | t_{i}].
    TODO (10/11/2022):
            (1) Input arguments checking as always.
            (2) If possible handle the case where some dumbass fucks up and does not give the temporal variable
    """
    # Input args checking goes here
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None')
        return None
    try:
        pts_dims = pts.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        pts_dims = 0  # the given points are 0 dimensional (basically just a list)
    if pts_dims==1 or pts_dims==0:
        print('ERROR - YO Idiot 2d vector curl needs the pts input argument to have at least 3 columns (x, y then t)')
        print('Rerun this shit again properly. For now returning None')
        return None
    try:
        val_dims = values.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        val_dims = 0  # the given points are 0 dimensional (basically just a list)
    if val_dims+1!=pts_dims:
        print('ERROR - HEY IDIOT, the columns  in the pts or values input tensor is not correct. Values tensor')
        print('Should have as many columns as there are spatial variables. So the number of columsn that the values')
        print('tensor should have is 1 less than the number of columns in the pts input args as the input args ')
        print('tensor has one column of the temporal (t) variable \nReturning None to handle this Dumb Dumb')
        return None
    u1_grad = ScalarGrad(values[:,0:1], pts)
    u2_grad = ScalarGrad(values[:,1:2], pts)
    u3_grad = ScalarGrad(values[:,2:3], pts)
#     curl = torch.cat((u3_grad[:,1:2]-u2_grad[:,2:3], u1_grad[:,2:3]-u3_grad[:,0:1], u2_grad[:, 0:1]-u1_grad[:,1:2]), dim=1)
    return torch.cat((u3_grad[:,1:2]-u2_grad[:,2:3], u1_grad[:,2:3]-u3_grad[:,0:1], u2_grad[:, 0:1]-u1_grad[:,1:2]), dim=1)

def VectCurl3D2(values:torch.Tensor, pts:torch.Tensor)->torch.Tensor:
    """
    Given a 3D vector field u with component functions [u_{1} | u_{2} | u_{3}] evaluated (values) at the points (pts)
    (x_{i},y_{i},z_{i},t_{i}) for 0<=i<=N, for some natural number N, this functions returns the spatial 3D curl
    of u at all the points as a N by 3 2D tensor/matrix [u_{3}_y - u_{2}_z, u_{1}_z - u_{3}_x, u_{2}_x + u_{1}_x]. 
    This is simply done by taking the gradient of each component function and concatenating the difference between 
    the two correct partials of the u_{1}, u_{2} and u_{3} into a N by 3 tensor
    NOTE: This functions can work only for 3 spatial dimenions with one temporal dimension but should be a 
        tiny/little bit faster than the other function seen above (VectCurl3D1)
    Input arguments are as follows:
        * values (tensor) - 2D N by 3 tensor that contains the values of the component functions of u,
            u_{1}, u_{2} and u_{3} evaluated at all the coordinates found in the pts tensor as a column 
            the of the values tensors (I.E 
            [u_{1}(x_{i},y_{i},z_{i},t_{i}) | u_{2}(x_{i},y_{i},z_{i},t_{i})| u_{3}(x_{i},y_{i},z_{i},t_{i})])
        * pts (tensor) - 2D N by 4 tensor that contains the coordinate points (x_{i}, y_{i}, z_{i}, t_{i})
            (in this order - temporal variable always is the last column) where the vector field u 
            has been determined/evaluated at and whose values can be found in the values tensors. 
            The points should look like this: [x_{i} | y_{i} | z_{i} | t_{i}].
    TODO (10/11/2022):
            (1) Input arguments checking as always.
            (2) If possible handle the case where some dumbass fucks up and does not give the temporal variable
    """
    # Input args checking goes here
    if values.device!=pts.device:
        print('ERROR: The passed values and pts tensors are not on the same device for whatever reason')
        print('Will return a None')
        return None
    try:
        pts_dims = pts.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        pts_dims = 0  # the given points are 0 dimensional (basically just a list)
    if pts_dims==1 or pts_dims==0:
        print('ERROR - YO Idiot 2d vector cudlhneeds e pts input argument to have at least 2 3olumns (1st, yhen t)')
        print('Rerun this shit again properly. For now returning None')
        return None
    try:
        val_dims = values.shape[1] # the number of spatial dimension (i.e 1 for just t ,2 for (x,t), 3 for (x,y,t) etc.)
    except IndexError:
        val_dims = 0  # the given points are 0 dimensional (basically just a list)
    if val_dims+1!=pts_dims:
        print('ERROR - HEY IDIOT, the columns  in the pts or values input tensor is not correct. Values tensor')
        print('Should have as many columns as there are spatial variables. So the number of columsn that the values')
        print('tensor should have is 1 less than the number of columns in the pts input args as the input args ')
        print('tensor has one column of the temporal (t) variable \nReturning None to handle this Dumb Dumb')
        return None
    u1_grad = grad(outputs=values[:,0:1].sum(),inputs=pts, create_graph=True)[0]
    u2_grad = grad(outputs=values[:,1:2].sum(),inputs=pts, create_graph=True)[0]
    u3_grad = grad(outputs=values[:,2:3].sum(),inputs=pts, create_graph=True)[0]
    return torch.cat((u3_grad[:,1:2]-u2_grad[:,2:3], u1_grad[:,2:3]-u3_grad[:,0:1], u2_grad[:, 0:1]-u1_grad[:,1:2]), dim=1)

#### EXTRA Non-Partial Derivative Related functions

def tensor_powers(inpt:torch.Tensor, powers:int=0)->torch.Tensor:
    """
    Return a tensor/array that contains the values of input**0, input**1 ... input**powers
    stored column wise
    """
    device=inpt.device
    pwrs = torch.tensor([i for i in range(powers+1)], device=device)
    return inpt ** pwrs

def pairwise_prds(mat1:torch.Tensor, mat2:torch.Tensor)->torch.Tensor:
    """
    Given two 2D arrays/matrices (np arrays or torch arrays or whatever is like that) creates and returns a matrix that 
    contains all the pairwise productes between the columns in mat2 and mat2. Thus the matrices should have the same 
    number of rows
    TODO - 
        (1) Input Args checking
        (2) Add the capability for both numpy arrays and torch tensors
    """
    if mat1.device!=mat2.device:
        print('ERROR: The passed mat1 and mat2 tensors are not on the same device for whatever reason')
        print('Will return a None')
        return None
    device = mat1.device
    if mat1.shape[0] != mat2.shape[0]: return np.inf
    try:
        col1 = mat1.shape[1]
    except IndexError:
        return None
    try:
        col2 = mat2.shape[1]
    except IndexError:
        return None
    n = col1*col2
    if isinstance(mat1, torch.Tensor) and isinstance(mat2, torch.Tensor):
        prods = torch.empty(size=(mat1.shape[0], n),  device=device)
    elif isinstance(mat1, np.ndarray) and isinstance(mat2, np.ndarray):
        prods = np.empty(shape=(mat1.shape[0], n),)
    for i in range(col1):
        for j in range(col2):
            prods[:, i*col2+j:i*col2+j+1] = mat1[:,i:i+1]*mat2[:,j:j+1]
    return prods