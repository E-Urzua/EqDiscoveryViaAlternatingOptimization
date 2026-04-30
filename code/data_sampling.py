from typing import List, Union
from numpy import ndarray, array
from scipy.stats.qmc import Halton, Sobol, LatinHypercube
from scipy.stats import qmc
from secrets import randbits
import torch

class Rand_Col_Sampler():
    """
    
    """

    def __init__(self,
            sampler:str='halton',
            dims:int=2,
            seed:int=None,
            bounds:Union[ndarray, List[List[float]]]=[[0,1],[0,1]],
            requires_grad:bool=False,
            device:torch.device=torch.device('cpu'),
            data_type:torch.dtype=None):
        
        if sampler not in ['halton', 'sobol', 'lhc', 'torch sobol', 'rand']:
            raise ValueError(f"sampler argument needs to be one of the following halton, 'sobol, lhc, torch sobol, or rand ")

        if not isinstance(dims, int):
            raise TypeError(f"dims argument needs to be a int type object")
        if dims<1:
            raise ValueError(f"dims arguemnet needs to be a positive integer")
        self.dims = dims

        if seed and (not isinstance(seed, int)):
            raise TypeError(f"dims argument needs to be a int type object")
        if seed==None:
            seed = randbits(32)
        if seed<1:
            raise ValueError(f"dims arguemnet needs to be a positive integer")
        self.seed = seed

        bnds = array(bounds)
        if bnds.ndim!=2:
            raise ValueError(f"The bounds argument needs to be such it has two dimensions ")
        self.bounds = bnds

        if not isinstance(requires_grad, bool):
            raise TypeError(f"the requires_grad argument needs to be a boolean")
        self.grad = requires_grad

        if not isinstance(device, torch.device):
            raise TypeError(f"Input 'device' needs to be a torch.device type object")
        self.device = device
        if data_type==None:
            data_type=torch.get_default_dtype()
        if not isinstance(data_type, torch.dtype):
            raise TypeError(f"Data type argument needs to be a torch.dtype object ")
        self.data_type = data_type

        self.sampler = self._set_up(name=sampler)
        self.name = sampler
    
    def _set_up(self, name:str):

        if name =='sobol':
            sampler = Sobol(d=self.dims, optimization='random-cd', seed=self.seed)
        elif name=='halton':
            sampler = Halton(d=self.dims, optimization="random-cd", seed=self.seed)
        elif name=='lhc':
            sampler = LatinHypercube(d=self.dims,strength=1, optimization="random-cd", seed=self.seed)
        elif name=='torch sobol':
            sampler = torch.quasirandom.SobolEngine(dimension=self.dims, scramble=True, seed=self.seed)
            self.bounds = torch.from_numpy(self.bounds).to(dtype=self.data_type)
        else:
            sampler = torch.Generator(device=self.device)
            sampler.manual_seed(self.seed)
            # self.bounds = torch.from_numpy(self.bounds).to(dtype=self.data_type, device=self.device)
            self.bounds = torch.from_numpy(self.bounds).to(dtype=self.data_type,)
        
        return sampler

    def sample(self, n_pnts:int):
        """
        
        """
        # if self.name in ['halton', 'sobol', 'lhc']:
        #     pnts =  torch.from_numpy(qmc.scale(sample=self.sampler.random(n=n_pnts), l_bounds=self.bounds[:,0], u_bounds=self.bounds[:,1])).to(device=self.device, dtype=self.data_type)
        # elif self.name == 'torch sobol':
        #     pnts = self.sampler.draw(n=n_pnts, out=None, dtype=self.data_type)*(self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]
        # else:
        #     pnts = torch.rand(size=(n_pnts, self.dims), generator=self.sampler, dtype=self.data_type, device=self.device,)*(self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]
        # return pnts.requires_grad_(self.grad)

        if self.name in ['halton', 'sobol', 'lhc']:
            pnts =  qmc.scale(sample=self.sampler.random(n=n_pnts), l_bounds=self.bounds[:,0], u_bounds=self.bounds[:,1])
            # pnts =  torch.from_numpy(qmc.scale(sample=self.sampler.random(n=n_pnts), l_bounds=self.bounds[:,0], u_bounds=self.bounds[:,1])).to(device=self.device, dtype=self.data_type)
        elif self.name == 'torch sobol':
            pnts = (self.sampler.draw(n=n_pnts, out=None, dtype=self.data_type)*(self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]).numpy()
            # pnts = self.sampler.draw(n=n_pnts, out=None, dtype=self.data_type)*(self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]
        else:
            pnts = (torch.rand(size=(n_pnts, self.dims), generator=self.sampler, dtype=self.data_type,)*(self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]).numpy()
            # pnts = torch.rand(size=(n_pnts, self.dims), generator=self.sampler, dtype=self.data_type, device=self.device,)*(self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]

        return pnts
    
