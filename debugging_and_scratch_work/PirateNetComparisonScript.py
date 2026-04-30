import os
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
from time import perf_counter
import scipy.io as sio
import matplotlib.pyplot as plt
import torch

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

class KlnGrd_Net(torch.nn.Module):

    def __init__(self, 
                in_dim:int,
                out_dim:int,
                n_hid_lyr:int,
                nodes_per_lyr:int,
                scale:float,
                embd_dim:int):

        super().__init__()
        # kernel = torch.distributions.normal.Normal(loc=0.0, scale=scale).sample((in_dim, embd_dim//2))
        # self.register_buffer(name='kernel', tensor=kernel)

        # network needs to match what was used in the normal plain old MLP results 
        # lyrs_lst = [torch.nn.Linear(in_features=embd_dim, out_features=nodes_per_lyr, bias=True)]

        self.mduls = torch.nn.ModuleDict()
        self.mduls[f"U"] = torch.nn.Linear(in_features=in_dim, out_features=nodes_per_lyr, bias=True)
        torch.nn.init.xavier_normal_(self.mduls[f"U"].weight.data, gain=1.41)
        self.mduls['U_Act_Func'] = torch.nn.Tanh()

        self.mduls[f"V"] = torch.nn.Linear(in_features=in_dim, out_features=nodes_per_lyr, bias=True)
        torch.nn.init.xavier_normal_(self.mduls[f"V"].weight.data, gain=1.41)
        self.mduls['V_Act_Func'] = torch.nn.Tanh()
        
        for i in range(n_hid_lyr):

            self.mduls[f"PiBtle{i}InLyr"] = torch.nn.Linear(in_features=in_dim, out_features=nodes_per_lyr, bias=True)
            torch.nn.init.xavier_normal_(self.mduls[f"PiBtle{i}InLyr"].weight.data, gain=1.41)
            self.mduls[f"PiBtle{i}MidLyr"] = torch.nn.Linear(in_features=nodes_per_lyr, out_features=nodes_per_lyr, bias=True)
            torch.nn.init.xavier_normal_(self.mduls[f"PiBtle{i}MidLyr"].weight.data, gain=1.41)
            self.mduls[f"PiBtle{i}OutLyr"] = torch.nn.Linear(in_features=nodes_per_lyr, out_features=in_dim, bias=True)
            torch.nn.init.xavier_normal_(self.mduls[f"PiBtle{i}OutLyr"].weight.data, gain=1.41)
            self.mduls[f"PiBtle{i}Alpha"] = torch.nn.Linear(in_features=1, out_features=1, bias=False)
            # self.mduls[f"PiBtle{i}Alpha"].weight.data = torch.tensor([[0.0]])
            self.mduls.register_parameter(name=f"alpha{i}", param=torch.nn.Parameter(data=torch.zeros((1,),), requires_grad=True))

        
        self.mduls['OutLayer'] = torch.nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        torch.nn.init.xavier_normal_(self.mduls['OutLayer'].weight.data, gain=1.41)

        self.num_layers = n_hid_lyr

        
    def forward(self, x:torch.Tensor):
        """
            B.C are that U(-1,t)=U(1,t)=0 for all t. So force this, we multiply the 
            output of the network by (x^2 - 1)=(x-1)(x+1)
        """
        # y = torch.cat((torch.cos(torch.matmul(x, self.kernel)),torch.sin(torch.matmul(x, self.kernel)) ), dim=1)
        bc_cond_tsnr = (x[:,0:1].pow(2) - 1)

        U = self.mduls['U_Act_Func'](self.mduls[f"U"](x))
        V = self.mduls['V_Act_Func'](self.mduls[f"V"](x))
        # the pi- bottle neck forward passes. 
        for i in range(self.num_layers):
            copy = x
            x = torch.nn.Tanh()(self.mduls[f"PiBtle{i}InLyr"](x))
            x = x*U + (1-x)*V
            x = torch.nn.Tanh()(self.mduls[f"PiBtle{i}MidLyr"](x))
            x = x*U + (1-x)*V
            x = torch.nn.Tanh()(self.mduls[f"PiBtle{i}OutLyr"](x))
            x = self.mduls.__dict__["_parameters"][f"alpha{i}"]*x + (1-self.mduls.__dict__["_parameters"][f"alpha{i}"])*copy
            # x = self.mduls[f"PiBtle{i}Alpha"](x) - self.mduls[f"PiBtle{i}Alpha"](copy) + copy
            # x = self.mduls[f"PiBtle{i}Alpha"](x - copy) + copy

        y = self.mduls['OutLayer'](x)
        return y*bc_cond_tsnr

class PeriodicBC_Network(torch.nn.Module):

    def __init__(self, 
                in_dim:int,
                out_dim:int,
                n_hid_lyr:int,
                nodes_per_lyr:int,
                scale:float,
                embd_dim:int,
                period:float):

        super().__init__()
        # kernel = torch.distributions.normal.Normal(loc=0.0, scale=scale).sample((in_dim+1, embd_dim//2))
        # self.register_buffer(name='kernel', tensor=kernel)
        # # network needs to match what was used in the normal plain old MLP results 
        # lyrs_lst = [torch.nn.Linear(in_features=embd_dim, out_features=nodes_per_lyr, bias=True)]

        self.mduls = torch.nn.ModuleDict()
        self.mduls[f"U"] = torch.nn.Linear(in_features=in_dim+1, out_features=nodes_per_lyr, bias=True)
        torch.nn.init.xavier_normal_(self.mduls[f"U"].weight.data, gain=1.41)
        self.mduls['U_Act_Func'] = torch.nn.Tanh()

        self.mduls[f"V"] = torch.nn.Linear(in_features=in_dim+1, out_features=nodes_per_lyr, bias=True)
        torch.nn.init.xavier_normal_(self.mduls[f"V"].weight.data, gain=1.41)
        self.mduls['V_Act_Func'] = torch.nn.Tanh()
        
        for i in range(n_hid_lyr):

            self.mduls[f"PiBtle{i}InLyr"] = torch.nn.Linear(in_features=in_dim+1, out_features=nodes_per_lyr, bias=True)
            torch.nn.init.xavier_normal_(self.mduls[f"PiBtle{i}InLyr"].weight.data, gain=1.41)
            self.mduls[f"PiBtle{i}MidLyr"] = torch.nn.Linear(in_features=nodes_per_lyr, out_features=nodes_per_lyr, bias=True)
            torch.nn.init.xavier_normal_(self.mduls[f"PiBtle{i}MidLyr"].weight.data, gain=1.41)
            self.mduls[f"PiBtle{i}OutLyr"] = torch.nn.Linear(in_features=nodes_per_lyr, out_features=in_dim+1, bias=True)
            torch.nn.init.xavier_normal_(self.mduls[f"PiBtle{i}OutLyr"].weight.data, gain=1.41)
            self.mduls[f"PiBtle{i}Alpha"] = torch.nn.Linear(in_features=1, out_features=1, bias=False)
            # self.mduls[f"PiBtle{i}Alpha"].weight.data = torch.tensor([[0.0]])
            self.mduls.register_parameter(name=f"alpha{i}", param=torch.nn.Parameter(data=torch.zeros((1,),), requires_grad=True))

        
        self.mduls['OutLayer'] = torch.nn.Linear(in_features=in_dim+1, out_features=out_dim, bias=True)
        torch.nn.init.xavier_normal_(self.mduls['OutLayer'].weight.data, gain=1.41)

        self.num_layers = n_hid_lyr

        self.prd = period

    def forward(self, x:torch.Tensor):
        """
        
        """
        embedded = torch.hstack([ torch.cos(self.prd*x[:,0:1]), torch.sin(self.prd*x[:,0:1]), x[:,1:]])
        U = self.mduls['U_Act_Func'](self.mduls[f"U"](embedded))
        V = self.mduls['V_Act_Func'](self.mduls[f"V"](embedded))
        # the pi- bottle neck forward passes. 
        for i in range(self.num_layers):
            copy = embedded
            embedded = torch.nn.Tanh()(self.mduls[f"PiBtle{i}InLyr"](embedded))
            embedded = embedded*U + (1-embedded)*V
            embedded = torch.nn.Tanh()(self.mduls[f"PiBtle{i}MidLyr"](embedded))
            embedded = embedded*U + (1-embedded)*V
            embedded = torch.nn.Tanh()(self.mduls[f"PiBtle{i}OutLyr"](embedded))
            embedded = self.mduls.__dict__["_parameters"][f"alpha{i}"]*embedded + (1-self.mduls.__dict__["_parameters"][f"alpha{i}"])*copy
            # embedded = self.mduls[f"PiBtle{i}Alpha"](embedded) - self.mduls[f"PiBtle{i}Alpha"](copy) + copy
            # embedded = self.mduls[f"PiBtle{i}Alpha"](embedded - copy) + copy

        y = self.mduls['OutLayer'](embedded)
        return y
        

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

    saveName = 'PirateNetResultsForSlurmJob'+str(jobID)+'ArrayNum'+str(jobVer)
    sv_mdl_name = 'LrnrForSlurmJob'+str(jobID)+'ArrayNum'+str(jobVer)
    # Optim Results Names
    # saveName = 'OptimalVer4ResultsForSlurmJob'+str(jobID)+'ArrayNum'+str(jobVer)
    # sv_mdl_name = 'OptimalVer4LrnrForSlurmJob'+str(jobID)+'ArrayNum'+str(jobVer)

    data = sio.loadmat(file_name=DataFile)
    x : np.ndarray = data['x'].flatten()
    t : np.ndarray = data['t'].flatten()
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

    if dflag==14:
        net = KlnGrd_Net(
            in_dim=2,
            out_dim=1,
            n_hid_lyr=2,
            nodes_per_lyr=nPrLU,
            scale=2.0,
            embd_dim=24
            ).to(device=dvc, dtype=torch.float32)
    else:
        ntwrk_prd = (2*torch.pi) / (bounds[0,1] - bounds[0,0])
        net = PeriodicBC_Network(
            in_dim=2,
            out_dim=1 ,
            n_hid_lyr=2,
            nodes_per_lyr=nPrLU,
            scale=2.0,
            embd_dim=24,
            period=ntwrk_prd,
            ).to(device=dvc, dtype=torch.float32)
        
    libr.fit(network=net, inpts=torch.rand(size=(2,2), device=dvc, dtype=torch.float32))
    lib_names = libr.get_library_names(net_out_func_names=['u'])

    init_lmbda = torch.zeros(size=(len(lib_names),1), device=dvc,)
    lrnr = EqLearner1D(net=net,
            Lmbda=init_lmbda.requires_grad_(True),
            lib_func=libr,
            sprs_slvr=slvr,
            data_dict=d_dict,
            tmprl_ords=[tmp_order],
            col_pnts_smplr=col_smplr,
            N_col_pnts=nCpnts,
            ntwk_out_names=["u"],
            device=dvc,
            data_type=torch.float32
    )

    os.makedirs(name="PirateNetRunStuff", exist_ok=True)
    os.chdir('PirateNetRunStuff')

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
    lrnr.WriteResults(data_set_name=data_set, file_name=data_set+"PirateNet", precision=5, true_eq=crctEQ, errors=[err, errs], act_func='Tanh()', **res_dict)
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