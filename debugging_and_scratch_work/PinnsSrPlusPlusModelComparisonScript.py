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

    # saveName = 'ResultsForSlurmJob'+str(jobID)+'ArrayNum'+str(jobVer)
    # sv_mdl_name = 'LrnrForSlurmJob'+str(jobID)+'ArrayNum'+str(jobVer)
    # Optim Results Names
    saveName = 'OptimalVer4ResultsForSlurmJob'+str(jobID)+'ArrayNum'+str(jobVer)
    sv_mdl_name = 'OptimalVer4LrnrForSlurmJob'+str(jobID)+'ArrayNum'+str(jobVer)

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
                     seed=data_seed, to_float=True,)

    # (d_seed, _, Ns, subsample_prcntg,
    #             X_trn, U_trn,
    #             X_tst, U_tst,
    #             bounds) = PDELearningMatData(fname=DataFile, Sptldims=1, split=0.80, smpleprcnt=0.20, noisePrcntg=noise, 
    #                  seed=data_seed, to_float=True, N_trn_pnts=nDpnts)

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

    if prd_stuff or four_stuff or rnd_wght_fct:
        net = Networks.MLP(in_dim=2, num_layers=numLyrsU, hid_dim=nPrLU,out_dim=1,
                          activ_func=torch.nn.Tanh(),
                          prd_stuff=prd_stuff, four_stuff=four_stuff,
                          rnd_wght_fct=rnd_wght_fct).to(device=dvc)

    else:
        net = torch.nn.Sequential(
                torch.nn.Linear(in_features=2, out_features=nPrLU, bias=True),
                torch.nn.Tanh(),
                torch.nn.Linear(in_features=nPrLU, out_features=nPrLU, bias=True),
                torch.nn.Tanh(),
                torch.nn.Linear(in_features=nPrLU, out_features=nPrLU, bias=True),
                torch.nn.Tanh(),
                torch.nn.Linear(in_features=nPrLU, out_features=nPrLU, bias=True),
                torch.nn.Tanh(),
                torch.nn.Linear(in_features=nPrLU, out_features=nPrLU, bias=True),
                torch.nn.Tanh(),
                torch.nn.Linear(in_features=nPrLU, out_features=nPrLU, bias=True),
                torch.nn.Tanh(),
                torch.nn.Linear(in_features=nPrLU, out_features=1, bias=True)
        ).to(device=dvc, dtype=torch.float32)
    
        # # for param in net.parameters():
        # #     if param.ndim==2:
        # #         torch.nn.init.xavier_normal_(param, gain=1.41)
        # #     elif param.ndim==1:
        # #         torch.nn.init.zeros_(param)
    
        for name, param in net.named_parameters(): 
            if 'weight'in name:
                torch.nn.init.xavier_normal_(param.data, gain=1.41)
            # if 'bias' in name:
            #     torch.nn.init.zeros_(param.data)
    
    # net = Networks.MLP(in_dim=2, num_layers=numLyrsU, hid_dim=nPrLU,out_dim=1,
    #                       activ_func=torch.nn.Tanh(),
    #                       prd_stuff=prd_stuff, four_stuff=four_stuff,
    #                       rnd_wght_fct=rnd_wght_fct).to(device=dvc)

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

    run_time = 0
    strt = perf_counter()
    lrnr.AdamsOptimTraining(mode="pre", alpha=preAlpha, gamma=preGamma, min_epochs=min_epchs, max_epochs=max_epchs, 
                            lrn_rt=lrn_rt, lp_ord=1.0, outputFreq=200,
                            betas=(0.9, 0.999), threshold=False,)
    run_time += (perf_counter() - strt)

    lrnr.ContourLikeComparisonPlot(dataset_name=data_set, pts=pts, values=U, fig_title=' ', state="PreTrained", show_fig=False, save_fig=True, save_title=saveName+'PreTrained', dif=False,exact=False, learned=False)
    
    # pre_fig, pre_axs = plt.subplot_mosaic(mosaic=[["Data","EQ"],["Tot","Tot"]], sharex=False, sharey=False, figsize=(20,10), layout='constrained')
    # pre_axs["Data"].semilogy(np.arange(0, lrnr._AdamsPreTrnEpochs), lrnr.AdamsPreTrnDataLoss, label="Train Loss")
    # pre_axs["Data"].semilogy(np.arange(0, lrnr._AdamsPreTrnEpochs), lrnr.AdamsPreTstDataLoss, 'r', label='Test Loss')
    # pre_axs["Data"].set_title("Data Loss")
    # pre_axs["EQ"].semilogy(np.arange(0, lrnr._AdamsPreTrnEpochs), lrnr.AdamsPreTrnEqLoss * lrnr._AdamsPreTrnAlpha, label="Train Loss")
    # pre_axs["EQ"].semilogy(np.arange(0, lrnr._AdamsPreTrnEpochs), lrnr.AdamsPreTstEqLoss * lrnr._AdamsPreTrnAlpha, 'r', label='Test Loss')
    # pre_axs["EQ"].set_title(r"EQ Loss times $\alpha$")
    # pre_axs["Tot"].semilogy(np.arange(0, lrnr._AdamsPreTrnEpochs), lrnr.AdamsPreTrnLoss, label="Train Loss")
    # pre_axs["Tot"].semilogy(np.arange(0, lrnr._AdamsPreTrnEpochs), lrnr.AdamsPreTstLoss, 'r', label='Test Loss')
    # pre_axs["Tot"].set_title(r"Combined loss = $\mathcal{L}_{D} + \alpha \mathcal{L}_{pde} +\gamma L_{1}\left(\lambda\right)$ | $(\alpha, \gamma) = $" + f"({lrnr._AdamsPreTrnAlpha}, {lrnr._AdmasPreTrnGamma})");
    # pre_fig.suptitle(r"Pretaining with $\alpha$ = " + f"{lrnr._AdamsPreTrnAlpha}");

    # loss_plots_dir = os.path.join(os.getcwd(), "LossPlots", data_set)
    # os.makedirs(name=loss_plots_dir, exist_ok=True)
    # pre_fig.savefig(fname=os.path.join(loss_plots_dir,f"ingPreTrainingLossesForSlurmJob{jobID}ArrayNum{jobVer}.png"), format='png')
    
    # pre_l1_fig, pre_l1_ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, layout="constrained")
    # pre_l1_ax.semilogy(lrnr.AdamsPreLpLosses, 'k.', label="Calced L1")
    # pre_l1_fig.savefig(fname=os.path.join(loss_plots_dir,f"ingL1_LossesPreTrainedForSlurmJob{jobID}ArrayNum{jobVer}.png"), format='png')
    
    strt = perf_counter()
    lrnr.ADO_Training(iters=ADO_iters, optim_alpha_grwth_methd="poly", LBFGS=False, lbfgs_epochs=10, Early_Term=True, Save_File_Name=None, Early_RFE_Term=True, p=0.5)
    # lrnr.ADO_Training(iters=ADO_iters, optim_alpha_grwth_methd="poly", LBFGS=False, lbfgs_epochs=10, Early_Term=False, Save_File_Name=None, Early_RFE_Term=True, p=0.5, train_alphas=ADO_alphas/ADO_alphas)
    run_time += (perf_counter() - strt)

    # lrnr.ContourLikeComparisonPlot(dataset_name=data_set, pts=pts, values=U, fig_title=' ', state="ADO-Trained", show_fig=False, save_fig=True, save_title=saveName+'AdoTrained', dif=False,exact=False,learned=False)
    # lrnr.Loss_Score_Complexity_Plot(dataset_name=Dname, save_dir_name=saveName, save_figs=True)
    # n_compltd_ado_iters = lrnr._ADO_epchs.shape[0]
    # for ado_iter in range(n_compltd_ado_iters):
    #     indices = range(lrnr._ADO_epchs[:ado_iter].sum(), lrnr._ADO_epchs[:ado_iter+1].sum())
    #     ado_fig, ado_axs = plt.subplot_mosaic(mosaic=[["Data","EQ"],["Tot","Tot"]], sharex=True, sharey=False, figsize=(20,10), layout='constrained')
    #     ado_axs["Data"].semilogy(lrnr._ADOtrnDataLs[indices,0], 'b', label="Trn Data")
    #     ado_axs["Data"].semilogy(lrnr._ADOtstDataLs[indices,0], 'r', label="Tst Data")
    #     ado_axs["Data"].legend()
    #     ado_axs["Data"].set_title("Data Loss")

    #     ado_axs["EQ"].semilogy(lrnr._ADOtrnColloLs[indices,0],'b', label="Trn Data")
    #     ado_axs["EQ"].semilogy(lrnr._ADOtstColloLs[indices,0],'r', label="Tst Data")
    #     ado_axs["EQ"].legend()
    #     ado_axs["EQ"].set_title("EQ Loss")

    #     ado_axs["Tot"].semilogy(lrnr._ADOTrnLosses[indices,0], 'b-.', label="Trn Data")
    #     ado_axs["Tot"].semilogy(lrnr._ADOTstLosses[indices,0], 'r', label="Tst Data")
    #     ado_axs["Tot"].legend()
    #     ado_axs["Tot"].set_title(r"Combined loss = $\mathcal{L}_{D} + \alpha \mathcal{L}_{pde}$");
    
    #     ado_fig.savefig(fname=os.path.join(loss_plots_dir,f"ingAdoIter{ado_iter}EqLossesForSlurmJob{jobID}ArrayNum{jobVer}.png"), format='png')

    strt = perf_counter()
    lrnr.AdamsOptimTraining(mode="post", alpha=pst_alpha, gamma=0.0, min_epochs=min_pst_epchs, max_epochs=max_pst_epchs, 
                            lrn_rt=pst_lrn_rt, lp_ord=1.0, outputFreq=200,
                            betas=(0.9, 0.999), threshold=False,)
    run_time += (perf_counter() - strt)

    # pst_fig, pst_axs = plt.subplot_mosaic(mosaic=[["Data","EQ"],["Tot","Tot"]], sharex=False, sharey=False, figsize=(20,10), layout='constrained')
    # pst_axs["Data"].semilogy(lrnr.AdamsPstTrnDataLoss, label="Train Loss")
    # pst_axs["Data"].semilogy(lrnr.AdamsPstTstDataLoss, 'r', label='Test Loss')
    # pst_axs["Data"].set_title("Data Loss")
    # pst_axs["EQ"].semilogy(lrnr.AdamsPstTrnEqLoss, label="Train Loss")
    # pst_axs["EQ"].semilogy(lrnr.AdamsPstTstEqLoss, 'r', label='Test Loss')
    # pst_axs["EQ"].set_title("EQ Loss")
    # pst_axs["Tot"].semilogy(lrnr.AdamsPstTrnLoss, label="Train Loss")
    # pst_axs["Tot"].semilogy(lrnr.AdamsPstTstLoss, 'r', label='Test Loss')
    # pst_axs["Tot"].set_title(r"Combined loss = $\mathcal{L}_{D} + \mathcal{L}_{pde}$");
    # pst_fig.suptitle(r"Post Training with $\alpha$ = " + f"{lrnr._PstTrnAdamsAlpha}");
    # pst_fig.savefig(fname=os.path.join(loss_plots_dir,f"ingPostTrainingLossesForSlurmJob{jobID}ArrayNum{jobVer}.png"), format='png')
    
    # pst_l1_fig, pst_l1_ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, layout="constrained")
    # pst_l1_ax.semilogy(lrnr.AdamsPstLpLosses, 'k.', label="Calced L1")
    # pst_l1_fig.savefig(fname=os.path.join(loss_plots_dir,f"ingL1_LossesPostTrainedForSlurmJob{jobID}ArrayNum{jobVer}.png"), format='png')

    res_dict['run_time'] =  run_time
    err, errs, RHS_eq, = OneDimSols(dataset=Dname, lib=lib_names, lrnd_sol=lrnr.lmbda.data.cpu().numpy())
    learned = lrnr.Learned_EQ(output=False, sup_zeros=True,)
    print("The learned equation(s) was ...\n" + learned)
    crctEQ = 'u_'+'t'*tmp_order +' = '  + RHS_eq
    fig_title = 'Learned EQ - '+learned+'\n Correct EQ - '+crctEQ
    lrnr.WriteResults(data_set_name=data_set, file_name=data_set+"OptimalVer4", precision=5, true_eq=crctEQ, errors=[err, errs], act_func='Tanh()', **res_dict)
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