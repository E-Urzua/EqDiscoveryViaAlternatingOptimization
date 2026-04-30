import torch
import numpy as np

def DefaultEQsModelParameters(dFlag:int, ver:int, run:int=0, data_type:str="", computer:str="desktop"):
    """
    Create the models parameters specific to the data set that is being used to test with
    the selection of the training data points according to this methods D version. Also
    choose the seeds for the data selection and collocation point creation that is dependent
    on the selected subsample percentage and the ratio of collocation points to training 
    points. 
    
    """

    if dFlag==3: # Data set Allen_Cahn parameters
        # Data Parameters stuff
        Dname = 'Allen_Cahn'
        nDpnts = 10000
        nTrn = nDpnts//4
        nCpnts = 50000
        # LIbrary stuff
        sptl_ord = 2
        tmp_order = 1
        polyDeg = 3
        if ver==1:
            noise = 50
            # Network parameters
            numLyrsU = 5
            nPrLU = 48
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False
        elif ver==2:
            noise = 75
            # Network parameters
            numLyrsU = 5
            nPrLU = 48
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False
        elif ver==3:
            noise = 100
            # Network parameters
            numLyrsU = 5
            nPrLU = 48
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False
        else:
            raise ValueError(f"For the Allen Cahn Data Set, there are only three different versions so ver=1,2,3 only")
    
    elif dFlag==6: # Data set Burgers_Exp parameters
        Dname = 'Burgers_Exp'
        nDpnts = 4000
        nCpnts = 50000
        nTrn = nDpnts//4
        noise = 100
        # LIbrary stuff
        sptl_ord = 2
        tmp_order = 1
        polyDeg = 2
        # Network parameters
        numLyrsU = 5
        nPrLU = 48
        prd_stuff = None
        four_stuff = None
        rnd_wght_fct = False
    
    elif dFlag==8: # Data set Burgers_Sine parameters
        Dname = 'Burgers_Sine'
        nCpnts = 50000
        # LIbrary stuff
        sptl_ord = 2
        tmp_order = 1
        polyDeg = 2
        if ver==1:
            nDpnts = 10000
            nTrn = nDpnts//4
            noise = 100
            # Network parameters
            numLyrsU = 5
            nPrLU = 48
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False
        elif ver==2:
            nDpnts = 4000
            nTrn = nDpnts//4
            noise = 100
            # Network parameters
            numLyrsU = 5
            nPrLU = 48
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False
        elif ver==3:
            nDpnts = 2000
            nTrn = nDpnts//4
            noise = 60
            # Network parameters
            numLyrsU = 5
            nPrLU = 48
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False
        elif ver==4:
            nDpnts = 1000
            nTrn = nDpnts//4
            noise = 40
            # Network parameters
            numLyrsU = 5
            nPrLU = 48
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False
        else:
            raise ValueError(f"For the Burgers_Sine Data Set, there are only four different versions so ver=1,2,3,4 only")
    
    elif dFlag==9: # Data set heat_Sine_Exp parameters
        Dname = 'Heat_Sine_Exp*'
        nDpnts = 10000
        nTrn = nDpnts//4
        nCpnts = 50000
        noise = 100
        sptl_ord = 2
        tmp_order = 1
        polyDeg = 2
        # Network parameters
        numLyrsU = 5
        nPrLU = 48
        prd_stuff = None
        four_stuff = None
        rnd_wght_fct = False
        
    elif dFlag==10: # Data set heat_sine parameters
        Dname = 'heat_sine*'
        nDpnts = 10000
        nTrn = nDpnts//4
        nCpnts = 50000
        noise = 100
        sptl_ord = 2
        tmp_order = 1
        polyDeg = 2
         # Network parameters
        numLyrsU = 5
        nPrLU = 48
        prd_stuff = None
        four_stuff = None
        rnd_wght_fct = False
        
    elif dFlag==13: # Data set Korteweg-De Vries Equation parameters
        Dname = 'KdV_Sine'
        nDpnts = 10000
        nTrn = nDpnts//4
        nCpnts = 50000
        # Network parameters
        numLyrsU = 5
        nPrLU = 56
        prd_stuff = None
        four_stuff = None
        rnd_wght_fct = False
        if ver==1:
            noise = 10
            sptl_ord = 3
            tmp_order = 1
            polyDeg = 5
            # Network parameters
            numLyrsU = 5
            nPrLU = 48
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False

        elif ver==2:
            noise = 60
            sptl_ord = 3
            tmp_order = 1
            polyDeg = 5
            # Network parameters
            numLyrsU = 5
            nPrLU = 48
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False

        elif ver==3:
            noise = 75
            sptl_ord = 3
            tmp_order = 1
            polyDeg = 3
            # Network parameters
            numLyrsU = 5
            nPrLU = 48
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False
            
        elif ver==4:
            noise = 100
            sptl_ord = 3
            tmp_order = 1
            polyDeg = 3
            # Network parameters
            numLyrsU = 5
            nPrLU = 56
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False
            
        else:
            raise ValueError(f"For the KdV_Sine Data Set, there are only four different versions so ver=1,2,3,4 only")
    
    elif dFlag==14: # Data set Klein-Gordon Equation with Exp() init cond. parameters
        Dname = 'KG_Exp'
        nDpnts = 10000
        nCpnts = 50000
        nTrn = nDpnts//4
        # Network parameters
        numLyrsU = 5
        nPrLU = 48
        prd_stuff = None
        four_stuff = None
        rnd_wght_fct = False
        sptl_ord = 2
        tmp_order = 2
        polyDeg = 2
        if ver==1:
            noise = 50
            # Network parameters
            numLyrsU = 5
            nPrLU = 48
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False
        else:
            noise = 100
            # Network parameters
            numLyrsU = 5
            nPrLU = 48
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False
    
    elif dFlag==15: # Data set Dynamic Beam Equation parameters with Exp init. cond.
        Dname = 'Beam_Exp*'
        nDpnts = 10000
        nCpnts = 50000
        nTrn = nDpnts//4
        sptl_ord = 4
        tmp_order = 2
        polyDeg = 2
        # polyDeg = 1
        # Network parameters
        numLyrsU = 5
        nPrLU = 48
        prd_stuff = None
        four_stuff = None
        rnd_wght_fct = True
        if ver==1:
            noise = 25
            # Network parameters
            numLyrsU = 5
            nPrLU = 48
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False
        elif ver==2:
            noise = 50
            # Network parameters
            numLyrsU = 5
            nPrLU = 48
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False

    # # Training parameters
    # trnEpchs = 3000
    # btchSz = nDpnts
    # lrn_rt = 0.001
    # preAlpha = 0.001
    # preGamma = 1e-6
    # # ADO training
    # ado_adms_epochs = 1000
    # ADO_iters = 5
    # Kfolds = 40
    # ADO_alphas  = torch.linspace(0.01, 1.0, ADO_iters)
    # # Post-Training.
    # pst_adams_epchs = 3000
    # pst_alpha = 1.0
    # pst_lrn_rt = 0.001

    # Ver2
    # Training parameters
    min_epchs = 500
    max_epchs = 10000
    lrn_rt = 0.001
    # preAlpha = 1.0
    # preAlpha = 0.1
    # preAlpha = 0.01
    preAlpha = 0.001
    preGamma = 1e-6
    # ADO training
    ADO_iters = 6
    Kfolds = 10
    # Kfolds = 1
    ADO_alphas  = torch.linspace(1.0, 1.0, ADO_iters)
    # Post-Training.
    min_pst_epchs = 200
    max_pst_epchs = 10000
    pst_alpha=1.0
    pst_lrn_rt = 0.001
    
    xtra = {}

    if data_type=="SensorData":
        subfolder = "RandSensorDataSelectSeedFiles/"
        ending = '.txt'
    elif data_type=="RandPointsData":
        subfolder = "RandPointsDataSelectSeedFiles/"
        ending = '.txt'
    else:
        raise ValueError("Only acceptable values for 'data_type' inputs are 'SensorData' or 'RandPointsData' .")

    # Now return all the paremeter specific to each data set.
    if "blazar" in computer:
        fname = (  "PdeSeedFiles/" + subfolder + Dname + "_" + "N" + str(int(noise)) + "_" +
                        "P" + str(nDpnts) + ending)
    else:
        fname = (  "PdeSeedFiles/" + subfolder + Dname.replace('*', '%2A') + "_" + "N" + str(int(noise)) + "_" +
                        "P" + str(nDpnts) + ending)
        Dname = Dname.replace('*', '%2A')
    
    # if 'desktop' == computer:
    #          Dname = Dname.replace('*', '%2A')

    file = open(fname, 'r')
    lns = file.read()
    seeds = lns.split('\n')[:-1]  # there is blank line at the end that is why. 
    file.close()
    try:
        seed = int(seeds[run])
    except IndexError:
        raise ValueError(f"The run value was too high. There were less runs that asked for...")
    
    # seed=None  # for/when testing
    

    return (Dname, nDpnts, nTrn, nCpnts, noise, seed, sptl_ord, tmp_order, polyDeg, 
            numLyrsU, nPrLU, prd_stuff, four_stuff, rnd_wght_fct,
            ADO_iters, Kfolds, ADO_alphas, 
            min_epchs, max_epchs, lrn_rt, preAlpha, preGamma,
            min_pst_epchs, max_pst_epchs, pst_alpha, pst_lrn_rt)

def OptimizedEQsModelParameters(dFlag:int, ver:int, run:int=0, data_type:str="", computer:str="desktop"):
    """
    Create the models parameters specific to the data set that is being used to test with
    the selection of the training data points according to this methods D version. Also
    choose the seeds for the data selection and collocation point creation that is dependent
    on the selected subsample percentage and the ratio of collocation points to training 
    points. 
    
    """

    if dFlag==3: # Data set Allen_Cahn parameters
        # Data Parameters stuff
        Dname = 'Allen_Cahn'
        nDpnts = 10000
        nTrn = nDpnts//4
        nCpnts = 50000
        # LIbrary stuff
        sptl_ord = 2
        tmp_order = 1
        polyDeg = 3
        # Network parameters
        numLyrsU = 5
        nPrLU = 48
        prd_stuff = None
        four_stuff = None
        rnd_wght_fct = False
        # Training parameters
        lrn_rt = 0.001/10
        preAlpha = 0.001
        preGamma = 1e-6
        if ver==1:
            noise = 50
        elif ver==2:
            noise = 75
        elif ver==3:
            noise = 100
        else:
            raise ValueError(f"For the Allen Cahn Data Set, there are only three different versions so ver=1,2,3 only")
    
    elif dFlag==6: # Data set Burgers_Exp parameters
        Dname = 'Burgers_Exp'
        nDpnts = 4000
        nCpnts = 50000
        nTrn = nDpnts//4
        noise = 100
        # LIbrary stuff
        sptl_ord = 2
        tmp_order = 1
        polyDeg = 2

        # Network parameters
        numLyrsU = 5
        nPrLU = 48
        prd_stuff = None
        four_stuff = None
        rnd_wght_fct = False
        # Training Hyper Parameters
        lrn_rt = 0.001
        preAlpha = 1.0
        preGamma = 1e-6
        # ADO training
        ADO_iters = 6
        Kfolds = 1
        Kfolds = 10
        ADO_alphas = torch.linspace(preAlpha*1.0, 1.0, ADO_iters)
 
    elif dFlag==8: # Data set Burgers_Sine parameters
        Dname = 'Burgers_Sine'
        nCpnts = 50000
        # LIbrary stuff
        sptl_ord = 2
        tmp_order = 1
        polyDeg = 2
        # Network parameters
        numLyrsU = 5
        nPrLU = 48
        prd_stuff = None
        four_stuff = None
        rnd_wght_fct = False
        # Training Hyper Parameters
        lrn_rt = 0.001
        preAlpha = 1.0
        preGamma = 1e-6
        # ADO training
        ADO_iters = 6
        Kfolds = 1
        Kfolds = 10
        ADO_alphas = torch.linspace(preAlpha*1.0, 1.0, ADO_iters)
        xtra = {}
        if ver==1:
            nDpnts = 10000
            nTrn = nDpnts//4
            noise = 100
            
        elif ver==2:
            nDpnts = 4000  
            nTrn = nDpnts//4
            noise = 100
        elif ver==3:
            nDpnts = 2000
            nTrn = nDpnts//4
            noise = 60
        elif ver==4:
            nDpnts = 1000
            nTrn = nDpnts//4
            noise = 40
        else:
            raise ValueError(f"For the Burgers_Sine Data Set, there are only four different versions so ver=1,2,3,4 only")
    
    elif dFlag==9: # Data set heat_Sine_Exp parameters
        Dname = 'Heat_Sine_Exp*'
        nDpnts = 10000
        nTrn = nDpnts//4
        nCpnts = 50000
        noise = 100
        sptl_ord = 2
        tmp_order = 1
        polyDeg = 2
        # Network parameters
        numLyrsU = 5
        nPrLU = 48
        prd_stuff = None
        four_stuff = None
        rnd_wght_fct = False
        # Training parameters
        lrn_rt = 0.001
        preAlpha = 1.0
        preGamma = 1e-6
        # ADO training
        ADO_iters = 6
        # Kfolds = 1
        Kfolds = 10
        ADO_alphas  = torch.linspace(1.0, 1.0, ADO_iters)

    elif dFlag==10: # Data set heat_sine parameters
        Dname = 'heat_sine*'
        nDpnts = 10000
        nTrn = nDpnts//4
        nCpnts = 50000
        btchSz = nDpnts
        noise = 100
        sptl_ord = 2
        tmp_order = 1
        polyDeg = 2
         # Network parameters
        numLyrsU = 5
        nPrLU = 48
        prd_stuff = None
        four_stuff = None
        rnd_wght_fct = False
        # Training parameters
        trnEpchs = 3000
        lrn_rt = 0.001
        preAlpha = 1.0
        preGamma = 1e-6
        # ADO training
        ado_adms_epochs = 1000
        ADO_iters = 4
        Kfolds = 1
        ADO_alphas  = torch.linspace(1.0, 1.0, ADO_iters)
        # Post-Training.
        pst_adams_epchs = 1000
        pst_alpha=1.0
        pst_lrn_rt = 0.001
    
    elif dFlag==13: # Data set Korteweg-De Vries Equation parameters
        Dname = 'KdV_Sine'
        nDpnts = 10000
        nTrn = nDpnts//4
        nCpnts = 50000
        # Network parameters
        numLyrsU = 5
        nPrLU = 56
        prd_stuff = None
        four_stuff = None
        rnd_wght_fct = False
        xtra = {}
        if ver==1:
            noise = 10
            sptl_ord = 3
            tmp_order = 1
            polyDeg = 5
            # Training parameters
            lrn_rt = 0.001
            preAlpha = 1.0
            # preGamma = 1e-6
            preGamma = 1e-6
            # ADO training
            ADO_iters = 6
            # Kfolds = 1
            Kfolds = 10
            ADO_alphas  = torch.linspace(1.0, 1.0, ADO_iters)
            
        elif ver==2:
            noise = 60
            sptl_ord = 3
            tmp_order = 1
            polyDeg = 5
            # Network parameters
            numLyrsU = 5
            nPrLU = 56
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False
            # Training parameters
            lrn_rt = 0.001
            preAlpha = 1.0
            # preGamma = 1e-6
            preGamma = 1e-6
            # ADO training
            ADO_iters = 6
            # Kfolds = 1
            Kfolds = 10
            ADO_alphas  = torch.linspace(1.0, 1.0, ADO_iters)

        elif ver==3:
            noise = 75
            sptl_ord = 3
            tmp_order = 1
            polyDeg = 3
            # Training parameters
            lrn_rt = 0.001
            preAlpha = 1.0
            # preGamma = 1e-6
            preGamma = 1e-6
            # ADO training
            ADO_iters = 4
            # Kfolds = 1
            Kfolds = 10
            ADO_alphas  = torch.linspace(1.0, 1.0, ADO_iters)
            
        elif ver==4:
            noise = 100
            sptl_ord = 3
            tmp_order = 1
            polyDeg = 3
            # Training parameters
            lrn_rt = 0.001
            preAlpha = 1.0
            # preGamma = 1e-6
            preGamma = 1e-6
            # ADO training
            ADO_iters = 4
            # Kfolds = 1
            Kfolds = 10
            ADO_alphas  = torch.linspace(1.0, 1.0, ADO_iters)
        else:
            raise ValueError(f"For the KdV_Sine Data Set, there are only four different versions so ver=1,2,3,4 only")
    
    elif dFlag==14: # Data set Klein-Gordon Equation with Exp() init cond. parameters
        Dname = 'KG_Exp'
        nDpnts = 10000
        nCpnts = 50000
        nTrn = nDpnts//4
        nCpnts = 50000
        # Network parameters
        numLyrsU = 5
        nPrLU = 48
        prd_stuff = None
        four_stuff = None
        rnd_wght_fct = False
        sptl_ord = 2
        tmp_order = 2
        polyDeg = 2
        # Network parameters
        numLyrsU = 5
        nPrLU = 48
        prd_stuff = None
        four_stuff = None
        rnd_wght_fct = False
        if ver==1:
            noise = 50
            # Training parameters
            lrn_rt = 0.001
            preAlpha = 0.001
            preGamma = 1e-6
            # preGamma = 0.0
            # ADO training
            ADO_iters = 6
            # Kfolds = 1
            Kfolds = 10
            ADO_alphas  = torch.linspace(0.01, 1.0, ADO_iters)
        else:
            noise = 100
            # Training parameters
            lrn_rt = 0.001
            # preAlpha = 0.0
            preAlpha = 0.001
            preGamma = 1e-6
            # ADO training
            ADO_iters = 6
            # Kfolds = 1
            Kfolds = 10
            ADO_alphas  = torch.linspace(0.01, 1.0, ADO_iters)
    
    elif dFlag==15: # Data set Dynamic Beam Equation parameters with Exp init. cond.
        Dname = 'Beam_Exp*'
        nDpnts = 10000
        nCpnts = 50000
        nTrn = nDpnts//4
        sptl_ord = 4
        tmp_order = 2
        polyDeg = 2
        # polyDeg = 1
        # Network parameters
        numLyrsU = 5
        nPrLU = 48
        prd_stuff = None
        four_stuff = None
        rnd_wght_fct = True
        xtra = {}
        if ver==1:
            noise = 25
            # Network parameters
            numLyrsU = 5
            nPrLU = 48
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False

            # # Training parameters
            # trnEpchs = 300
            # # btchSz = 512
            # btchSz = 1024
            # lrn_rt = 0.001
            # # preAlpha = 0.0
            # preAlpha = 0.001
            # preGamma = 0
            # # ADO training
            # ado_adms_epochs = 150
            # ADO_iters = 4
            # Kfolds = 1
            # ADO_alphas  = torch.linspace(0.01, 1.0, ADO_iters)
            # # Post-Training.
            # pst_adams_epchs = 300
            # pst_alpha=1.0
            # pst_lrn_rt = 0.001
            # lbfgsEpochs = 100
            # # lbfgsEpochs = 5
            # lbfgsLrnRt = 0.01
            # xtra = {'lbfgsEpochs':lbfgsEpochs, 'lbfgsLrnRt':lbfgsLrnRt}

            # Training parameters
            trnEpchs = 3000
            btchSz = nDpnts
            lrn_rt = 0.001
            # preAlpha = 0.0
            preAlpha = 0.001
            preGamma = 1e-6
            # preGamma = 0.0
            # ADO training
            ado_adms_epochs = 1000
            ADO_iters = 4
            Kfolds = 1
            ADO_alphas  = torch.linspace(0.01, 1.0, ADO_iters)
            # Post-Training.
            pst_adams_epchs = 3000
            pst_alpha = 1.0
            pst_lrn_rt = 0.001
        elif ver==2:
            noise = 50
            
            # btchSz = 512
            # trnEpchs = 150
            # # trnEpchs = 0
            # lrn_rt = 0.001
            # preAlpha = 0.5
            # preGamma = 0.0
            # # ADO training
            # # ADO_iters = 2
            # ADO_iters = 4
            # Kfolds = 1
            # ADO_alphas  = np.linspace(preAlpha*2, 1.0, ADO_iters)
            # # Post-Training.
            # pst_adams_epchs = 3000
            # pst_alpha=1.0
            # pst_lrn_rt = 0.001
            # lbfgsEpochs = 150
            # # lbfgsEpochs = 0
            # lbfgsLrnRt = 0.01
            # xtra = {'lbfgsEpochs':lbfgsEpochs, 'lbfgsLrnRt':lbfgsLrnRt}
            # Network parameters
            numLyrsU = 5
            nPrLU = 48
            prd_stuff = None
            four_stuff = None
            rnd_wght_fct = False
            # Training parameters
            trnEpchs = 3000
            btchSz = nDpnts
            lrn_rt = 0.001
            # preAlpha = 0.0
            preAlpha = 0.001
            preGamma = 1e-6
            # preGamma = 0.0
            # ADO training
            ado_adms_epochs = 1000
            ADO_iters = 4
            Kfolds = 1
            ADO_alphas  = torch.linspace(0.01, 1.0, ADO_iters)
            # Post-Training.
            pst_adams_epchs = 3000
            pst_alpha = 1.0
            pst_lrn_rt = 0.001

    xtra = {}

    min_epchs = 500
    max_epchs = 10000
    # ADO training
    ADO_iters = 6
    Kfolds = 10
    
    ADO_alphas  = torch.linspace(1.0, 1.0, ADO_iters)
    # Post-Training.
    min_pst_epchs = 200
    max_pst_epchs = 10000
    pst_alpha = 1.0
    pst_lrn_rt = 0.001

    if data_type=="SensorData":
        subfolder = "RandSensorDataSelectSeedFiles/"
        ending = '.txt'
    elif data_type=="RandPointsData":
        subfolder = "RandPointsDataSelectSeedFiles/"
        ending = '.txt'
    else:
        raise ValueError("Only acceptable values for 'data_type' inputs are 'SensorData' or 'RandPointsData' .")

    # Now return all the paremeter specific to each data set.
    if "blazar" in computer:
        fname = (  "PdeSeedFiles/" + subfolder + Dname + "_" + "N" + str(int(noise)) + "_" +
                        "P" + str(nDpnts) + ending)
    else:
        fname = (  "PdeSeedFiles/" + subfolder + Dname.replace('*', '%2A') + "_" + "N" + str(int(noise)) + "_" +
                        "P" + str(nDpnts) + ending)
        Dname = Dname.replace('*', '%2A')
    
    # if 'desktop' == computer:
    #          Dname = Dname.replace('*', '%2A')

    file = open(fname, 'r')
    lns = file.read()
    seeds = lns.split('\n')[:-1]  # there is blank line at the end that is why. 
    file.close()
    try:
        seed = int(seeds[run])
    except IndexError:
        raise ValueError(f"The run value was too high. There were less runs that asked for...")
    
    # seed=None  # for/when testing
    

    return (Dname, nDpnts, nTrn, nCpnts, noise, seed, sptl_ord, tmp_order, polyDeg, 
            numLyrsU, nPrLU, prd_stuff, four_stuff, rnd_wght_fct,
            ADO_iters, Kfolds, ADO_alphas, 
            min_epchs, max_epchs, lrn_rt, preAlpha, preGamma,
            min_pst_epchs, max_pst_epchs, pst_alpha, pst_lrn_rt)

