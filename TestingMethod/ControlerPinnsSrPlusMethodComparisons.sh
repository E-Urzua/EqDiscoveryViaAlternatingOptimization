#!/bin/bash

#SBATCH --partition=ai4science.p
#SBATCH --exclude=hypernova,kilonova
#SBATCH --time=36:00:00   # walltime
#SBATCH --array=0-39
# #SBATCH --array=0-7:1
#SBATCH -J "KdvSnV1Opt"   # job name
#SBATCH -o KdvSnV1Opt-%j-%a.out  # Out file XXX-jobID-ArrayTaskID
#SBATCH --ntasks=1
# SBATCH --gpus=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --mail-user=urzuae@uci.edu   # email address
#SBATCH --mail-type=END
#SBATCH --mail-type FAIL
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
. /pkg/modules/init/bash
module load anaconda3/2025.06
eval "$(/pkg/anaconda3/2025.06/bin/conda shell.bash hook)"
conda activate LrnPDEs_Pytorch
cd /home/urzuae/LrnPDEs_Pytorch/PaperStuff/PinnsSrPlus
job=$SLURM_JOB_ID
aryID=$SLURM_ARRAY_TASK_ID
node_name=$SLURMD_NODENAME
run=$(( $job - $aryID -1 ))
if [ $aryID -eq 39 ]; then
    run=$job
fi

python PinnsSrPlusPlusModelComparisonScript.py -dset=13 -ver=1 -data_type=0 -JobID=${run} -arrayID=${aryID} -node=${node_name}
