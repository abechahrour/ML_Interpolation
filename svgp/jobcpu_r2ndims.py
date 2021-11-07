#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=example_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3gb
#SBATCH --time=20:00:00
#SBATCH --account=lsa3
#SBATCH --partition=standard
#SBATCH --output=./output/slurm_%A.out
# The application(s) to execute along with its input arguments and options:

# Load modules
module load tensorflow/2.3.2
/bin/hostname


python3 R2_ndims.py -f Periodic -e 100000 -d 8 -b 5000 -n_train 5000000 -n_test 500000 -n_ind 2000



#python3 run_nn.py -f Poly -nds 32 -lyr 5 -l mse -a elu -d 1 -nout 1 -e 4000 -b 1000 -n_train 5000000 -n_test 500000 -opt adam -lr 0.001 -load 0

#python3 run_test.py -nds 128 -lyr 8 -l mse -a tanh -d 2 -nout 1 -e 80000 -b 5000 -n_train 5000 -n_test 10000
#python3 train.py -nd 32 -lyr 8 -l mae -d 2 -nout 22 -e 100 -b 100 -real 0 -load 0
#python3 NN_PrecisionAnalysis.py -nd 128 -lyr 8 -l mae -d 1 -nout 1 -e 100 -b 1024 -n_train 10000000 -n_test 1000000
#python3 PVC_Regression_TwoNN_test.py
