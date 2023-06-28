#!/bin/bash
#SBATCH --job-name="fusion"
#SBATCH --output="fusion_gs.%j.%N.out"
#SBATCH -p skx-normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --export=ALL
#SBATCH -t 11:00:05

# this is for stampede2, should change for other servers
#module load tacc-singularity/3.7.2

BINPATH="singularity exec --env LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64 artifact.sif  /source/fusion/build/demo/"
LOGS=./logs/ 
SCRIPTPATH=./
UFDB=./mm/
LOGS=./logs/
MATLIST=./spd_list.txt

THRD=20
NUM_THREAD=20
export OMP_NUM_THREADS=20

mkdir $LOGS



# DAGP
bash $SCRIPTPATH/run_exp.sh  "$BINPATH/dagp_demo" "$UFDB" 1 $THRD $MATLIST > $LOGS/dagp_kernels.csv

# mkdir $LOGS/plots

#  for f in $LOGS/*.csv; do
#  	python3 graph_gen.py -i $f -o $LOGS/plots/
#  done


#  python3 graph_gen.py -d $LOGS/plots/







