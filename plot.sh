#!/bin/bash



BINPATH="singularity exec artifact.sif /usr/bin/python3.8"
LOGS=./logs/ 
SCRIPTPATH=./
UFDB=./mm/
LOGS=./logs/
MATLIST=./spd_list.txt



${BINPATH} plot_overall_gflops.py ./logs/ 

${BINPATH} gauss_seidel.py ./logs/k_kernel_gs4.csv 

${BINPATH} spmv_spmv.py ./logs/ 