#!/bin/bash



BINPATH="singularity exec artifact.sif /usr/bin/python3.8"
LOGS=./logs/ 
SCRIPTPATH=./
UFDB=./mm/
LOGS=./logs/
MATLIST=./spd_list.txt



${BINPATH} dl_matrices.py ./mm/ ./spd_list.txt 

