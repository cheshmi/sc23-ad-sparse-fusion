# Sparse Fusion Benchmark
This directory contains scripts for reproducing results in 
the sparse fusion paper at SC23.





[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8088414.svg)](https://doi.org/10.5281/zenodo.8088414)



  


# How to run the artifact
* The sympiler-bench repository should be cloned: 
```
	git clone https://github.com/cheshmi/sc23-ad-sparse-fusion.git
	cd sc23-ad-sparse-fusion
```

* The singularity image should be copied to the same directory that the code is cloned. The `*.sif` is provided as part of the artifact ( see [here](https://zenodo.org/record/8083006)). 

 
You can test the image by running the following command from the current directory:
```    
    singularity exec --env LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64 artifact.sif /source/fusion/build/demo/sptrsv_sptrsv_demo
```    
The output is a set of comma separated values such as specifications for a random matrix and execution time of different tools for the matrix.

* You can run the following script and come back in half an hour:
```
bash run_sparse_fusion_ad.sh
```


Otherwise, you can follow below steps:

* The datasets should be downloaded by calling:
```    
    bash dl_matrix_dataset.sh 
```    
Matrices are downloaded into the _mm_ directory in the current directory (This requires internet connection). Only 10 matrices are selected to speedup the evaluation time.

* The sparse fusion (Experiment 1) experiment can be executed by emitting:
```
	bash run_sfusion.sh
```
For running on compute node:
```
	sbatch run_sfusion.sh
```
You might need to update scripts with new absolute paths to the dataset and the image file. Also singularity module should be loaded for running on a server.


* The MKL evaluation (Experiment 2) can be done by running:
```
	bash run_mkl.sh
```    

* The DAGP evaluation (Experiment 3) can be reproduced by calling:
```
bash run_dagp.sh
``` 


* The Gauss-Seidel case study (Experiment 4) can be reproduced by calling:
```
	bash run_gs.sh
```
    
* Upon successful completion of experiments, all results should be stored as comma separated values (CSV) files under the _./logs/_ directory and are ready to be plotted. You can call:
```
bash plot.sh
```

 to create plots. Plots are stored in the current directory as PDF files.

