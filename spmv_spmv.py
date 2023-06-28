
import sys

import numpy as np
from numpy import ma
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gmean

from graph_utils import *
from plot_overall_gflops import pre_process, get_common_values, get_unfused_timing, clean_files, no_ker_types, mat_no, get_timing_matched, get_best_config

mat_no = 10

def plot_spmv_spmv(input_path1):
    df, ker_type, nnz, Serial, joint_LS, joint_LBC, joint_dagp, LBC_nonfused, mkl_unfused, config, wf_config, sf, df_mkl = \
        None, None, None, None, None, None, None, None, None, None, None, None, None
    kernel_data_id = []
    csv_file_list = clean_files(input_path1)
    exe_ser = np.zeros((no_ker_types, mat_no))
    exe_sf, flop_sf, ins_sf = np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no))
    exe_ulbc, flop_ulbc, ins_ulbc = np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no))
    exe_umkl, flop_umkl, ins_umkl = np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no))
    df_flop, nnz, dim, mat_list_global = get_common_values(input_path1 + 'flop_counts.csv')

    if 'spmv_spmv_static.csv' in csv_file_list:
        df, ker_type, Serial = pre_process(input_path1 + 'spmv_spmv_static.csv', mat_list_global)
        idxk = 0
        t_fl = 2*nnz / 1e9

        exe_ulbc[idxk], ins_ulbc[idxk], exe_umkl[idxk], ins_umkl[idxk], df_mkl = get_unfused_timing(df, mat_list_global, 'Parallel Non-fused CSR-CSR p1', 'Parallel Non-fused CSR-CSR p2',
                                                                                                    None,
                                                                                                    os.path.join(input_path1, 'spmv_spmv_mkl.csv'), 'Parallel MKL Non-fused CSR-CSR', 'Parallel MKL Non-fused CSR-CSR Analysis Time')
        config, wf_config = get_best_config(df, Serial, mat_list_global)
        exe_sf[idxk] =  np.minimum(get_timing_matched(mat_list_global, df, config[0][0]), get_timing_matched(mat_list_global, df, config[1][0]),get_timing_matched(mat_list_global, df, config[2][0] ))  #df[config[0][0]].values
        flop_sf[idxk], flop_ulbc[idxk], flop_umkl[idxk] = t_fl / exe_sf[idxk], t_fl / exe_ulbc[idxk], t_fl / exe_umkl[idxk]
        ins_sf[idxk] = get_timing_matched(mat_list_global, df, 'Parallel Fused CSR-CSR Analysis Time BFS') #df['Parallel Fused CSR-CSR Separated Analysis Time'].values
        kernel_data_id.append(idxk)

        geomean_sf = gmean(flop_sf[idxk]/flop_umkl[idxk])
        geomean_sf_un = gmean(flop_sf[idxk]/flop_ulbc[idxk])
        print('geomean_sf ', geomean_sf, 'geomean_sf_un ', geomean_sf_un)
        # percentage of cases where flop_sf is better than flop_umkl
        print(' % better: ', len(np.where(flop_sf[idxk]/flop_umkl[idxk]>=1))/132)
        # percentage of cases where flop_sf is better than flop_ulbc

    # plot flop_sf vs flop_ulbc vs flop_umkl
    fig, ax = plt.subplots()
    for idxk in kernel_data_id:
        ax.scatter(nnz, flop_ulbc[idxk], label=ker_type[idxk], marker='*')
        ax.scatter(nnz, flop_umkl[idxk], label='Unfused SpMV-SpMV, MKL',  c='g', marker='x')
        ax.scatter(nnz, flop_sf[idxk], label='Fused SpMV-SpMV, Sparse Fusion', facecolors='none', edgecolors='r', marker='o',)
    ax.set_xlabel('NNZ')
    ax.set_ylabel('FLOP/s')
    ax.set_xscale('log')

    ax.grid(False)
    # set x and y axis label
    ax.set_xlabel('NNZ', fontsize=20, fontweight='bold')
    #ax.set_ylabel('Time (seconds)', fontsize=20, fontweight='bold')
    # set x and y axis tick size
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    # set right and top axis off
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # set left and bottom axis bold
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    fig.set_size_inches(18, 8)
    # set x tick black
    ax.tick_params(axis='x', colors='black')
    # set y tick black
    ax.tick_params(axis='y', colors='black')
    ax.set_xscale('log')
    #ax.set_yscale('log')
    # show legend
    #fig.legend(handles, labels, fontsize=14, ncol=3, loc='upper center', frameon=True, borderaxespad=1)
    ax.legend(loc='upper left', fontsize=20, ncol=3, frameon=True, borderaxespad=1)
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')
    #fig.show()
    fig.savefig('mv-mv.pdf', bbox_inches='tight')
    #fig.show()

if __name__ == '__main__':
    plot_spmv_spmv(sys.argv[1])
