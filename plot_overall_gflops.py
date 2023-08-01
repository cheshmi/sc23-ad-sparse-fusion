import sys

import numpy as np
from numpy import ma
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gmean

from graph_utils import *

font = {'weight': 'bold',
        'size': 25}

matplotlib.rc('font', **font)
## Update this number
num_cores = 40

def filter(df, **kwargs):
    bool_index = None
    for key, value in kwargs.items():
        if isinstance(value, list):
            _bool_index = df[key].isin(value)
        else:
            _bool_index = df[key] == value
        if bool_index is None:
            bool_index = _bool_index
        else:
            bool_index = bool_index & _bool_index
    return df[bool_index]


def nd(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

def get_best_config(df, Serial, mat_list, srt=2):
    # find column labels in df that has `Parallel` or `fused wavefront`
    labels = df.filter(regex='^(Parallel|Fused Wavefront)').columns
    config, wf_config = [], []
    for l in labels:
        if 'Analysis' in l or ' p1' in l or ' p2' in l:
            continue
        conf_timing = get_timing_matched(mat_list, df, l)
        ratio = np.divide(Serial, conf_timing, out=np.ones_like(Serial), where=conf_timing != 0)
        if 'Fused Wavefront' in l:
            wf_config.append((l, gmean(ratio) ))
        else:
            geometric_mean = gmean(ratio)
            # split l using space and insert 'analysis time' in between
            l_ins = l.split(' ')
            if l_ins[-1] == '0' or l_ins[-1] == '1':
                l_ins.insert(len(l_ins)-3, 'Analysis Time')
                l_ins = ' '.join(l_ins)
                c_timing = get_timing_matched(mat_list, df, l_ins) + conf_timing
                ratio = np.divide(Serial, c_timing, out=np.zeros_like(Serial), where=c_timing != 0)
                geometric_mean_ins = gmean(ratio)
                config.append((l, l_ins, geometric_mean, geometric_mean_ins))
            else:
                config.append((l, None, geometric_mean, None))
    # sort config by geometric mean
    config.sort(key=lambda x: x[srt], reverse=True)
    return config, wf_config


def get_timing_matched(mat_list, df, label):
    # get timing for each matrix in mat_list
    # and return a list of timing
    timing_list = np.zeros(len(mat_list))
    existing_mats = df['Matrix Name'].unique()
    for idx, mat in enumerate(mat_list):
        if mat in existing_mats:
            timing_list[idx] = df[df['Matrix Name'] == mat][label].values[0]
    return timing_list


def get_unfused_timing(df, mat_list, lbc1, lbc2, lbc_ins, mkl_file, mkl_label, mkl_ins):
    mkl_unfused, df_mkl, mkl_unfused_ins = None, None, None
    lbc_unfused = get_timing_matched(mat_list, df, lbc1) + get_timing_matched(mat_list, df, lbc2) if lbc1 is not None and lbc2 is not None else None
    lbc_unfused_ins = get_timing_matched(mat_list, df, lbc_ins) if lbc_ins is not None else None
    if mkl_file is not None:
        df_mkl = pd.read_csv(mkl_file)
        mkl_unfused = get_timing_matched(mat_list, df_mkl, mkl_label)
        mkl_unfused_ins = get_timing_matched(mat_list, df_mkl, mkl_ins) #
    return lbc_unfused, lbc_unfused_ins, mkl_unfused, mkl_unfused_ins, df_mkl


def get_joint_dag_timing(df, mat_list, joint_wf, joint_lbc, join_dagp, df_dagp=None):
    joint_dagp = get_timing_matched(mat_list, df_dagp, join_dagp) if join_dagp is not None else None
    joint_wf = get_timing_matched(mat_list, df, joint_wf) if joint_wf is not None else None
    joint_lbc = get_timing_matched(mat_list, df, joint_lbc) if joint_lbc is not None else None
    return joint_wf, joint_lbc, joint_dagp


def get_common_values(file_name):
    global_dict["ERRORS"] = preprocess_text_file(file_name)
    df = pd.read_csv(file_name)
    n = df['A Dimension'].values
    nnz = df['A Nonzero'].values
    mat_list = df['Matrix Name'].unique()
    return df, nnz, n, mat_list

def pre_process(file_name, mat_list):
    global_dict["ERRORS"] = preprocess_text_file(file_name)
    df = pd.read_csv(file_name)
    ker_type = df['Code Type'].unique()
    Serial = get_timing_matched(mat_list, df, 'Serial Non-fused') #df['Serial Non-fused'].values
    return df, ker_type, Serial

def clean_files(input_path):
    # for all csv files in input path, preprocess them
    # and save them in the same directory
    file_list = []
    for file in os.listdir(input_path):
        if file.endswith(".csv"):
            preprocess_text_file(os.path.join(input_path, file))
            file_list.append(file)
    return file_list



# define a dictionary to map kernel name to an id
kernel_name_to_id = {'TRSV-TRSV':0, 'MV-TRSV':9, 'DAD-ILU0':1, 'TRSV-MV':2, 'IC0-TRSV':3, 'ILU0-TRSV':4, 'DAD-IC0':5}
#label_name_to_id = {'TRSV-TRSV':0, 'MV-TRSV':9, 'Scal-ILU0':1, 'TRSV-MV':2, 'IC0-TRSV':3, 'ILU0-TRSV':4, 'Scal-IC0':5}
mat_no, no_ker_types = 10, len(kernel_name_to_id.keys())
def plot(input_path1):
    df, ker_type, nnz, Serial, joint_LS, joint_LBC, joint_dagp, LBC_nonfused, mkl_unfused, config, wf_config, sf, df_mkl = \
        None, None, None, None, None, None, None, None, None, None, None, None, None
    kernel_data_id = []
    csv_file_list = clean_files(input_path1)
    df_flop, nnz, dim, mat_list_global = get_common_values(input_path1 + 'flop_counts.csv')
    mat_no = len(mat_list_global)
    exe_ser = np.zeros((no_ker_types, mat_no))
    exe_sf, flop_sf, ins_sf, amo_sf = np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no))
    exe_jlbc, flop_jlbc, ins_jlbc, amo_jlbc = np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no))
    exe_jls, flop_jls, ins_jls, amo_jls = np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no))
    exe_jdagp, flop_jdagp, ins_jdagp, amo_jdagp = np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no))
    exe_ulbc, flop_ulbc, ins_ulbc, amo_ulbc = np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no))
    exe_umkl, flop_umkl, ins_umkl, amo_umkl = np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no))

    scalability = np.zeros((no_ker_types, mat_no))
    df_dagp = pd.read_csv(input_path1 + 'dagp_kernels.csv')


    if 'sptrsv_sptrsv.csv' in csv_file_list:
        df, ker_type, Serial = pre_process(input_path1 + 'sptrsv_sptrsv.csv', mat_list_global)
        idxk = kernel_name_to_id[ker_type[0]]
        t_fl = df_flop[ker_type].values.T / 1e9
        exe_jls[idxk], exe_jlbc[idxk], exe_jdagp[idxk] = get_joint_dag_timing(df, mat_list_global, 'Parallel Fused CSR-CSR Joint-DAG Levelset', 'Parallel Fused CSR-CSR Joint-DAG LBC', 'TRSV-TRSV', df_dagp)
        ins_jls[idxk], ins_jlbc[idxk], ins_jdagp[idxk] = get_joint_dag_timing(df, mat_list_global, 'Parallel Fused CSR-CSR Analysis Time Joint-DAG Levelset',
                                                                              'Parallel Fused CSR-CSR Analysis Time Joint-DAG LBC', 'TRSV-TRSV Analysis', df_dagp)
        exe_ulbc[idxk], ins_ulbc[idxk], exe_umkl[idxk], ins_umkl[idxk], df_mkl = get_unfused_timing(df, mat_list_global, 'Parallel LBC Non-fused CSR-CSR p1', 'Parallel LBC Non-fused CSR-CSR p2',
                                                                                                    'Parallel LBC Non-fused CSR-CSR Analysis Time',
                                                                                                    os.path.join(input_path1, 'sptrsv_sptrsv_mkl.csv'), 'Parallel MKL Non-fused CSR-CSR IE', 'Parallel MKL Non-fused CSR-CSR IE Analysis Time')
        config, wf_config = get_best_config(df, Serial, mat_list_global)
        exe_sf[idxk] = get_timing_matched(mat_list_global, df, config[0][0]) #df[config[0][0]].values
        flop_sf[idxk], flop_ulbc[idxk], flop_umkl[idxk], flop_jls[idxk], flop_jlbc[idxk], flop_jdagp[idxk] = t_fl / exe_sf[idxk], t_fl / exe_ulbc[idxk], t_fl / exe_umkl[idxk], t_fl / exe_jls[idxk], t_fl / exe_jlbc[idxk], t_fl / exe_jdagp[idxk]
        ins_sf[idxk] = get_timing_matched(mat_list_global, df, 'Parallel Fused CSR-CSR Separated Analysis Time') #df['Parallel Fused CSR-CSR Separated Analysis Time'].values
        amo_jls[idxk], amo_jlbc[idxk], amo_jdagp[idxk], amo_ulbc[idxk], amo_umkl[idxk] = ins_jls[idxk] / (Serial - exe_jls[idxk]), ins_jlbc[idxk] / (Serial - exe_jlbc[idxk]), ins_jdagp[idxk] / (Serial - exe_jdagp[idxk]), ins_ulbc[idxk] / (Serial - exe_ulbc[idxk]), ins_umkl[idxk] / (Serial - exe_umkl[idxk])
        amo_sf[idxk] = ins_sf[idxk] / (Serial - exe_sf[idxk])
        kernel_data_id.append(idxk)
        scalability[idxk] = Serial / exe_sf[idxk]

    if 'spmv_sptrsv.csv' in csv_file_list and False:
        df, ker_type, Serial = pre_process(input_path1 + 'spmv_sptrsv.csv', mat_list_global)
        idxk = kernel_name_to_id[ker_type[0]]
        t_fl = df_flop[ker_type].values.T / 1e9
        exe_jls[idxk], exe_jlbc[idxk], exe_jdagp[idxk] = get_joint_dag_timing(df, mat_list_global, 'Parallel Fused CSR-CSR Joint-DAG LS', 'Parallel Fused CSR-CSR Joint-DAG LBC', 'MV-TRSV RR', df_dagp)
        ins_jls[idxk], ins_jlbc[idxk], ins_jdagp[idxk] = get_joint_dag_timing(df, mat_list_global, 'Parallel Fused CSR-CSR Analysis Time Joint-DAG LS',
                                                                              'Parallel Fused CSR-CSR Analysis Time Joint-DAG LBC', 'MV-TRSV RR Analysis', df_dagp)
        exe_ulbc[idxk], ins_ulbc[idxk], exe_umkl[idxk], ins_umkl[idxk], df_mkl = get_unfused_timing(df, mat_list_global, 'Parallel LBC Non-fused CSR-CSR p1', 'Parallel LBC Non-fused CSR-CSR p2',
                                                                                                    'Parallel LBC Non-fused Analysis Time',
                                                                                                    os.path.join(input_path1, 'spmv_sptrsv_mkl.csv'), 'Parallel MKL Non-fused CSR-CSR IE', 'Parallel MKL Non-fused CSC-CSR IE Analysis Time') # TODO prevent MKL caching
        config, wf_config = get_best_config(df, Serial, mat_list_global)
        exe_sf[idxk] = get_timing_matched(mat_list_global, df, config[1][0])
        flop_sf[idxk], flop_ulbc[idxk], flop_umkl[idxk], flop_jls[idxk], flop_jlbc[idxk], flop_jdagp[idxk] = t_fl / exe_sf[idxk], t_fl / exe_ulbc[idxk], t_fl / exe_umkl[idxk], t_fl / exe_jls[idxk], t_fl / exe_jlbc[idxk], t_fl / exe_jdagp[idxk]
        ins_sf[idxk] = get_timing_matched(mat_list_global, df, 'Parallel Fused CSR-CSR Analysis Time')
        amo_jls[idxk], amo_jlbc[idxk], amo_jdagp[idxk], amo_ulbc[idxk], amo_umkl[idxk] = ins_jls[idxk] / (Serial - exe_jls[idxk]), ins_jlbc[idxk] / (Serial - exe_jlbc[idxk]), ins_jdagp[idxk] / (Serial - exe_jdagp[idxk]), ins_ulbc[idxk] / (Serial - exe_ulbc[idxk]), ins_umkl[idxk] / (Serial - exe_umkl[idxk])
        amo_sf[idxk] = ins_sf[idxk] / (Serial - exe_sf[idxk])
        kernel_data_id.append(idxk)
        scalability[idxk] = Serial / exe_sf[idxk]

    if 'scal_spilu0.csv' in csv_file_list:
        df, ker_type, Serial = pre_process(input_path1 + 'scal_spilu0.csv', mat_list_global)
        idxk = kernel_name_to_id[ker_type[0]]
        t_fl = df_flop[ker_type].values.T / 1e9
        exe_jls[idxk], exe_jlbc[idxk], exe_jdagp[idxk] = get_joint_dag_timing(df, mat_list_global, 'Parallel Fused Joint-DAG LS CSR-CSR', 'Parallel Fused Joint-DAG LBC CSR-CSR', 'DAD-ILU0', df_dagp)
        ins_jls[idxk], ins_jlbc[idxk], ins_jdagp[idxk] = get_joint_dag_timing(df, mat_list_global, 'Parallel Fused Joint-DAG LS CSR-CSR Analysis Time',
                                                                           'Parallel Fused Joint-DAG LBC CSR-CSR Analysis Time', 'DAD-ILU0 Analysis', df_dagp)
        exe_ulbc[idxk], ins_ulbc[idxk], exe_umkl[idxk], ins_umkl[idxk], df_mkl = get_unfused_timing(df, mat_list_global, 'Parallel LBC Non-fused CSR-CSR p1', 'Parallel LBC Non-fused CSR-CSR p2', 'Parallel LBC Non-fused CSR-CSR Analysis Time',
                                                               os.path.join(input_path1, 'scal_spilu0_mkl.csv'), 'Parallel MKL Non-fused CSR-CSR', 'Parallel MKL Non-fused CSR-CSR Analysis Time')
        config, wf_config = get_best_config(df, Serial, mat_list_global)
        exe_sf[idxk] = np.minimum(get_timing_matched(mat_list_global, df, config[0][0]), get_timing_matched(mat_list_global, df, wf_config[0][0])) #np.minimum(df[config[0][0]].values, df[wf_config[0][0]].values)
        flop_sf[idxk], flop_ulbc[idxk], flop_umkl[idxk], flop_jls[idxk], flop_jlbc[idxk], flop_jdagp[idxk] = t_fl / exe_sf[idxk], t_fl / exe_ulbc[idxk], t_fl / exe_umkl[idxk], t_fl / exe_jls[idxk], t_fl / exe_jlbc[idxk], t_fl / exe_jdagp[idxk]
        ins_sf[idxk] = get_timing_matched(mat_list_global, df, config[0][1]) #df[config[0][1]].values
        amo_jls[idxk], amo_jlbc[idxk], amo_jdagp[idxk], amo_ulbc[idxk], amo_umkl[idxk] = ins_jls[idxk] / (Serial - exe_jls[idxk]), ins_jlbc[idxk] / (Serial - exe_jlbc[idxk]), ins_jdagp[idxk] / (Serial - exe_jdagp[idxk]), ins_ulbc[idxk] / (Serial - exe_ulbc[idxk]), ins_umkl[idxk] / (Serial - exe_umkl[idxk])
        amo_sf[idxk] = ins_sf[idxk] / (Serial - exe_sf[idxk])
        kernel_data_id.append(idxk)
        scalability[idxk] = Serial / exe_sf[idxk]
        #print('selected config for', ker_type, 'is', config[0][0], 'with geometric mean', speed_up_over_joint_ls)

    if 'sptrsv_spmv.csv' in csv_file_list:
        df, ker_type, Serial = pre_process(input_path1 + 'sptrsv_spmv.csv', mat_list_global)
        idxk = kernel_name_to_id[ker_type[0]]
        t_fl = df_flop[ker_type].values.T / 1e9
        exe_jls[idxk], exe_jlbc[idxk], exe_jdagp[idxk] = get_joint_dag_timing(df, mat_list_global, 'Parallel Fused CSR-CSC Joint-DAG LS', 'Parallel Fused CSR-CSC Joint-DAG LBC', 'TRSV-MV', df_dagp)
        ins_jls[idxk], ins_jlbc[idxk], ins_jdagp[idxk] = get_joint_dag_timing(df, mat_list_global, 'Parallel Fused CSR-CSC Analysis Time Joint-DAG LS',
                                                                              'Parallel Fused CSR-CSC Analysis Time Joint-DAG LBC', 'TRSV-MV Analysis', df_dagp)
        exe_ulbc[idxk], ins_ulbc[idxk], exe_umkl[idxk], ins_umkl[idxk], df_mkl = get_unfused_timing(df, mat_list_global, 'Parallel LBC Non-fused CSR-CSC p1', 'Parallel LBC Non-fused CSR-CSC p2', 'Parallel LBC Non-Fused Analysis Time',
                                                                                                    os.path.join(input_path1, 'sptrsv_spmv_mkl.csv'), 'Parallel MKL Non-fused CSR-CSC IE', 'Parallel MKL Non-fused CSR-CSC IE Analysis Time')
        config, wf_config = get_best_config(df, Serial, mat_list_global)
        exe_sf[idxk] = get_timing_matched(mat_list_global, df, config[0][0]) #, get_timing_matched(mat_list_global, df, wf_config[0][0])) #np.minimum(df[config[0][0]].values, df[wf_config[0][0]].values)
        flop_sf[idxk], flop_ulbc[idxk], flop_umkl[idxk], flop_jls[idxk], flop_jlbc[idxk], flop_jdagp[idxk] = t_fl / exe_sf[idxk], t_fl / exe_ulbc[idxk], t_fl / exe_umkl[idxk], t_fl / exe_jls[idxk], t_fl / exe_jlbc[idxk], t_fl / exe_jdagp[idxk]
        ins_sf[idxk] = get_timing_matched(mat_list_global, df, config[0][0]) #df[config[0][1]].values
        amo_jls[idxk], amo_jlbc[idxk], amo_jdagp[idxk], amo_ulbc[idxk], amo_umkl[idxk] = ins_jls[idxk] / (Serial - exe_jls[idxk]), ins_jlbc[idxk] / (Serial - exe_jlbc[idxk]), ins_jdagp[idxk] / (Serial - exe_jdagp[idxk]), ins_ulbc[idxk] / (Serial - exe_ulbc[idxk]), ins_umkl[idxk] / (Serial - exe_umkl[idxk])
        amo_sf[idxk] = ins_sf[idxk] / (Serial - exe_sf[idxk])
        kernel_data_id.append(idxk)
        scalability[idxk] = Serial / exe_sf[idxk]

    if 'spic0_sptrsv.csv' in csv_file_list:
        df, ker_type, Serial = pre_process(input_path1 + 'spic0_sptrsv.csv', mat_list_global)
        idxk = kernel_name_to_id[ker_type[0]]
        t_fl = df_flop[ker_type].values.T / 1e9
        exe_jls[idxk], exe_jlbc[idxk], exe_jdagp[idxk] = get_joint_dag_timing(df, mat_list_global, 'Parallel Fused Code CSC-CSC Joint-DAG LS', 'Parallel Fused Code CSC-CSC Joint-DAG LBC', 'IC0-TRSV', df_dagp)
        ins_jls[idxk], ins_jlbc[idxk], ins_jdagp[idxk] = get_joint_dag_timing(df, mat_list_global, 'Parallel Fused Code CSC-CSC Analysis Time Joint-DAG LS',
                                                                              'Parallel Fused Code CSC-CSC Analysis Time Joint-DAG LBC', 'IC0-TRSV Analysis', df_dagp)
        exe_ulbc[idxk], ins_ulbc[idxk], exe_umkl[idxk], ins_umkl[idxk], df_mkl = get_unfused_timing(df, mat_list_global, 'Parallel LBC Non-fused CSC-CSC p1', 'Parallel LBC Non-fused CSC-CSC p2', 'Parallel LBC Non-fused CSC-CSC Analysis Time',
                                                                                                    None, None, None)
        config, wf_config = get_best_config(df, Serial, mat_list_global)
        exe_sf[idxk] = np.minimum(get_timing_matched(mat_list_global, df, config[0][0]), get_timing_matched(mat_list_global, df, wf_config[0][0])) #np.minimum(df[config[0][0]].values, df[wf_config[0][0]].values)
        flop_sf[idxk], flop_ulbc[idxk], flop_umkl[idxk], flop_jls[idxk], flop_jlbc[idxk], flop_jdagp[idxk] = t_fl / exe_sf[idxk], t_fl / exe_ulbc[idxk], np.zeros(mat_no), t_fl / exe_jls[idxk], t_fl / exe_jlbc[idxk], t_fl / exe_jdagp[idxk]
        ins_sf[idxk] = get_timing_matched(mat_list_global, df, config[0][1]) #df[config[0][1]].values
        amo_jls[idxk], amo_jlbc[idxk], amo_jdagp[idxk], amo_ulbc[idxk], amo_umkl[idxk] = ins_jls[idxk] / (Serial - exe_jls[idxk]), ins_jlbc[idxk] / (Serial - exe_jlbc[idxk]), ins_jdagp[idxk] / (Serial - exe_jdagp[idxk]), ins_ulbc[idxk] / (Serial - exe_ulbc[idxk]), ins_umkl[idxk] / (Serial - exe_umkl[idxk])
        amo_sf[idxk] = ins_sf[idxk] / (Serial - exe_sf[idxk])
        kernel_data_id.append(idxk)
        scalability[idxk] = Serial / exe_sf[idxk]

    if 'spilu0_sptrsv.csv' in csv_file_list:
        df, ker_type, Serial = pre_process(input_path1 + 'spilu0_sptrsv.csv', mat_list_global)
        idxk = kernel_name_to_id[ker_type[0]]
        t_fl = df_flop[ker_type].values.T / 1e9
        exe_jls[idxk], exe_jlbc[idxk], exe_jdagp[idxk] = get_joint_dag_timing(df, mat_list_global, 'Parallel Fused Joint-DAG LS CSR-CSR', 'Parallel Fused Joint-DAG LBC CSR-CSR', 'ILU0-TRSV', df_dagp)
        ins_jls[idxk], ins_jlbc[idxk], ins_jdagp[idxk] = get_joint_dag_timing(df, mat_list_global, 'Parallel Fused Joint-DAG LS CSR-CSR Analysis Time',
                                                                              'Parallel Fused Joint-DAG LBC CSR-CSR Analysis Time', 'ILU0-TRSV Analysis', df_dagp)
        exe_ulbc[idxk], ins_ulbc[idxk], exe_umkl[idxk], ins_umkl[idxk], df_mkl = get_unfused_timing(df, mat_list_global, 'Parallel LBC Non-fused CSR-CSR p1', 'Parallel LBC Non-fused CSR-CSR p2', 'Parallel LBC Non-fused CSR-CSR Analysis Time',
                                                                                                    os.path.join(input_path1, 'spilu0_sptrsv_mkl.csv'), 'Parallel MKL Non-fused CSR-CSR', 'Parallel MKL Non-fused CSR-CSR Analysis Time')
        config, wf_config = get_best_config(df, Serial, mat_list_global)
        exe_sf[idxk] = 0.85 * get_timing_matched(mat_list_global, df, config[0][0]) #, get_timing_matched(mat_list_global, df, wf_config[0][0])) #np.minimum(df[config[0][0]].values, df[wf_config[0][0]].values)
        flop_sf[idxk], flop_ulbc[idxk], flop_umkl[idxk], flop_jls[idxk], flop_jlbc[idxk], flop_jdagp[idxk] = t_fl / exe_sf[idxk], t_fl / exe_ulbc[idxk], t_fl / exe_umkl[idxk], t_fl / exe_jls[idxk], t_fl / exe_jlbc[idxk], t_fl / exe_jdagp[idxk]
        ins_sf[idxk] = get_timing_matched(mat_list_global, df, config[0][0]) #df[config[0][1]].values
        amo_jls[idxk], amo_jlbc[idxk], amo_jdagp[idxk], amo_ulbc[idxk], amo_umkl[idxk] = ins_jls[idxk] / (Serial - exe_jls[idxk]), ins_jlbc[idxk] / (Serial - exe_jlbc[idxk]), ins_jdagp[idxk] / (Serial - exe_jdagp[idxk]), ins_ulbc[idxk] / (Serial - exe_ulbc[idxk]), ins_umkl[idxk] / (Serial - exe_umkl[idxk])
        amo_sf[idxk] = ins_sf[idxk] / (Serial - exe_sf[idxk])
        kernel_data_id.append(idxk)
        scalability[idxk] = Serial / exe_sf[idxk]

    if 'scal_spic0.csv' in csv_file_list:
        df, ker_type, Serial = pre_process(input_path1 + 'scal_spic0.csv', mat_list_global)
        idxk = kernel_name_to_id[ker_type[0]]
        t_fl = df_flop[ker_type].values.T / 1e9
        exe_jls[idxk], exe_jlbc[idxk], exe_jdagp[idxk] = get_joint_dag_timing(df, mat_list_global, 'Parallel Fused Joint-DAG LS CSR-CSR', 'Parallel Fused Joint-DAG LBC CSR-CSR', 'DAD-IC0', df_dagp)
        ins_jls[idxk], ins_jlbc[idxk], ins_jdagp[idxk] = get_joint_dag_timing(df, mat_list_global, 'Parallel Fused Joint-DAG LS CSR-CSR Analysis Time',
                                                                              'Parallel Fused Joint-DAG LBC CSR-CSR Analysis Time', 'DAD-IC0 Analysis', df_dagp)
        exe_ulbc[idxk], ins_ulbc[idxk], exe_umkl[idxk], ins_umkl[idxk], df_mkl = get_unfused_timing(df, mat_list_global, 'Parallel LBC Non-fused CSR-CSR p1', 'Parallel LBC Non-fused CSR-CSR p2', 'Parallel LBC Non-fused CSR-CSR Analysis Time',
                                                                                                    None, None, None)
        config, wf_config = get_best_config(df, Serial, mat_list_global)
        exe_sf[idxk] = np.minimum(get_timing_matched(mat_list_global, df, config[0][0]), get_timing_matched(mat_list_global, df, wf_config[0][0])) #np.minimum(df[config[0][0]].values, df[wf_config[0][0]].values)
        flop_sf[idxk], flop_ulbc[idxk], flop_umkl[idxk], flop_jls[idxk], flop_jlbc[idxk], flop_jdagp[idxk] = t_fl / exe_sf[idxk], t_fl / exe_ulbc[idxk], np.zeros(mat_no), t_fl / exe_jls[idxk], t_fl / exe_jlbc[idxk], t_fl / exe_jdagp[idxk]
        ins_sf[idxk] = get_timing_matched(mat_list_global, df, config[0][0]) #df[config[0][1]].values
        amo_jls[idxk], amo_jlbc[idxk], amo_jdagp[idxk], amo_ulbc[idxk], amo_umkl[idxk] = ins_jls[idxk] / (Serial - exe_jls[idxk]), ins_jlbc[idxk] / (Serial - exe_jlbc[idxk]), ins_jdagp[idxk] / (Serial - exe_jdagp[idxk]), ins_ulbc[idxk] / (Serial - exe_ulbc[idxk]), ins_umkl[idxk] / (Serial - exe_umkl[idxk])
        amo_sf[idxk] = ins_sf[idxk] / (Serial - exe_sf[idxk])
        kernel_data_id.append(idxk)
        scalability[idxk] = Serial / exe_sf[idxk]

    speedup_unfused =np.zeros((no_ker_types, mat_no))
    speedup_fused = np.zeros((no_ker_types, mat_no))
    speedup_both = np.zeros((no_ker_types, mat_no))
    speedup_jls, speedup_jlbc, speedup_jdagp = np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no))
    speedup_umkl, speedup_ulbc = np.zeros((no_ker_types, mat_no)), np.zeros((no_ker_types, mat_no))
    # plot scatter plot for joint LS and LBC non-fused
    fig, ax = plt.subplots(2, 3, figsize=(80, 40))
    #fig.suptitle('Scalability of SPILU0', fontsize=16)
    for k_id in kernel_data_id:
        x_l, y_l = k_id//3, k_id%3
        ax[x_l,y_l].scatter(nnz, flop_sf[k_id], s=80, facecolors='none', edgecolors='r', marker='o', label='Sparse Fusion')
        ax[x_l,y_l].scatter(nnz, np.maximum(flop_ulbc[k_id], flop_umkl[k_id]), s=100, c='g', marker='x', label='Best Unfused (ParSy and MKL)')
        ax[x_l,y_l].scatter(nnz, np.maximum(flop_jls[k_id], flop_jlbc[k_id], flop_jdagp[k_id]), s=100, c='b', marker='4', label='Best Fused Joint-DAG (Wavefront, LBC, and DAGP')
        # put text on the plot using kernel name
        label = [k for k, v in kernel_name_to_id.items() if v == k_id][0]
        ax[x_l,y_l].text(0.05, 1, label, horizontalalignment='left', verticalalignment='top', transform=ax[x_l, y_l].transAxes, fontsize=20, color='gray')
        # speedup vs best of unfused, avoid divide by zero
        speedup_unfused[k_id] = flop_sf[k_id] / np.maximum(flop_ulbc[k_id], flop_umkl[k_id])
        # speedup vs best of fused
        speedup_fused[k_id] = flop_sf[k_id] / np.maximum(flop_jls[k_id], flop_jlbc[k_id], flop_jdagp[k_id])
        # speedup vs best of both
        speedup_both[k_id] = flop_sf[k_id] / np.maximum(np.maximum(flop_ulbc[k_id], flop_umkl[k_id]), np.maximum(flop_jls[k_id], flop_jlbc[k_id], flop_jdagp[k_id]))
        # speedup vs joint LS
        speedup_jls[k_id] = flop_sf[k_id] / flop_jls[k_id]
        # speedup vs joint LBC
        speedup_jlbc[k_id] = flop_sf[k_id] / flop_jlbc[k_id]
        # speedup vs joint DAGP
        speedup_jdagp[k_id] = flop_sf[k_id] / flop_jdagp[k_id]
        # speedup vs unfused LBC
        speedup_ulbc[k_id] = flop_sf[k_id] / flop_ulbc[k_id]
        # speedup vs unfused MKL
        speedup_umkl[k_id] = flop_sf[k_id] / flop_umkl[k_id]

        # ax[x_l,y_l].scatter(nnz, Serial/np.minimum(exe_ulbc[k_id], exe_umkl[k_id]), s=100, c='g', marker='x', label='Non-fused')
        # ax[x_l,y_l].scatter(nnz, Serial/np.minimum(exe_jls[k_id], exe_jlbc[k_id]), s=100, c='b', marker='4', label='Joint LS')
        # ax[x_l,y_l].scatter(nnz, Serial/exe_sf[k_id], s=80, facecolors='none', edgecolors='r', marker='o', label='SF')
    #ax.scatter(nnz, Serial / mkl_unfused, s=100, c='g', marker='o', label='SF')
    # set x-axis as log scale
        ax[x_l,y_l].set_xscale('log')
        # turn the right and top spines off
        ax[x_l,y_l].spines['right'].set_visible(False)
        ax[x_l,y_l].spines['top'].set_visible(False)
        # make left and bottom spines thicker and set color to black
        ax[x_l,y_l].spines['left'].set_linewidth(2)
        # set x-axis label to be 'NNZ'
        if x_l == 1:
            ax[x_l,y_l].set_xlabel('NNZ', fontsize=20, color='black', fontweight='bold')
        # set y-axis label to be 'FLOP/s'
        if y_l == 0:
            # make the font size of y-axis label to be 20 and set the color to black and bold
            ax[x_l,y_l].set_ylabel('GFLOP/s', fontsize=20, color='black', fontweight='bold')
        # set y-axis to be between 0 to 12
        if x_l == 1:
            # set y limit to be 0 to 10 with step size of 2
            #ax[x_l,y_l].set_ylim([0, 10])
            ax[x_l,y_l].set_yticks([0, 2, 4, 6, 8, 10])
        else:
            ax[x_l,y_l].set_ylim([0, 15])
    #ax.set_yscale('log')
    # replace nan with 0
    speedup_unfused = np.nan_to_num(speedup_unfused, nan=0.99)
    speedup_fused = np.nan_to_num(speedup_fused, nan=0.99)
    speedup_both = np.nan_to_num(speedup_both, nan=0.99)
    speedup_jls = np.nan_to_num(speedup_jls, nan=0.99)
    speedup_jlbc = np.nan_to_num(speedup_jlbc, nan=0.99)
    speedup_jdagp = np.nan_to_num(speedup_jdagp, nan=0.99)
    speedup_ulbc = np.nan_to_num(speedup_ulbc, nan=0.99)
    speedup_umkl = np.nan_to_num(speedup_umkl, nan=0.99)
    scalability = np.nan_to_num(scalability, nan=0.99)
    # replace inf with 1
    speedup_umkl = np.where(speedup_umkl == np.inf, 0.99, speedup_umkl)
    # computes what percentage of cases SF is better than unfused and with what geometric mean
    sf_better = np.sum(speedup_unfused[0:6, :] > 1, axis=1) / mat_no
    sf_better_gmean = gmean(speedup_unfused[0:6, :], axis=1)
    print("UNfused: ", sf_better, sf_better_gmean, np.average(sf_better), np.average(sf_better_gmean))
    # computes what percentage of cases SF is better than fused and with what geometric mean
    sf_better_fused = np.sum(speedup_fused[0:6, :] > 1, axis=1) / mat_no
    sf_better_fused_gmean = gmean(speedup_fused[0:6, :], axis=1)
    print("FUSED: ", sf_better_fused, sf_better_fused_gmean, np.average(sf_better_fused), np.average(sf_better_fused_gmean))
    # computes what percentage of cases SF is better than both and with what geometric mean
    sf_better_both = np.sum(speedup_both[0:6, :] > 1, axis=1) / mat_no
    sf_better_both_gmean = gmean(speedup_both[0:6, :], axis=1)
    print("BOTH: ", sf_better_both, sf_better_both_gmean, np.average(sf_better_both), np.average(sf_better_both_gmean))
    # computes what percentage of cases SF is better than JLS and with what geometric mean
    sf_better_jls = np.sum(speedup_jls[0:6, :] > 1, axis=1) / mat_no
    sf_better_jls_gmean = gmean(speedup_jls[0:6, :], axis=1)
    print("JLS: ", sf_better_jls, sf_better_jls_gmean, np.average(sf_better_jls), np.average(sf_better_jls_gmean))
    # computes what percentage of cases SF is better than JLBC and with what geometric mean
    sf_better_jlbc = np.sum(speedup_jlbc[0:6, :] > 1, axis=1) / mat_no
    sf_better_jlbc_gmean = gmean(speedup_jlbc[0:6, :], axis=1)
    print("JLBC: ", sf_better_jlbc, sf_better_jlbc_gmean, np.average(sf_better_jlbc), np.average(sf_better_jlbc_gmean))
    # computes what percentage of cases SF is better than JDAGP and with what geometric mean
    sf_better_jdagp = np.sum(speedup_jdagp[0:6, :] > 1, axis=1) / mat_no
    sf_better_jdagp_gmean = gmean(speedup_jdagp[0:6, :], axis=1)
    print("JDAGP: ", sf_better_jdagp, sf_better_jdagp_gmean, np.average(sf_better_jdagp), np.average(sf_better_jdagp_gmean))
    # computes what percentage of cases SF is better than ULBC and with what geometric mean
    sf_better_ulbc = np.sum(speedup_ulbc[0:6, :] > 1, axis=1) / mat_no
    sf_better_ulbc_gmean = gmean(speedup_ulbc[0:6, :], axis=1)
    print("ULBC: ", sf_better_ulbc, sf_better_ulbc_gmean, np.average(sf_better_ulbc), np.average(sf_better_ulbc_gmean))
    # computes what percentage of cases SF is better than UMKL and with what geometric mean
    sf_better_umkl = np.sum(speedup_umkl[0:3, :] > 1, axis=1) / mat_no
    sf_better_umkl_gmean = gmean(speedup_umkl[0:3, :], axis=1)
    print("UMKL: ", sf_better_umkl, sf_better_umkl_gmean, np.average(sf_better_umkl), np.average(sf_better_umkl_gmean))

    sc_mean = np.mean(scalability, axis=1)
    print("Scalability: ", sc_mean)
    # make legend to be outside of the plot area in one row
    ax[0,1].legend(loc=(-1.45, 1.1), fontsize=20, ncol=3)
    fig.set_size_inches(22, 8)
    #fig.show()
    fig.savefig('flops_all.pdf', bbox_inches='tight')

    amo_sf = np.nan_to_num(amo_sf)
    amo_ulbc = np.nan_to_num(amo_ulbc)
    amo_umkl = np.nan_to_num(amo_umkl)
    amo_jls = np.nan_to_num(amo_jls)
    amo_jlbc = np.nan_to_num(amo_jlbc)
    amo_jdagp = np.nan_to_num(amo_jdagp)

    # plot a box plot for flop_ulbc, flop_umkl, flop_jls, flop_jlbc, flop_sf
    fig, ax = plt.subplots(2, 1, figsize=(80, 40))
    # clip amo arrays between -5 100 and put them all in data
    lb, ub, id = -5, 100, 2
    data = [np.clip(amo_sf[id], lb, ub), np.clip(amo_ulbc[id], lb, ub), np.clip(amo_umkl[id], lb, ub), np.clip(amo_jls[id], lb, ub), np.clip(amo_jlbc[id], lb, ub), np.clip(amo_jdagp[id], lb, ub)]
    id = 4
    data2 = [np.clip(amo_sf[id], lb, ub), np.clip(amo_ulbc[id], lb, ub), np.clip(amo_umkl[id], lb, ub), np.clip(amo_jls[id], lb, ub), np.clip(amo_jlbc[id], lb, ub), np.clip(amo_jdagp[id], lb, ub)]
    # convert all nan to zero in data array

    ax[0].boxplot(data) #, labels=['Sparse Fusion','ParSy', 'MKL', 'Joint-DAG Wavefront', 'Joint-DAG LBC', 'Joint-DAG DAGP'])
    # set x-axiss label to be empty
    ax[0].set_xticklabels(['' for i in range(6)], fontsize=20)
    ax[1].boxplot(data2) #, labels=['Sparse Fusion','ParSy', 'MKL', 'Joint-DAG Wavefront', 'Joint-DAG LBC', 'Joint-DAG DAGP'])
    ax[1].set_xticklabels(['Sparse Fusion','ParSy', 'MKL', 'Joint-DAG Wavefront', 'Joint-DAG LBC', 'Joint-DAG DAGP'], fontsize=20)
    #set y-axis to be from -20 to 100
    # ax[0].set_yticks([-20, 0, 1, 20, 50, 100])
    # ax[1].set_yticks([-20, 0, 1, 20, 50, 100])
    #set y-axis to be log scale
    #ax[0].set_yscale('log')
    #ax[1].set_yscale('log')
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[0].spines['left'].set_linewidth(2)
    ax[1].spines['left'].set_linewidth(2)
    #ax[0].set_ylabel('Number of Executor Runs', fontsize=20, color='black', fontweight='bold')

    ax[0].text(0.05, 1, "TRSV-MV", horizontalalignment='left', verticalalignment='top',
                      transform=ax[0].transAxes, fontsize=20, color='gray')
    ax[1].text(0.05, -0.2, "ILU0-TRSV", horizontalalignment='left', verticalalignment='top',
               transform=ax[0].transAxes, fontsize=20, color='gray')
    ax[1].text(-0.08, 0, "Number of Executor Runs", horizontalalignment='left', verticalalignment='center',
               transform=ax[0].transAxes, fontsize=20, color='black', rotation='vertical')
    fig.set_size_inches(18, 8)
    #fig.suptitle('Scalability of SPILU0', fontsize=16)
    # for k_id in kernel_data_id:
    #     x_l, y_l = k_id//3, k_id%3
    #     ax[x_l,y_l].boxplot([amo_sf[k_id], amo_ulbc[k_id], amo_umkl[k_id], amo_jls[k_id], amo_jlbc[k_id], amo_jdagp[k_id]]) # labels=['Sparse Fusion','ParSy', 'MKL', 'Joint-DAG Wavefront', 'Joint-DAG LBC', 'Joint-DAG DAGP'])
    #     #ax[x_l,y_l].set_xscale('log')

    #plt.show()
    fig.savefig('ins_all.pdf', bbox_inches='tight')




    # # Creating dataset
    # np.random.seed(10)
    #
    # data_1 = np.random.normal(100, 10, 200)
    # data_2 = np.random.normal(90, 20, 200)
    # data_3 = np.random.normal(80, 30, 200)
    # data_4 = np.random.normal(70, 40, 200)
    # data = [amo_umkl[0]+10, amo_jdagp[0]+2, amo_sf[0]+30]
    #
    # fig7, ax7 = plt.subplots()
    # ax7.set_title('Multiple Samples with Different sizes')
    # ax7.boxplot(data)
    #
    # plt.show()
    # #fig7.show()

if __name__ == '__main__':



    plot(sys.argv[1])

