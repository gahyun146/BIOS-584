# HW8_main.py
import os
import numpy as np
import scipy.io as sio

from self_py_fun.HW8Fun import produce_trun_mean_cov, plot_trunc_mean, plot_trunc_cov

bp_low = 0.5
bp_upp = 6
electrode_num = 16
electrode_name_ls = ['F3','Fz','F4','T7','C3','Cz','C4','T8',
                     'CP3','CP4','P3','Pz','P4','PO7','PO8','Oz']
subject_name = 'K114'
time_index = np.linspace(0, 800, 25)

output_dir = os.path.abspath("K114")
os.makedirs(output_dir, exist_ok=True)

data_path = os.path.join(os.getcwd(), "..", "data", "K114_001_BCI_TRN_Truncated_Data_0.5_6.mat")
data_path = os.path.abspath(data_path)
eeg_trunc_obj = sio.loadmat(data_path)
eeg_trunc_signal = eeg_trunc_obj['Signal']
eeg_trunc_type = np.squeeze(eeg_trunc_obj['Type'], axis=1)
eeg_trunc_type = (eeg_trunc_type == 1).astype(int)

tar_mean, ntar_mean, tar_cov, ntar_cov, all_cov = produce_trun_mean_cov(
    eeg_trunc_signal, eeg_trunc_type, electrode_num
)

plot_trunc_mean(
    tar_mean, ntar_mean, subject_name, time_index, electrode_num, electrode_name_ls,
    save_path=os.path.join(output_dir, "Mean.png")
)

plot_trunc_cov(
    tar_cov, "Target", time_index, electrode_name_ls,
    save_path=os.path.join(output_dir, "Covariance_Target.png")
)

plot_trunc_cov(
    ntar_cov, "Non-Target", time_index, electrode_name_ls,
    save_path=os.path.join(output_dir, "Covariance_Non-Target.png")
)

plot_trunc_cov(
    all_cov, "All", time_index, electrode_name_ls,
    save_path=os.path.join(output_dir, "Covariance_All.png")
)

print("✅ All figures saved successfully in:", output_dir)
