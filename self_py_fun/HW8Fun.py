import os
import numpy as np
import matplotlib.pyplot as plt

def produce_trun_mean_cov(input_signal, input_type, E_val):
    length_per_electrode = 25
    sample_size_len = input_signal.shape[0]

    signals_reshaped = input_signal.reshape(sample_size_len, E_val, length_per_electrode)
    target_signals = signals_reshaped[input_type == 1]
    non_target_signals = signals_reshaped[input_type == 0]

    signal_tar_mean = target_signals.mean(axis=0)
    signal_ntar_mean = non_target_signals.mean(axis=0)

    signal_tar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    signal_ntar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    signal_all_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))

    for e in range(E_val):
        signal_tar_cov[e] = np.cov(target_signals[:, e, :], rowvar=False)
        signal_ntar_cov[e] = np.cov(non_target_signals[:, e, :], rowvar=False)
        signal_all_cov[e] = np.cov(signals_reshaped[:, e, :], rowvar=False)

    return signal_tar_mean, signal_ntar_mean, signal_tar_cov, signal_ntar_cov, signal_all_cov


def plot_trunc_mean(eeg_tar_mean, eeg_ntar_mean, subject_name, time_index, E_val, electrode_name_ls,
                    y_limit=[-5, 8], fig_size=(12, 12), save_path=None):
    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    axes = axes.flatten()

    for i in range(E_val):
        axes[i].plot(time_index, eeg_tar_mean[i], color='red', label='Target')
        axes[i].plot(time_index, eeg_ntar_mean[i], color='blue', label='Non-Target')
        axes[i].set_title(electrode_name_ls[i])
        axes[i].set_xlabel("Time (ms)")
        axes[i].set_ylabel("Amplitude (µV)")
        axes[i].set_ylim(y_limit)

    fig.suptitle(f"{subject_name} Mean ERP", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Saved mean plot to: {save_path}")

    plt.close()


def plot_trunc_cov(eeg_cov, cov_type, time_index, electrode_name_ls, participant_name=None, save_path=None):
    E_val = eeg_cov.shape[0]
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    X, Y = np.meshgrid(time_index, time_index)
    vmin = np.min(eeg_cov)
    vmax = np.max(eeg_cov)

    for i in range(E_val):
        im = axes[i].contourf(X, Y, eeg_cov[i], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].set_title(electrode_name_ls[i])
        axes[i].set_xlabel("Time (ms)")
        axes[i].set_ylabel("Time (ms)")
        axes[i].invert_yaxis()

    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02)
    fig.suptitle(f"{cov_type} Covariance", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])
    fig.subplots_adjust(right=0.85)

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Saved covariance plot to: {save_path}")

    plt.close()
