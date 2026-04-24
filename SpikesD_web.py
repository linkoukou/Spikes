# -*- coding: UTF-8 -*-
'''
-----------------------------------------------------------------------------------------------------
    Streamlit Web Application: Spikes Decoder (SpikesD) Web Version
    Version:          v0.1.0 
    Description:      Spike Detection Web Application 
    Author:           Li Jialin (adapted for Streamlit)
-----------------------------------------------------------------------------------------------------
'''
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d, label
from scipy.signal import medfilt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import os, struct, re
import io
import base64
from typing import Tuple, Optional, Callable, Literal, List, Dict, Any, Union
import tempfile

# Interactive charts: Plotly (zoom/pan like Matplotlib NavigationToolbar); optional click events
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None  # type: ignore

try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except ImportError:
    HAS_PLOTLY_EVENTS = False
    plotly_events = None  # type: ignore

# Set page configuration
st.set_page_config(
    page_title="Spikes Decoder Web",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global constants
MAD_EPS = 1e-6
MIN_SIGNAL_LENGTH = 10
MIN_WIDTH_MIN = 1
BUFFER_MIN = 0
HP_CUTOFF = 20
HP_ORDER = 2
PEAK_WINDOW_MS = 2
KERNEL_MIN = 1
MIN_GAP_MIN_PTS = 1
MAX_DATA_POINTS = 10**8
DEFAULT_DTYPE = np.float32
DEFAULT_DTYPE_SIZE = 4

# Initialize session state
if 'dat_channels' not in st.session_state:
    st.session_state.dat_channels = []
if 'dat_trace_labels' not in st.session_state:
    st.session_state.dat_trace_labels = []
if 'dat_sampling_rates' not in st.session_state:
    st.session_state.dat_sampling_rates = []
if 'dat_y_units' not in st.session_state:
    st.session_state.dat_y_units = []
if 'dat_channel_results' not in st.session_state:
    st.session_state.dat_channel_results = {}
if 'current_channel' not in st.session_state:
    st.session_state.current_channel = 0
if 'spike_classifications' not in st.session_state:
    st.session_state.spike_classifications = {}
if 'dat_sampling_rate' not in st.session_state:
    st.session_state.dat_sampling_rate = 10000.0
if 'original_dat_path' not in st.session_state:
    st.session_state.original_dat_path = ""
if 'file_loaded' not in st.session_state:
    st.session_state.file_loaded = False

# ====================== Core Algorithm Functions (Unchanged) ======================

def create_taper_weights(length: int, taper_left: int, taper_right: int) -> np.ndarray:
    """Create taper weight array using cosine window for smooth transition"""
    weights = np.ones(length)
    
    if taper_left > 0 and taper_left <= length:
        left_taper_weights = 0.5 * (1 - np.cos(np.pi * np.arange(taper_left) / taper_left))
        weights[:taper_left] = left_taper_weights
    
    if taper_right > 0 and taper_right <= length:
        right_taper_weights = 0.5 * (1 + np.cos(np.pi * np.arange(1, taper_right + 1) / taper_right))
        weights[-taper_right:] = right_taper_weights
    
    return weights


def constrained_polynomial_fit(
    signal: np.ndarray,
    core_start: int,
    core_end: int,
    extend_pts: int = 100,
    degree: int = 2,
    boundary_weight: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Polynomial fitting with boundary constraints for continuity"""
    n = len(signal)
    fit_start = max(0, core_start - extend_pts)
    fit_end = min(n - 1, core_end + extend_pts)
    
    x_full = np.arange(fit_start, fit_end + 1)
    y_full = signal[fit_start:fit_end + 1].copy()
    
    weights = np.ones_like(y_full, dtype=float)
    core_mask = (x_full >= core_start) & (x_full <= core_end)
    weights[core_mask] = 0.1
    
    transition_left = (x_full >= fit_start) & (x_full < core_start)
    transition_right = (x_full > core_end) & (x_full <= fit_end)
    if np.any(transition_left):
        n_left = np.sum(transition_left)
        weights[transition_left] = boundary_weight * np.linspace(0.5, 1.0, n_left)
    if np.any(transition_right):
        n_right = np.sum(transition_right)
        weights[transition_right] = boundary_weight * np.linspace(1.0, 0.5, n_right)
    
    if fit_start < core_start:
        weights[0] = boundary_weight * 2
    if core_end < fit_end:
        weights[-1] = boundary_weight * 2
    
    coeffs = np.polyfit(x_full, y_full, deg=degree, w=weights)
    trend_full = np.polyval(coeffs, x_full)
    
    if np.any(core_mask):
        local_median = np.median(y_full[core_mask] - trend_full[core_mask])
    else:
        local_median = 0
    
    corrected_full = y_full - trend_full + local_median
    core_slice = slice(core_start - fit_start, core_end - fit_start + 1)
    corrected_core = corrected_full[core_slice]
    
    return corrected_core, trend_full


def _baseline_connect_correction(
    signal: np.ndarray,
    core_start: int,
    core_end: int,
    fs: float,
    base_window_ms: float = 50.0,
    fit_degree: int = 1,
    smooth_taper_ms: float = 2.0
) -> np.ndarray:
    """Remove slow drift artifacts by connecting baseline before and after"""
    n = len(signal)
    base_pts = int(base_window_ms * fs / 1000)
    taper_pts = max(1, int(smooth_taper_ms * fs / 1000))
    
    # Pre-baseline window
    pre_start = max(0, core_start - base_pts)
    pre_end = core_start - 1
    if pre_end <= pre_start:
        pre_baseline = np.median(signal[max(0, core_start-10):core_start])
    else:
        x_pre = np.arange(pre_start, pre_end + 1)
        y_pre = signal[pre_start:pre_end + 1]
        if fit_degree == 0 or len(y_pre) < 2:
            pre_baseline = np.median(y_pre)
        else:
            coeff = np.polyfit(x_pre, y_pre, deg=1)
            pre_baseline = np.polyval(coeff, core_start)
    
    # Post-baseline window
    post_start = core_end + 1
    post_end = min(n - 1, core_end + base_pts)
    if post_end <= post_start:
        post_baseline = np.median(signal[core_end:min(n-1, core_end+10)])
    else:
        x_post = np.arange(post_start, post_end + 1)
        y_post = signal[post_start:post_end + 1]
        if fit_degree == 0 or len(y_post) < 2:
            post_baseline = np.median(y_post)
        else:
            coeff = np.polyfit(x_post, y_post, deg=1)
            post_baseline = np.polyval(coeff, core_end)
    
    # Generate trend line
    core_length = core_end - core_start + 1
    x_core = np.arange(core_start, core_end + 1)
    trend = np.linspace(pre_baseline, post_baseline, core_length)
    
    # Correction
    corrected_core = signal[core_start:core_end+1] - trend
    global_median = np.median(signal)
    trend_median = np.median(trend)
    corrected_core = corrected_core + (global_median - trend_median)
    
    # Boundary smoothing
    if taper_pts > 0 and core_length > 2 * taper_pts:
        left_taper = create_taper_weights(core_length, taper_pts, 0)
        right_taper = create_taper_weights(core_length, 0, taper_pts)
        blended = corrected_core.copy()
        
        if taper_pts > 0:
            original_left = signal[core_start:core_start+taper_pts]
            blended[:taper_pts] = (1 - 0.1*left_taper[:taper_pts]) * corrected_core[:taper_pts] + \
                                  0.1*left_taper[:taper_pts] * original_left
        
        if taper_pts > 0:
            original_right = signal[core_end-taper_pts+1:core_end+1]
            blended[-taper_pts:] = (1 - 0.1*right_taper[-taper_pts:]) * corrected_core[-taper_pts:] + \
                                   0.1*right_taper[-taper_pts:] * original_right
        
        return blended
    
    return corrected_core


def apply_tapered_correction(
    signal: np.ndarray,
    artifact_mask: np.ndarray,
    correction_func: Callable[[np.ndarray, int, int], np.ndarray],
    taper_pts: int = 50,
    preserve_edges: bool = True,
    aggressive_taper: bool = False
) -> np.ndarray:
    """Apply correction with gradual transition for smooth boundaries"""
    n = len(signal)
    result = signal.copy()
    
    labeled, n_features = label(artifact_mask)
    
    for region_id in range(1, n_features + 1):
        region_mask = (labeled == region_id)
        region_idx = np.where(region_mask)[0]
        if len(region_idx) == 0:
            continue
        
        core_start, core_end = region_idx[0], region_idx[-1]
        trans_start = max(0, core_start - taper_pts)
        trans_end = min(n - 1, core_end + taper_pts)
        
        if trans_end <= trans_start:
            continue
        
        full_segment = signal[trans_start:trans_end + 1].copy()
        full_length = len(full_segment)
        
        try:
            corrected_core = correction_func(signal, core_start, core_end)
        except:
            continue
        
        core_length = core_end - core_start + 1
        if len(corrected_core) != core_length:
            continue
        
        corrected_full = full_segment.copy()
        core_in_segment_start = core_start - trans_start
        core_in_segment_end = core_end - trans_start
        
        corrected_full[core_in_segment_start:core_in_segment_end + 1] = corrected_core
        
        if aggressive_taper:
            hard_taper = min(5, taper_pts // 4)
            weights = np.ones(full_length)
            
            left_taper_len = min(hard_taper, core_in_segment_start)
            if left_taper_len > 0:
                left_weights = 0.5 * (1 - np.cos(np.pi * np.arange(left_taper_len) / left_taper_len))
                weights[:left_taper_len] = left_weights
            
            right_taper_len = min(hard_taper, full_length - core_in_segment_end - 1)
            if right_taper_len > 0:
                right_weights = 0.5 * (1 + np.cos(np.pi * np.arange(1, right_taper_len + 1) / right_taper_len))
                weights[-right_taper_len:] = right_weights
        else:
            weights = np.ones(full_length)
            
            left_taper = min(taper_pts, core_in_segment_start)
            if left_taper > 0:
                left_weights = 0.5 * (1 - np.cos(np.pi * np.arange(left_taper) / left_taper))
                if len(left_weights) < core_in_segment_start:
                    left_weights = np.concatenate([np.zeros(core_in_segment_start - left_taper), left_weights])
                weights[:core_in_segment_start] = left_weights[:core_in_segment_start]
            
            right_taper = min(taper_pts, full_length - core_in_segment_end - 1)
            if right_taper > 0:
                right_start = core_in_segment_end + 1
                right_weights = 0.5 * (1 + np.cos(np.pi * np.arange(1, right_taper + 1) / right_taper))
                if len(right_weights) < full_length - right_start:
                    right_weights = np.concatenate([right_weights, np.zeros(full_length - right_start - right_taper)])
                weights[right_start:] = right_weights[:full_length - right_start]
        
        blended = (1 - weights) * full_segment + weights * corrected_full
        result[trans_start:trans_end + 1] = blended
    
    return result


def remove_stimulus_artifact_v2(
    signal: np.ndarray,
    fs: float,
    threshold_factor: float = 5.0,
    min_width_ms: float = 1.0,
    buffer_ms: float = 2.0,
    taper_ms: float = 5.0,
    fill_method: Literal["baseline_correct", "linear_fit", "highpass_filter", "constrained_fit", "baseline_connect"] = "constrained_fit",
    use_manual: bool = False,
    manual_start_s: float = 2.0,
    manual_end_s: float = 3.0,
    visualize: bool = False,
    debug_info: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[plt.Figure]]:
    """Refactored artifact removal with smooth boundary transition + baseline connection"""
    signal_clean = signal.copy().astype(np.float64)
    n = len(signal_clean)
    
    if n < MIN_SIGNAL_LENGTH:
        if debug_info is not None:
            debug_info.update({'error': 'Signal too short'})
        return signal_clean, np.zeros_like(signal_clean, dtype=bool), None
    
    buffer_pts = max(BUFFER_MIN, int(buffer_ms * fs / 1000))
    taper_pts = max(1, int(taper_ms * fs / 1000))
    
    artifact_mask_total = np.zeros_like(signal_clean, dtype=bool)
    processing_regions = []
    
    if use_manual:
        core_start = int(np.clip(manual_start_s * fs, 0, n - 1))
        core_end = int(np.clip(manual_end_s * fs, core_start + 1, n - 1))
        
        if core_end <= core_start:
            if debug_info is not None:
                debug_info.update({'error': 'Invalid manual artifact region'})
            return signal_clean, artifact_mask_total, None
        
        trans_start = max(0, core_start - taper_pts)
        trans_end = min(n - 1, core_end + taper_pts)
        artifact_mask_total[trans_start:trans_end + 1] = True
        
        processing_regions.append({
            'type': 'manual',
            'core': (core_start, core_end),
            'transition': (trans_start, trans_end)
        })
        
        if fill_method == "constrained_fit":
            corrected_core, trend_full = constrained_polynomial_fit(
                signal_clean, core_start, core_end, extend_pts=300, degree=2, boundary_weight=20.0
            )
            
            full_segment = signal_clean[trans_start:trans_end + 1].copy()
            corrected_full = full_segment.copy()
            core_in_start = core_start - trans_start
            core_in_end = core_end - trans_start
            
            if (0 <= core_in_start <= core_in_end < len(corrected_full) and 
                len(corrected_core) == core_in_end - core_in_start + 1):
                corrected_full[core_in_start:core_in_end+1] = corrected_core
                
                # 使用修正后的 create_taper_weights
                left_taper_len = min(core_in_start, taper_pts)
                right_taper_len = min(len(full_segment) - core_in_end - 1, taper_pts)
                
                weights = create_taper_weights(len(full_segment), left_taper_len, right_taper_len)
                weights = 0.3 + 0.7 * weights  # 增加基础权重避免完全替换
                
                blended = (1 - weights) * full_segment + weights * corrected_full
                signal_clean[trans_start:trans_end + 1] = blended
        elif fill_method == "baseline_connect":
            corrected_core = _baseline_connect_correction(
                signal_clean, core_start, core_end, fs
            )
            signal_clean[core_start:core_end+1] = corrected_core
        elif fill_method == "baseline_correct":
            # 简单的基线校正
            pre_baseline = np.median(signal_clean[max(0, core_start-100):core_start])
            post_baseline = np.median(signal_clean[core_end:min(n, core_end+100)])
            baseline = np.linspace(pre_baseline, post_baseline, core_end - core_start + 1)
            signal_clean[core_start:core_end+1] = signal_clean[core_start:core_end+1] - baseline
        elif fill_method == "linear_fit":
            # 线性拟合插值
            pre_val = np.mean(signal_clean[max(0, core_start-20):core_start])
            post_val = np.mean(signal_clean[core_end:min(n, core_end+20)])
            linear_interp = np.linspace(pre_val, post_val, core_end - core_start + 1)
            signal_clean[core_start:core_end+1] = linear_interp
        elif fill_method == "highpass_filter":
            # 高通滤波器移除低频趋势
            from scipy.signal import butter, filtfilt
            nyq = 0.5 * fs
            cutoff = HP_CUTOFF / nyq
            b, a = butter(HP_ORDER, cutoff, btype='high', analog=False)
            filtered_segment = filtfilt(b, a, signal_clean[core_start:core_end+1])
            signal_clean[core_start:core_end+1] = filtered_segment

    else:
        # 自动检测伪迹区域
        # 计算阈值
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        threshold = median + threshold_factor * mad
        
        # 检测超过阈值的点
        above_threshold = signal > threshold
        
        # 找到连续的区域
        diff = np.diff(np.concatenate(([0], above_threshold.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        
        # 过滤掉太小的区域
        min_width_pts = int(min_width_ms * fs / 1000)
        valid_regions = []
        for start, end in zip(starts, ends):
            if end - start + 1 >= min_width_pts:
                # 添加缓冲区
                buf_start = max(0, start - buffer_pts)
                buf_end = min(n - 1, end + buffer_pts)
                valid_regions.append((buf_start, buf_end))
        
        # 处理每个有效区域
        for start, end in valid_regions:
            artifact_mask_total[start:end+1] = True
            processing_regions.append({
                'type': 'auto',
                'core': (start, end),
                'transition': (start, end)
            })
            
            if fill_method == "constrained_fit":
                corrected_core, trend_full = constrained_polynomial_fit(
                    signal_clean, start, end, extend_pts=300, degree=2, boundary_weight=20.0
                )
                signal_clean[start:end+1] = corrected_core
            elif fill_method == "baseline_connect":
                corrected_core = _baseline_connect_correction(
                    signal_clean, start, end, fs
                )
                signal_clean[start:end+1] = corrected_core
            elif fill_method == "baseline_correct":
                pre_baseline = np.median(signal_clean[max(0, start-100):start])
                post_baseline = np.median(signal_clean[end:min(n, end+100)])
                baseline = np.linspace(pre_baseline, post_baseline, end - start + 1)
                signal_clean[start:end+1] = signal_clean[start:end+1] - baseline
            elif fill_method == "linear_fit":
                pre_val = np.mean(signal_clean[max(0, start-20):start])
                post_val = np.mean(signal_clean[end:min(n, end+20)])
                linear_interp = np.linspace(pre_val, post_val, end - start + 1)
                signal_clean[start:end+1] = linear_interp
            elif fill_method == "highpass_filter":
                from scipy.signal import butter, filtfilt
                nyq = 0.5 * fs
                cutoff = HP_CUTOFF / nyq
                b, a = butter(HP_ORDER, cutoff, btype='high', analog=False)
                filtered_segment = filtfilt(b, a, signal_clean[start:end+1])
                signal_clean[start:end+1] = filtered_segment

    # 如果需要可视化，生成图表
    fig = None
    if visualize:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 显示原始信号和处理后的信号
        t = np.arange(len(signal)) / fs
        ax1.plot(t, signal, 'b-', label='Original Signal', alpha=0.7)
        ax1.plot(t, signal_clean, 'r-', label='Processed Signal', alpha=0.8)
        
        # 标记伪迹区域
        artifact_regions = np.where(artifact_mask_total)[0]
        if len(artifact_regions) > 0:
            ax1.fill_between(t, np.min(signal), np.max(signal), 
                           where=artifact_mask_total, color='orange', alpha=0.3, label='Artifact Regions')
        
        ax1.set_title('Stimulus Artifact Removal')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 显示差异
        diff_signal = signal - signal_clean
        ax2.plot(t, diff_signal, 'g-', label='Difference (Original - Processed)', linewidth=0.8)
        ax2.set_title('Processing Difference')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Δ Amplitude')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()

    return signal_clean, artifact_mask_total, fig


def apply_filter(signal, fs, filter_type='none', cutoff=3000, order=2):
    """Apply filter to signal"""
    if filter_type == 'none' or len(signal) < 10:
        return signal.copy()

    nyq = 0.5 * fs
    try:
        if filter_type == 'low':
            normal_cutoff = float(cutoff) / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
        elif filter_type == 'high':
            normal_cutoff = float(cutoff) / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
        elif filter_type == 'band':
            low_cut, high_cut = map(float, cutoff.split(','))
            normal_cutoff = [low_cut / nyq, high_cut / nyq]
            b, a = butter(order, normal_cutoff, btype='band', analog=False)
        else:
            return signal.copy()

        return filtfilt(b, a, signal, padtype='constant', padlen=50)

    except:
        return signal.copy()


def detect_spikes_algo(signal, fs, threshold_ratio=5, min_gap=2, baseline_window=200, direction='down'):
    """Spike detection algorithm"""
    nan_mask = np.isnan(signal)
    if np.any(nan_mask):
        labeled, n_seg = label(~nan_mask)
        spikes_list = []
        corrected_total = signal.copy()
        for seg_id in range(1, n_seg+1):
            seg_mask = (labeled == seg_id)
            seg_signal = signal[seg_mask]
            if len(seg_signal) < baseline_window:
                continue
            seg_spikes, seg_corrected = _detect_spikes_algo_segment(
                seg_signal, fs, threshold_ratio, min_gap, baseline_window, direction)
            indices = np.where(seg_mask)[0]
            spikes_list.append(indices[seg_spikes.astype(int)])
            corrected_total[seg_mask] = seg_corrected
        spikes = np.concatenate(spikes_list) if spikes_list else np.array([])
        return spikes, corrected_total
    else:
        return _detect_spikes_algo_segment(signal, fs, threshold_ratio, min_gap, baseline_window, direction)


def _detect_spikes_algo_segment(signal, fs, threshold_ratio=5, min_gap=2, 
                               baseline_window=200, direction='down'):
    """Core spike detection algorithm"""
    try:
        kernel_size = baseline_window if baseline_window % 2 == 1 else baseline_window + 1
        kernel_size = max(KERNEL_MIN, kernel_size)
        baseline_signal = medfilt(signal, kernel_size=kernel_size)
        corrected_signal = signal - baseline_signal
        
        median = np.median(corrected_signal)
        mad = np.median(np.abs(corrected_signal - median))
        mad = max(mad, 1e-6)
        
        if direction == 'down':
            threshold = median - threshold_ratio * mad
            condition = corrected_signal < threshold
            extrema_func = np.argmin
        else:
            threshold = median + threshold_ratio * mad
            condition = corrected_signal > threshold
            extrema_func = np.argmax
        
        threshold_indices = np.where(condition)[0]
        if len(threshold_indices) == 0:
            return np.array([]), corrected_signal
        
        diffs = np.diff(threshold_indices)
        region_starts = np.where(diffs > 2)[0] + 1
        spike_candidates = threshold_indices[np.concatenate(([0], region_starts))]
        
        peak_window_pts = int(PEAK_WINDOW_MS * fs / 1000)
        peak_window_pts = max(MIN_GAP_MIN_PTS, peak_window_pts)
        
        locs = []
        for pos in spike_candidates:
            start_idx = max(0, pos)
            end_idx = min(len(corrected_signal), pos + peak_window_pts)
            window = corrected_signal[start_idx:end_idx]
            if len(window) > 0:
                extreme_idx = extrema_func(window)
                locs.append(start_idx + extreme_idx)
        
        locs = np.array(locs)
        if len(locs) == 0:
            return np.array([]), corrected_signal
        
        locs.sort()
        min_gap_pts = int(min_gap * fs / 1000)
        min_gap_pts = max(MIN_GAP_MIN_PTS, min_gap_pts)
        
        if len(locs) > 1:
            gaps = np.diff(locs)
            valid_gaps = np.concatenate(([True], gaps >= min_gap_pts))
            final_locs = locs[valid_gaps]
        else:
            final_locs = locs
        
        return final_locs, corrected_signal
    
    except Exception as e:
        st.error(f"Spike detection error: {e}")
        return np.array([]), signal


def extract_spike_waveforms(signal, spike_indices, window_size=30, return_indices=False):
    """Extract spike waveforms"""
    if len(spike_indices) == 0:
        waveforms = np.empty((0, window_size))
        if return_indices:
            return waveforms, np.array([], dtype=int)
        return waveforms
    
    waveforms = []
    valid_indices = []
    half_window = window_size // 2
    
    for pos, idx in enumerate(spike_indices):
        start = max(0, int(idx) - half_window)
        end = min(len(signal), int(idx) + half_window)
        
        if end - start < window_size:
            # 如果窗口不够大，进行零填充
            waveform = signal[start:end]
            padded_waveform = np.zeros(window_size)
            center_start = (window_size - len(waveform)) // 2
            center_end = center_start + len(waveform)
            padded_waveform[center_start:center_end] = waveform
            waveform = padded_waveform
        else:
            waveform = signal[start:end]
        
        waveforms.append(waveform)
        valid_indices.append(pos)
        
    waveforms = np.array(waveforms)
    if return_indices:
        return waveforms, np.array(valid_indices, dtype=int)
    return waveforms


def calculate_sta(waveforms):
    """Calculate spike-triggered average"""
    if len(waveforms) == 0:
        return None
    return np.mean(waveforms, axis=0)


# ====================== DAT File Reading Functions ======================

def extract_all_traces(tree):
    """Extract all trace records from tree structure"""
    traces = []
    if isinstance(tree, dict) and 'TrLabel' in tree:
        traces.append(tree)
    elif isinstance(tree, (list, tuple)):
        for item in tree:
            traces.extend(extract_all_traces(item))
    return traces


def get_bundle_header(fh):
    """Get bundle file header information"""
    original_pos = fh.tell()  # 保存当前位置
    try:
        fh.seek(0, 0)
        signature_bytes = fh.read(8)
        if len(signature_bytes) < 8:
            raise ValueError("File is too short to be a valid DAT file")
        signature = signature_bytes.decode('utf-8', errors='ignore').strip('\x00')
        bundle = {}
        little_endian_flag = None
        is_bundled = False
        
        if signature == 'DATA':
            bundle['oSignature'] = signature
            bundle['oVersion'] = None
            bundle['oTime'] = None
            bundle['oItems'] = None
            bundle['oIsLittleEndian'] = None
            bundle['oBundleItems'] = []
            bundle['BundleHeaderSize'] = 0
            is_bundled = False
        elif signature in ['DAT1', 'DAT2']:
            bundle['oSignature'] = signature
            version_bytes = fh.read(32)
            if len(version_bytes) < 32:
                raise ValueError("Incomplete version field in DAT file")
            bundle['oVersion'] = version_bytes.decode('utf-8', errors='ignore').strip('\x00')
            time_bytes = fh.read(8)
            if len(time_bytes) < 8:
                raise ValueError("Incomplete time field in DAT file")
            bundle['oTime'] = struct.unpack('<d', time_bytes)[0]
            
            bundle['oItems'] = struct.unpack('<i', fh.read(4))[0]
            bundle['oIsLittleEndian'] = struct.unpack('<B', fh.read(1))[0] != 0
            little_endian_flag = bundle['oIsLittleEndian']
            bundle['BundleHeaderSize'] = 256
            
            if signature == 'DAT1':
                bundle['oBundleItems'] = []
                is_bundled = False
            elif signature == 'DAT2':
                fh.seek(64, 0)
                bundle_items = []
                for _ in range(12):
                    item = {
                        'oStart': struct.unpack('<i', fh.read(4))[0],
                        'oLength': struct.unpack('<i', fh.read(4))[0],
                        'oExtension': fh.read(8).decode('utf-8', errors='ignore').strip('\x00'),
                        'BundleItemSize': 16
                    }
                    bundle_items.append(item)
                bundle['oBundleItems'] = bundle_items
                is_bundled = True
        else:
            raise ValueError(f"Unsupported file format: {signature}")
    
        return bundle, little_endian_flag, is_bundled

    finally:
        fh.seek(original_pos)  # 恢复原始位置


def get_tree(fh, sizes, position, endian):
    """Get tree structure"""
    tree_result = get_tree_reentrant(fh, [], sizes, 0, position, 0, endian)
    if isinstance(tree_result, tuple) and len(tree_result) == 3:
        tree, _, _ = tree_result
    else:
        tree = []
    return tree


def get_tree_reentrant(fh, tree, sizes, level, position, counter, endian):
    """Recursively get tree structure"""
    tree, position, counter, nchild = get_one_level(fh, tree, sizes, level, position, counter, endian)
    for _ in range(nchild):
        tree, position, counter = get_tree_reentrant(fh, tree, sizes, level + 1, position, counter, endian)
    return tree, position, counter


def get_one_level(fh, tree, sizes, level, position, counter, endian):
    """Get one level of tree structure"""
    fh.seek(position, 0)
    rec, counter = get_one_record(fh, level, counter, endian)
    
    while len(tree) <= counter:
        tree.append([None] * (len(sizes) + 1))
    
    tree[counter][level + 1] = rec
    position += sizes[level]
    fh.seek(position, 0)
    nchild = struct.unpack(f'{endian}i', fh.read(4))[0]
    position = fh.tell()
    
    return tree, position, counter, nchild


def get_one_record(fh, level, counter, endian):
    """Get one record"""
    counter += 1
    
    if level == 0:
        rec = get_root(fh, endian)
    elif level == 1:
        rec = get_group(fh, endian)
    elif level == 2:
        rec = get_series(fh, endian)
    elif level == 3:
        rec = get_sweep(fh, endian)
    elif level == 4:
        rec = get_trace(fh, endian)
    else:
        raise ValueError(f"Unexpected level: {level}")
    
    return rec, counter


def get_root(fh, endian):
    """Get root record"""
    root = {}
    root['RoVersion'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    root['RoMark'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    root['RoVersionName'] = fh.read(32).decode('utf-8', errors='ignore').strip('\x00')
    root['RoAuxFileName'] = fh.read(80).decode('utf-8', errors='ignore').strip('\x00')
    root['RoRootText'] = fh.read(400).decode('utf-8', errors='ignore').strip('\x00')
    root['RoStartTime'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    root['RoMaxSamples'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    root['RoCRC'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    root['RoFeatures'] = struct.unpack(f'{endian}h', fh.read(2))[0]
    root['RoFiller1'] = struct.unpack(f'{endian}h', fh.read(2))[0]
    root['RoFiller2'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    
    return root


def get_group(fh, endian):
    """Get group record"""
    group = {}
    group['GrMark'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    group['GrLabel'] = fh.read(32).decode('utf-8', errors='ignore').strip('\x00')
    group['GrText'] = fh.read(80).decode('utf-8', errors='ignore').strip('\x00')
    group['GrExperimentNumber'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    group['GrGroupCount'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    group['GrCRC'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    
    return group


def get_series(fh, endian):
    """Get series record"""
    series = {}
    series['SeMark'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    series['SeLabel'] = fh.read(32).decode('utf-8', errors='ignore').strip('\x00')
    series['SeComment'] = fh.read(80).decode('utf-8', errors='ignore').strip('\x00')
    series['SeSeriesCount'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    series['SeNumbersw'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    series['SeAmplStateOffset'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    series['SeAmplStateSeries'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    series['SeSeriesType'] = struct.unpack(f'{endian}B', fh.read(1))[0]
    series['SeFiller1'] = struct.unpack(f'{endian}B', fh.read(1))[0]
    series['SeFiller2'] = struct.unpack(f'{endian}B', fh.read(1))[0]
    series['SeFiller3'] = struct.unpack(f'{endian}B', fh.read(1))[0]
    series['SeTime'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    series['SePageWidth'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    series['SeSwUserParamDescr'] = []
    for _ in range(4):
        descr = {
            'Name': fh.read(32).decode('utf-8', errors='ignore').strip('\x00'),
            'Unit': fh.read(8).decode('utf-8', errors='ignore').strip('\x00')
        }
        series['SeSwUserParamDescr'].append(descr)
    series['SeFiller4'] = [struct.unpack(f'{endian}B', fh.read(1))[0] for _ in range(32)]
    series['SeSeUserParams'] = [struct.unpack(f'{endian}d', fh.read(8))[0] for _ in range(4)]
    series['SeUsername'] = fh.read(80).decode('utf-8', errors='ignore').strip('\x00')
    series['SeSeUserParamDescr'] = []
    for _ in range(4):
        descr = {
            'Name': fh.read(32).decode('utf-8', errors='ignore').strip('\x00'),
            'Unit': fh.read(8).decode('utf-8', errors='ignore').strip('\x00')
        }
        series['SeSeUserParamDescr'].append(descr)
    series['SeFiller5'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    series['SeCRC'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    return series


def get_sweep(fh, endian):
    """Get sweep record"""
    sweep = {}
    sweep['SwMark'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    sweep['SwLabel'] = fh.read(32).decode('utf-8', errors='ignore').strip('\x00')
    sweep['SwAuxDataFileOffset'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    sweep['SwStimCount'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    sweep['SwSweepCount'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    sweep['SwTime'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    sweep['SwTimer'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    sweep['SwSwUserParams'] = [struct.unpack(f'{endian}d', fh.read(8))[0] for _ in range(4)]
    sweep['SwTemperature'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    sweep['SwOldIntSol'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    sweep['SwOldExtSol'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    sweep['SwDigitalIn'] = struct.unpack(f'{endian}h', fh.read(2))[0]
    sweep['SwSweepKind'] = struct.unpack(f'{endian}h', fh.read(2))[0]
    sweep['SwFiller1'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    sweep['SwMarkers'] = [struct.unpack(f'{endian}d', fh.read(8))[0] for _ in range(4)]
    sweep['SwFiller2'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    sweep['SwCRC'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    return sweep


def get_trace(fh, endian):
    """Get trace record"""
    trace = {}
    trace['TrMark'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    trace['TrLabel'] = fh.read(32).decode('utf-8', errors='ignore').strip('\x00')
    trace['TrTraceCount'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    trace['TrData'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    trace['TrDataPoints'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    trace['TrInternalSolution'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    trace['TrAverageCount'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    trace['TrLeakCount'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    trace['TrLeakTraces'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    trace['TrDataKind'] = struct.unpack(f'{endian}H', fh.read(2))[0]
    trace['TrFiller1'] = struct.unpack(f'{endian}h', fh.read(2))[0]
    trace['TrRecordingMode'] = struct.unpack(f'{endian}B', fh.read(1))[0]
    trace['TrAmplIndex'] = struct.unpack(f'{endian}B', fh.read(1))[0]
    trace['TrDataFormat'] = struct.unpack(f'{endian}B', fh.read(1))[0]
    trace['TrDataAbscissa'] = struct.unpack(f'{endian}B', fh.read(1))[0]
    trace['TrDataScaler'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrTimeOffset'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrZeroData'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrYUnit'] = fh.read(8).decode('utf-8', errors='ignore').strip('\x00')
    trace['TrXInterval'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrXStart'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrXUnit'] = fh.read(8).decode('utf-8', errors='ignore').strip('\x00')
    trace['TrYRange'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrYOffset'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrBandwidth'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrPipetteResistance'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrCellPotential'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrSealResistance'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrCSlow'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrGSeries'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrRsValue'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrGLeak'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrMConductance'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrLinkDAChannel'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    trace['TrValidYrange'] = struct.unpack(f'{endian}B', fh.read(1))[0] != 0
    trace['TrAdcMode'] = struct.unpack(f'{endian}B', fh.read(1))[0]
    trace['TrAdcChannel'] = struct.unpack(f'{endian}h', fh.read(2))[0]
    trace['TrYmin'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrYmax'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrSourceChannel'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    trace['TrExternalSolution'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    trace['TrCM'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrGM'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrPhase'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrDataCRC'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    trace['TrCRC'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    trace['TrGS'] = struct.unpack(f'{endian}d', fh.read(8))[0]
    trace['TrSelfChannel'] = struct.unpack(f'{endian}i', fh.read(4))[0]
    trace['TrFiller2'] = struct.unpack(f'{endian}h', fh.read(2))[0]
    return trace


def local_import_group(datafile, trace, channel_number, endian, data_start):
    """Import channel data from DAT file"""
    traces = []
    file_size = os.path.getsize(datafile)
    if not (isinstance(trace, dict) and 'TrLabel' in trace):
        return []
    try:
        offset = int(trace['TrData']) + int(data_start)
        npts = int(trace['TrDataPoints'])
        data_format = trace.get('TrDataFormat', 2)
        
        if data_format == 0:
            dtype = np.int16
            dtype_size = 2
        elif data_format == 1:
            dtype = np.int32
            dtype_size = 4
        elif data_format == 2:
            dtype = np.float32
            dtype_size = 4
        elif data_format == 3:
            dtype = np.float64
            dtype_size = 8
        else:
            dtype = np.float32
            dtype_size = 4
            
        if offset < 0 or offset >= file_size:
            return []
        if npts <= 0 or npts > MAX_DATA_POINTS:
            return []
        if (offset + npts * dtype_size) > file_size:
            return []
            
        with open(datafile, 'rb') as f:
            f.seek(offset, 0)
            arr = np.fromfile(f, dtype=dtype, count=npts)
            
        scaler = trace.get('TrDataScaler', 1.0)
        zero = trace.get('TrZeroData', 0.0)
        arr = arr * scaler + zero
        traces.append(arr)
        
    except Exception as e:
        st.error(f"Error importing channel data: {e}")
        return []
        
    return [traces[0]] if traces else []


def import_patch_master_data(dat_file_path):
    """Main function to import PatchMaster DAT file"""
    SUPPORTED_SIGNATURES = {'DATA', 'DAT1', 'DAT2'}
    
    st.info(f'Parsing DAT file: {dat_file_path}')
    
    if not os.path.exists(dat_file_path):
        error_msg = f'File does not exist: {dat_file_path}'
        st.error(error_msg)
        return [], [], [], []
    
    try:
        with open(dat_file_path, 'rb') as f:
            file_head = f.read(64)
        file_signature = file_head[:4].decode('utf-8', errors='ignore')
        if file_signature not in SUPPORTED_SIGNATURES:
            error_msg = f'Unsupported file format: {file_signature} (only DATA/DAT1/DAT2 supported)'
            st.error(error_msg)
            return [], [], [], []
    except Exception as e:
        error_msg = f'Failed to read file header: {str(e)}'
        st.error(error_msg)
        return [], [], [], []
    
    try:
        file_dir, file_name = os.path.split(dat_file_path)
        file_name_no_ext, file_ext = os.path.splitext(file_name)
        pul_file_path = os.path.join(file_dir, f"{file_name_no_ext}.pul")
        
        endian = '<'
        with open(dat_file_path, 'rb') as fh:
            bundle, little_endian_flag, is_bundled = get_bundle_header(fh)
        
        if little_endian_flag is not None and not little_endian_flag:
            endian = '>'
            with open(dat_file_path, 'rb') as fh:
                bundle, _, is_bundled = get_bundle_header(fh)
        
        pul_fh = None
        data_start = 0
        try:
            if is_bundled:
                ext_list = [item['oExtension'] for item in bundle.get('oBundleItems', [])]
                pul_idx = next((i for i, ext in enumerate(ext_list) if ext == '.pul'), None)
                if pul_idx is not None:
                    data_start = bundle['oBundleItems'][pul_idx]['oStart']
                pul_fh = open(dat_file_path, 'rb')
                pul_fh.seek(data_start, 0)
            else:
                if not os.path.exists(pul_file_path):
                    st.error('Corresponding .pul file not found')
                    return [], [], [], []
                pul_fh = open(pul_file_path, 'rb')
            
            magic = pul_fh.read(4).decode('utf-8', errors='ignore')
            levels = struct.unpack(f'{endian}i', pul_fh.read(4))[0]
            sizes = [struct.unpack(f'{endian}i', pul_fh.read(4))[0] for _ in range(levels)]
            position = pul_fh.tell()
            tree = get_tree(pul_fh, sizes, position, endian)
        finally:
            if pul_fh:
                pul_fh.close()
        
        all_traces = extract_all_traces(tree)
        if not all_traces:
            st.warning('No valid channel data parsed')
            return [], [], [], []
        
        if is_bundled:
            ext_list = [item['oExtension'] for item in bundle.get('oBundleItems', [])]
            dat_idx = next((i for i, ext in enumerate(ext_list) if ext == '.dat'), None)
            if dat_idx is not None:
                data_start = bundle['oBundleItems'][dat_idx]['oStart']
        else:
            data_start = bundle.get('BundleHeaderSize', 0)
        
        data_channels = []
        trace_labels = []
        sampling_rates = []
        y_units = []
        
        for channel_idx, trace in enumerate(all_traces):
            channel_data = local_import_group(
                dat_file_path, trace, channel_idx + 1, endian, data_start
            )
            if channel_data:
                data_channels.extend(channel_data)
                trace_label = trace.get('TrLabel', f'Channel {channel_idx + 1}')
                trace_labels.append(trace_label)
                
                x_interval = trace.get('TrXInterval')
                sampling_rate = 1.0 / x_interval if (x_interval and x_interval > 0) else None
                sampling_rates.append(sampling_rate)
                
                y_unit = trace.get('TrYUnit', '')
                y_units.append(y_unit)
        
        if not data_channels:
            st.error('No valid channel data parsed')
            return [], [], [], []
        
        st.success(f'Successfully loaded {len(data_channels)} channels from {os.path.basename(dat_file_path)}')
        return data_channels, trace_labels, sampling_rates, y_units
        
    except Exception as e:
        error_msg = f'DAT file parsing error: {str(e)}'
        st.error(error_msg)
        return [], [], [], []


# ====================== Streamlit UI Components ======================

def main():
    """Main function: Streamlit application interface"""
    st.title("🧠 Spikes Decoder Web v0.1.0")
    st.markdown("---")
    
    # Sidebar: File upload and basic settings
    with st.sidebar:
        st.header("📁 File Settings")
        
        # File upload
        uploaded_file = st.file_uploader("Select DAT File", type=['.dat'])
        
        if uploaded_file is not None and not st.session_state.file_loaded:
            with st.spinner("Loading DAT file..."):
                # Save uploaded file to temporary location
                temp_dir = tempfile.gettempdir()
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Parse DAT file
                data_channels, trace_labels, sampling_rates, y_units = import_patch_master_data(temp_file_path)
                
                if data_channels and trace_labels:
                    # Clear previous results
                    st.session_state.dat_channel_results = {}
                    st.session_state.spike_classifications = {}
                    
                    # Store data in session state
                    st.session_state.dat_channels = data_channels
                    st.session_state.dat_trace_labels = trace_labels
                    st.session_state.dat_sampling_rates = sampling_rates
                    st.session_state.dat_y_units = y_units
                    st.session_state.original_dat_path = temp_file_path
                    
                    # Set global sampling rate (use first channel's rate)
                    if sampling_rates and sampling_rates[0]:
                        st.session_state.dat_sampling_rate = float(sampling_rates[0])
                    
                    st.session_state.file_loaded = True
                    st.session_state.current_channel = 0
                    
                    # Store raw data in results
                    for i, signal in enumerate(data_channels):
                        st.session_state.dat_channel_results[i] = {'raw': signal.copy()}
        
        # Clear button
        if st.button("Clear Loaded File"):
            st.session_state.dat_channels = []
            st.session_state.dat_trace_labels = []
            st.session_state.dat_sampling_rates = []
            st.session_state.dat_y_units = []
            st.session_state.dat_channel_results = {}
            st.session_state.spike_classifications = {}
            st.session_state.original_dat_path = ""
            st.session_state.file_loaded = False
            st.session_state.current_channel = 0
            st.rerun()
        
        st.header("⚙️ Basic Settings")
        if st.session_state.dat_sampling_rates and st.session_state.dat_sampling_rates[0]:
            default_fs = float(st.session_state.dat_sampling_rates[0])
        else:
            default_fs = 10000.0
            
        st.session_state.dat_sampling_rate = st.number_input(
            "Sampling Rate (Hz)", 
            min_value=1.0, 
            max_value=100000.0, 
            value=float(st.session_state.dat_sampling_rate),
            step=100.0
        )
        
        st.header("📊 Display Settings")
        plot_mode = st.selectbox(
            "Display Mode",
            ["Paged Subplots", "Waterfall Plot", "Grid Subplots"],
            index=0
        )
        
        if plot_mode == "Paged Subplots":
            channels_per_page = st.slider("Channels per Page", 1, 20, 4)
        else:
            channels_per_page = 4
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Signal Display", "🔧 Preprocessing", "🎯 Spike Detection", "📊 Results Analysis"])
    
    with tab1:
        display_signal_tab(plot_mode, channels_per_page)
    
    with tab2:
        preprocessing_tab()
    
    with tab3:
        spike_detection_tab()
    
    with tab4:
        analysis_tab()


def display_signal_tab(plot_mode, channels_per_page):
    """Signal display tab"""
    st.header("Signal Display")
    
    if not st.session_state.dat_channels:
        st.info("Please upload a DAT file first")
        return
    
    # Display file info
    if st.session_state.original_dat_path:
        st.info(f"Loaded file: {os.path.basename(st.session_state.original_dat_path)}")
    
    # Channel selection
    channel_options = [f"{i+1}: {label}" for i, label in enumerate(st.session_state.dat_trace_labels)]
    selected_channel = st.selectbox("Select Channel", channel_options, index=st.session_state.current_channel)
    ch_idx = int(selected_channel.split(':')[0]) - 1
    st.session_state.current_channel = ch_idx
    
    # Display current channel signal
    fig, ax = plt.subplots(figsize=(12, 4))
    fs = st.session_state.dat_sampling_rate
    signal = st.session_state.dat_channels[ch_idx]
    
    # Handle very long signals by downsampling for display
    display_signal = signal.copy()
    if len(signal) > 100000:
        step = len(signal) // 50000
        display_signal = signal[::step]
        t = np.arange(len(display_signal)) * step / fs
    else:
        t = np.arange(len(display_signal)) / fs
    
    ax.plot(t, display_signal, 'b-', linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{st.session_state.dat_trace_labels[ch_idx]} (Sampling rate: {fs:.1f} Hz)")
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Display channel info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Signal Length", f"{len(signal):,} points")
    with col2:
        duration = len(signal) / fs
    with col3:
        st.metric("Duration", f"{duration:.2f} s")
    
    # Multi-channel display
    st.subheader("Multi-Channel Display")
    
    if plot_mode == "Waterfall Plot":
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        ranges = []
        for ch in st.session_state.dat_channels:
            ch = np.asarray(ch)
            if ch.size == 0:
                ranges.append(1.0)
                continue
            p5, p95 = np.percentile(ch, [5, 95])
            r = float(p95 - p5)
            ranges.append(r if r > 0 else 1.0)
        base_offset = np.median(ranges) * 1.2
        
        max_points = 8000
        for i, raw_data in enumerate(st.session_state.dat_channels):
            raw_data = np.asarray(raw_data)
            if raw_data.size == 0:
                continue
            step = max(1, raw_data.size // max_points)
            y = raw_data[::step]
            t = (np.arange(y.size) * step) / fs
            ax2.plot(t, y + i * base_offset, lw=0.8, alpha=0.9)
        
        ax2.set_title("Multi-Channel Waterfall Plot")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Channel (Offset)")
        ax2.grid(alpha=0.2, linestyle='--')
        
        st.pyplot(fig2)
    
    else:
        n_channels = len(st.session_state.dat_channels)
        cols = 2
        rows = (n_channels + cols - 1) // cols
        fig2, axes = plt.subplots(rows, cols, figsize=(14, 3*rows))
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
        
        for i, (ax, channel) in enumerate(zip(axes, st.session_state.dat_channels)):
            if i >= n_channels:
                ax.axis('off')
                continue
            
            # Downsample for display
            display_signal = channel.copy()
            if len(channel) > 50000:
                step = len(channel) // 25000
                display_signal = channel[::step]
                t = np.arange(len(display_signal)) * step / fs
            else:
                t = np.arange(len(display_signal)) / fs
            
            ax.plot(t, display_signal, 'b-', linewidth=0.5)
            ax.set_title(f"{st.session_state.dat_trace_labels[i]}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig2)


def display_results_comparison(raw_data, processed_data, artifact_mask, fs, ch_idx):
    """Display processing results comparison"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    
    # Downsample for display
    if len(raw_data) > 100000:
        step = len(raw_data) // 50000
        raw_display = raw_data[::step]
        proc_display = processed_data[::step]
        mask_display = artifact_mask[::step]
        t = np.arange(len(raw_display)) * step / fs
    else:
        raw_display = raw_data
        proc_display = processed_data
        mask_display = artifact_mask
        t = np.arange(len(raw_display)) / fs
    
    ax1.plot(t, raw_display, 'b-', label='Original Signal', alpha=0.7, linewidth=0.8)
    ax1.plot(t, proc_display, 'r-', label='Processed', alpha=0.8, linewidth=0.8)
    ax1.set_title(f"Channel {ch_idx+1}: Signal Comparison")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if np.any(mask_display):
        ax2.fill_between(t, np.min(raw_display), np.max(raw_display), 
                        where=mask_display, color='orange', alpha=0.3, label='Artifact Region')
    ax2.plot(t, raw_display - proc_display, 'g-', label='Difference', linewidth=0.8)
    ax2.set_title("Processing Difference")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Δ Amplitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)


def preprocessing_tab():
    """Preprocessing tab"""
    st.header("Signal Preprocessing")
    
    if not st.session_state.dat_channels:
        st.info("Please upload a DAT file first")
        return
    
    # Channel selection
    channel_options = [f"{i+1}: {label}" for i, label in enumerate(st.session_state.dat_trace_labels)]
    selected_channel = st.selectbox("Select Processing Channel", channel_options, 
                                   index=st.session_state.current_channel, key="preprocess_channel")
    ch_idx = int(selected_channel.split(':')[0]) - 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Filter Settings")
        filter_type = st.selectbox("Filter Type", ["none", "low", "high", "band"], index=0)
        
        if filter_type == "band":
            col_band1, col_band2 = st.columns(2)
            with col_band1:
                low_cut = st.number_input("Low Cutoff (Hz)", 1.0, 10000.0, 300.0, step=10.0, key="low_cut")
            with col_band2:
                high_cut = st.number_input("High Cutoff (Hz)", 1.0, 10000.0, 3000.0, step=10.0, key="high_cut")
            cutoff = f"{low_cut},{high_cut}"
        elif filter_type != "none":
            cutoff = st.number_input("Cutoff Frequency (Hz)", 1.0, 10000.0, 3000.0, step=10.0, key="cutoff")
        else:
            cutoff = 3000.0
    
    with col2:
        st.subheader("Artifact Removal Settings")
        
        # Manual artifact region option
        use_manual_artifact = st.checkbox("Enable Manual Artifact Region", value=False, 
                                         help="Manually specify artifact region for processing")
        
        if use_manual_artifact:
            st.info("Manually specify artifact region (default 2-3 seconds)")
            col_manual1, col_manual2 = st.columns(2)
            with col_manual1:
                manual_start_s = st.number_input("Artifact Start Time (s)", 0.0, 100.0, 2.0, 0.1, key="manual_start")
            with col_manual2:
                manual_end_s = st.number_input("Artifact End Time (s)", 0.0, 100.0, 3.0, 0.1, key="manual_end")
        else:
            manual_start_s = 2.0
            manual_end_s = 3.0
        
        remove_stimulus = st.checkbox("Enable Artifact Removal", value=False)
        
        if remove_stimulus:
            fill_method = st.selectbox(
                "Correction Method",
                ["highpass_filter (20Hz)", "constrained_fit (Recommended - Smooth)", "baseline_connect (Recommended - Baseline Connect)", 
                 "baseline_correct", "linear_fit"],
                index=0
            )
            
            st.subheader("Auto Detection Parameters")
            threshold_factor = st.slider("Detection Threshold Factor", 1.0, 20.0, 5.0, 0.5, key="threshold_factor")
            min_width_ms = st.slider("Minimum Width (ms)", 0.1, 10.0, 1.0, 0.1, key="min_width")
            buffer_ms = st.slider("Buffer (ms)", 0.0, 20.0, 2.0, 0.5, key="buffer_ms")
            taper_ms = st.slider("Taper Transition (ms)", 0.0, 20.0, 5.0, 0.5, key="taper_ms")
    
    # Apply preprocessing button
    if st.button("Apply Preprocessing", type="primary"):
        with st.spinner("Processing..."):
            fs = st.session_state.dat_sampling_rate
            raw_data = st.session_state.dat_channels[ch_idx].copy()
            
            data = raw_data
            artifact_mask = np.zeros_like(raw_data, dtype=bool)
            viz_fig = None
            
            if remove_stimulus:
                # Parse method name
                if "baseline_connect" in fill_method:
                    fill_method_clean = "baseline_connect"
                elif "constrained_fit" in fill_method:
                    fill_method_clean = "constrained_fit"
                elif "baseline_correct" in fill_method:
                    fill_method_clean = "baseline_correct"
                elif "linear_fit" in fill_method:
                    fill_method_clean = "linear_fit"
                elif "highpass_filter" in fill_method:
                    fill_method_clean = "highpass_filter"
                else:
                    fill_method_clean = "constrained_fit"
                
                # Call v2 artifact removal function
                data, artifact_mask, viz_fig = remove_stimulus_artifact_v2(
                    signal=raw_data,
                    fs=fs,
                    threshold_factor=threshold_factor,
                    min_width_ms=min_width_ms,
                    buffer_ms=buffer_ms,
                    taper_ms=taper_ms,
                    fill_method=fill_method_clean,
                    use_manual=use_manual_artifact,
                    manual_start_s=manual_start_s,
                    manual_end_s=manual_end_s,
                    visualize=True
                )
                
                # If visualization generated, show it
                if viz_fig is not None:
                    st.subheader("Artifact Removal Details")
                    st.pyplot(viz_fig)
            
            if filter_type != "none":
                data = apply_filter(data, fs, filter_type=filter_type, cutoff=cutoff)
            
            # Save results
            if ch_idx not in st.session_state.dat_channel_results:
                st.session_state.dat_channel_results[ch_idx] = {'raw': raw_data}
            
            st.session_state.dat_channel_results[ch_idx]['preprocessed'] = data
            st.session_state.dat_channel_results[ch_idx]['artifact_mask'] = artifact_mask
            
            if 'spikes' in st.session_state.dat_channel_results[ch_idx]:
                del st.session_state.dat_channel_results[ch_idx]['spikes']
            
            st.success(f"Channel {ch_idx+1} preprocessing completed")
            
            # Display results
            display_results_comparison(raw_data, data, artifact_mask, fs, ch_idx)


def _parse_plotly_selection_points(sel_state: Any) -> List[Dict[str, Any]]:
    """Extract Plotly selection points from st.session_state value (Streamlit plotly_chart)."""
    if sel_state is None:
        return []
    if isinstance(sel_state, dict):
        if "selection" in sel_state and isinstance(sel_state["selection"], dict):
            pts = sel_state["selection"].get("points", [])
            return pts if isinstance(pts, list) else []
        if "points" in sel_state:
            pts = sel_state["points"]
            return pts if isinstance(pts, list) else []
    return []


def _resolve_spike_index_for_removal(
    ev: Dict[str, Any],
    fs: float,
    n: int,
    active_spike_indices: set,
) -> Optional[int]:
    """Map a Plotly event to a spike sample index to remove; skip clicks on the signal trace (curve 0)."""
    cn = ev.get("curveNumber")
    if cn is not None and int(cn) == 0:
        return None
    idx: Optional[int] = None
    cd = ev.get("customdata")
    if cd is not None:
        # Streamlit selection payloads may wrap customdata differently across versions.
        # We accept: scalar, list/tuple/ndarray, or dict like {"idx": 123} / {"0": 123}.
        if isinstance(cd, dict):
            if "idx" in cd:
                idx = int(cd["idx"])
            elif 0 in cd:
                idx = int(cd[0])
            elif "0" in cd:
                idx = int(cd["0"])
            elif "value" in cd:
                idx = int(cd["value"])
            else:
                # try first numeric-looking value
                for v in cd.values():
                    try:
                        idx = int(v)
                        break
                    except Exception:
                        continue
        else:
            idx = int(cd[0]) if isinstance(cd, (list, tuple, np.ndarray)) else int(cd)
    elif "x" in ev:
        idx = int(round(float(ev["x"]) * fs))
    if idx is None or not (0 <= idx < n):
        return None
    if idx in active_spike_indices:
        return idx
    if active_spike_indices:
        nearest = min(active_spike_indices, key=lambda s: abs(s - idx))
        if abs(nearest - idx) <= max(3, int(0.0005 * fs)):
            return nearest
    return None


def _pad_list_to_len(lst: List[Any], n: int, fill: Any = None) -> List[Any]:
    """Pad or truncate list to length n so DataFrame columns align."""
    lst = list(lst)
    if len(lst) >= n:
        return lst[:n]
    return lst + [fill] * (n - len(lst))


def _build_spike_detection_figure_plotly(
    data: np.ndarray,
    fs: float,
    final_spikes: np.ndarray,
    spikes_auto: np.ndarray,
    manual_spikes: List[int],
    removed_spikes: List[int],
    time_range: Tuple[float, float],
    ch_label: str,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build interactive Plotly figure: line signal + spike markers (customdata = sample index).
    Zoom/pan/reset via Plotly modebar (similar to NavigationToolbar2Tk).
    """
    n = len(data)
    t_full = np.arange(n) / fs
    mask = (t_full >= time_range[0]) & (t_full <= time_range[1])
    idx_start = int(np.searchsorted(t_full, time_range[0], side="left"))
    idx_end = min(n, int(np.searchsorted(t_full, time_range[1], side="right")) + 1)
    if idx_end <= idx_start:
        idx_start, idx_end = 0, min(n, 1)

    # Downsample long traces for performance
    max_line_pts = 200_000
    seg_len = idx_end - idx_start
    step = max(1, seg_len // max_line_pts)
    idxs = np.arange(idx_start, idx_end, step)
    t_line = idxs / fs
    y_line = data[idxs].astype(float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_line,
            y=y_line,
            mode="lines",
            name="Signal",
            line=dict(color="#1f77b4", width=1),
            hovertemplate="Time: %{x:.3f} s<br>Amplitude: %{y:.4f}<extra></extra>",
        )
    )

    spikes_a = np.asarray(spikes_auto, dtype=int).ravel()
    manual_set = set(int(x) for x in manual_spikes)
    removed_set = set(int(x) for x in removed_spikes)

    def _add_spike_trace(indices: np.ndarray, name: str, color: str, symbol: str):
        if indices.size == 0:
            return
        in_win = indices[(indices >= 0) & (indices < n)]
        in_win = in_win[(t_full[in_win] >= time_range[0]) & (t_full[in_win] <= time_range[1])]
        if in_win.size == 0:
            return
        tx = in_win.astype(float) / fs
        ty = data[in_win].astype(float)
        fig.add_trace(
            go.Scatter(
                x=tx,
                y=ty,
                mode="markers",
                name=name,
                marker=dict(color=color, size=8, symbol=symbol, line=dict(width=0.5, color="white")),
                customdata=in_win.reshape(-1, 1),
                hovertemplate="idx=%{customdata[0]}<br>t=%{x:.4f}s<extra></extra>",
            )
        )

    # Red = auto-detected only; Green = user-added; Blue X = marked removed (still visible for reference)
    orig_auto = np.array(
        [i for i in spikes_a if i not in removed_set and i not in manual_set],
        dtype=int,
    )
    orig_manual = np.array([i for i in manual_set if i not in removed_set and 0 <= i < n], dtype=int)

    removed_show = np.array([i for i in removed_set if 0 <= i < n], dtype=int)
    removed_show = removed_show[(t_full[removed_show] >= time_range[0]) & (t_full[removed_show] <= time_range[1])]

    _add_spike_trace(orig_auto, f"Auto spikes ({len(orig_auto)})", "red", "circle")
    _add_spike_trace(orig_manual, f"Manual spikes ({len(orig_manual)})", "green", "triangle-up")
    if removed_show.size:
        _add_spike_trace(removed_show, f"Marked removed ({len(removed_show)})", "blue", "x")

    fig.update_layout(
        title=f"Spike Detection Results — {ch_label}",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="closest",
        dragmode="zoom",
        xaxis=dict(range=list(time_range)),
        uirevision="spike",
    )
    fig.update_xaxes(
        rangeslider=dict(visible=False),
    )
    # Plotly modebar: zoom, pan, box zoom, reset (like Matplotlib toolbar)
    config = {
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],  # use Streamlit selection instead when needed
        "displaylogo": False,
    }
    return fig, config


def spike_detection_tab():
    """Spike detection tab"""
    st.header("Spike Detection")
    
    if not st.session_state.dat_channels:
        st.info("Please upload a DAT file first")
        return
    
    # Channel selection
    channel_options = [f"{i+1}: {label}" for i, label in enumerate(st.session_state.dat_trace_labels)]
    selected_channel = st.selectbox("Select Detection Channel", channel_options, 
                                   index=st.session_state.current_channel, key="detect_channel")
    ch_idx = int(selected_channel.split(':')[0]) - 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Detection Parameters")
        threshold_ratio = st.slider("Threshold Ratio", 1.0, 20.0, 5.0, 0.5)
        min_gap = st.slider("Minimum Interval (ms)", 0.1, 10.0, 2.0, 0.1)
        baseline_window = st.slider("Baseline Window", 10, 500, 200, 10)
        direction = st.selectbox("Peak Direction", ["down", "up"], index=0)
    
    with col2:
        st.subheader("Waveform Extraction")
        window_size = st.slider("Waveform Window Size", 10, 100, 60, 2)
        
        # Detection button
        if st.button("Detect Spikes", type="primary"):
            with st.spinner("Detecting..."):
                fs = st.session_state.dat_sampling_rate
                
                # Get data (prioritize preprocessed data)
                if ch_idx in st.session_state.dat_channel_results and 'preprocessed' in st.session_state.dat_channel_results[ch_idx]:
                    data = st.session_state.dat_channel_results[ch_idx]['preprocessed'].copy()
                    source = "preprocessed"
                else:
                    data = st.session_state.dat_channels[ch_idx].copy()
                    source = "raw signal"
                
                # Detect spikes
                spikes, corrected = detect_spikes_algo(
                    data, fs, threshold_ratio, min_gap, baseline_window, direction
                )
                
                # Save results
                if ch_idx not in st.session_state.dat_channel_results:
                    st.session_state.dat_channel_results[ch_idx] = {'raw': st.session_state.dat_channels[ch_idx].copy()}
                
                st.session_state.dat_channel_results[ch_idx].update({
                    'detected_signal': data,
                    'corrected': corrected,
                    'spikes': spikes,
                    'fs': fs
                })
                
                st.success(f"Detected {len(spikes)} spikes (based on {source})")
    
    # 显示检测结果（放在参数下方）
    if ch_idx in st.session_state.dat_channel_results and 'spikes' in st.session_state.dat_channel_results[ch_idx]:
        res = st.session_state.dat_channel_results[ch_idx]
        spikes = res['spikes']
        corrected = res['corrected']
        data = res['detected_signal']
        fs = res['fs']
        
        # 显示Spike Detection Results图
        st.subheader("Spike Detection Results")
        
        # 添加时间范围选择滑块以实现缩放功能
        max_time = len(data) / fs  # 最大时间（秒）
        time_range = st.slider(
            "Select Time Range (s)", 
            0.0, 
            float(max_time), 
            (0.0, float(max_time)), 
            step=max(0.01, max_time/1000)  # 动态步长
        )
        
        # 添加人工编辑spike的选项
        st.markdown("**Manual Spike Editing:**")
        col_manual1, col_manual2, col_manual3 = st.columns(3)
        with col_manual1:
            add_spike_mode = st.checkbox("Add Spikes Mode", value=False, help="Click on the graph to add spikes")
        with col_manual2:
            remove_spike_mode = st.checkbox("Remove Spikes Mode", value=False, help="Click on spikes to remove them")
        with col_manual3:
            st.write("")  # 空列用于对齐
            if st.button("Clear Manual Spikes", key="clear_manual_spikes"):
                if 'manual_spikes' in st.session_state.dat_channel_results[ch_idx]:
                    del st.session_state.dat_channel_results[ch_idx]['manual_spikes']
                if 'removed_spikes' in st.session_state.dat_channel_results[ch_idx]:
                    del st.session_state.dat_channel_results[ch_idx]['removed_spikes']
        
        # 初始化手动spike列表
        if 'manual_spikes' not in st.session_state.dat_channel_results[ch_idx]:
            st.session_state.dat_channel_results[ch_idx]['manual_spikes'] = []
        if 'removed_spikes' not in st.session_state.dat_channel_results[ch_idx]:
            st.session_state.dat_channel_results[ch_idx]['removed_spikes'] = []
        
        # 获取合并后的spikes（自动检测的 + 手动添加的 - 手动删除的）
        all_detected_spikes = set(spikes)
        manual_spikes_set = set(st.session_state.dat_channel_results[ch_idx]['manual_spikes'])
        removed_spikes_set = set(st.session_state.dat_channel_results[ch_idx]['removed_spikes'])
        
        # 计算最终的spikes列表
        final_spikes = sorted(list((all_detected_spikes | manual_spikes_set) - removed_spikes_set))
        final_spikes = np.array(final_spikes)
        
        ch_label = st.session_state.dat_trace_labels[ch_idx]
        manual_list = list(st.session_state.dat_channel_results[ch_idx].get("manual_spikes", []))
        removed_list = list(st.session_state.dat_channel_results[ch_idx].get("removed_spikes", []))

        # ---------- Interactive Plotly (zoom/pan like Matplotlib NavigationToolbar) ----------
        if not HAS_PLOTLY:
            st.warning("Install **plotly** for interactive zoom/pan on the spike result plot: `pip install plotly`")
            fig1, ax1 = plt.subplots(figsize=(12, 4))
            t = np.arange(len(data)) / fs
            ax1.plot(t, data, "b-", lw=0.8)
            ax1.set_xlim(time_range[0], time_range[1])
            ax1.set_title(f"Spike Detection Results - {ch_label}")
            st.pyplot(fig1, width="stretch")
        else:
            fig_p, config_p = _build_spike_detection_figure_plotly(
                data,
                fs,
                final_spikes,
                spikes,
                manual_list,
                removed_list,
                time_range,
                ch_label,
            )
            plot_key = f"spike_plotly_sel_{ch_idx}"

            active_spike_set = set(int(x) for x in np.asarray(final_spikes, dtype=int).ravel())

            # Always render via st.plotly_chart (stable; avoids blank chart issues from some plotly_events versions)
            st.plotly_chart(
                fig_p,
                config=config_p,
                width="stretch",
                on_select="rerun" if remove_spike_mode else "ignore",
                selection_mode=["points", "lasso", "box"],
                key=plot_key,
            )

            if remove_spike_mode:
                st.caption("Remove: use **lasso**/**box select** on red/green markers, then click **Apply**.")
                if st.button("Apply removal from selection", key=f"apply_remove_sel_{ch_idx}"):
                    pts = _parse_plotly_selection_points(st.session_state.get(plot_key))
                    before = set(st.session_state.dat_channel_results[ch_idx].get("removed_spikes", []))
                    removed = list(before)
                    new_count = 0
                    for p in pts:
                        if not isinstance(p, dict):
                            continue
                        idx = _resolve_spike_index_for_removal(p, fs, len(data), active_spike_set)
                        if idx is not None and idx not in before:
                            removed.append(idx)
                            before.add(idx)
                            new_count += 1
                    st.session_state.dat_channel_results[ch_idx]["removed_spikes"] = sorted(removed)
                    if new_count:
                        st.success(f"Marked {new_count} spike(s) for removal.")
                    else:
                        st.warning("No removable spikes in the current selection (select red/green markers).")
                    st.rerun()

            if add_spike_mode:
                st.caption("Add: choose a **local time range**, then select **peak(s)** inside it to add.")
                col_p1, col_p2, col_p3 = st.columns([2, 2, 2])
                with col_p1:
                    add_t0 = st.number_input(
                        "Local range start (s)",
                        min_value=float(time_range[0]),
                        max_value=float(time_range[1]),
                        value=float(time_range[0]),
                        step=float(max(0.001, (time_range[1] - time_range[0]) / 1000)),
                        key=f"add_t0_{ch_idx}",
                    )
                with col_p2:
                    add_t1 = st.number_input(
                        "Local range end (s)",
                        min_value=float(time_range[0]),
                        max_value=float(time_range[1]),
                        value=min(float(time_range[0] + 0.1), float(time_range[1])),
                        step=float(max(0.001, (time_range[1] - time_range[0]) / 1000)),
                        key=f"add_t1_{ch_idx}",
                    )
                with col_p3:
                    max_candidates = st.number_input(
                        "Max peaks",
                        min_value=5,
                        max_value=200,
                        value=50,
                        step=5,
                        key=f"add_maxpk_{ch_idx}",
                    )

                t0, t1 = float(min(add_t0, add_t1)), float(max(add_t0, add_t1))
                i0 = int(np.clip(int(round(t0 * fs)), 0, len(data) - 1))
                i1 = int(np.clip(int(round(t1 * fs)), i0 + 1, len(data)))
                seg = data[i0:i1]
                if seg.size < 3:
                    st.warning("Local range too small.")
                else:
                    # find_peaks finds maxima; invert for 'down'
                    y = -seg if direction == "down" else seg
                    peaks, props = find_peaks(y)
                    if peaks.size == 0:
                        st.info("No peaks found in this local range.")
                    else:
                        # Rank by peak height in y (i.e., extremeness)
                        heights = y[peaks]
                        order = np.argsort(heights)[::-1]
                        peaks = peaks[order][: int(max_candidates)]
                        idxs = (i0 + peaks).astype(int)
                        times = idxs.astype(float) / fs
                        amps = data[idxs].astype(float)
                        df_peaks = pd.DataFrame(
                            {
                                "add": [False] * len(idxs),
                                "t(s)": times,
                                "idx": idxs,
                                "amp": amps,
                            }
                        )
                        df_peaks = df_peaks.sort_values("t(s)").reset_index(drop=True)
                        edited = st.data_editor(
                            df_peaks,
                            hide_index=True,
                            width="stretch",
                            column_config={
                                "add": st.column_config.CheckboxColumn("add"),
                                "t(s)": st.column_config.NumberColumn("t(s)", format="%.6f"),
                                "idx": st.column_config.NumberColumn("idx"),
                                "amp": st.column_config.NumberColumn("amp", format="%.6f"),
                            },
                            key=f"peaks_editor_{ch_idx}",
                        )
                        if st.button("Add selected peak(s) as spikes", key=f"apply_add_peaks_{ch_idx}"):
                            to_add = (
                                edited.loc[edited["add"] == True, "idx"].astype(int).tolist()  # noqa: E712
                                if isinstance(edited, pd.DataFrame)
                                else []
                            )
                            if not to_add:
                                st.warning("No peaks selected.")
                            else:
                                ms = list(st.session_state.dat_channel_results[ch_idx].get("manual_spikes", []))
                                ms_set = set(int(x) for x in ms)
                                new_added = 0
                                for idx in to_add:
                                    if int(idx) not in ms_set:
                                        ms.append(int(idx))
                                        ms_set.add(int(idx))
                                        new_added += 1
                                st.session_state.dat_channel_results[ch_idx]["manual_spikes"] = sorted(ms)
                                st.success(f"Added {new_added} spike(s) from selected peaks.")
                                st.rerun()

        # 如果启用了手动编辑模式，提供坐标输入功能
        if add_spike_mode:
            st.subheader("Add Spike Manually")
            col_add1, col_add2 = st.columns(2)
            with col_add1:
                spike_time_input = st.number_input("Spike Time (s)", value=0.0, step=0.001, format="%.3f", 
                                                  help="Enter the time in seconds where you want to add a spike")
            with col_add2:
                if st.button("Add Spike at Time", key="add_spike_at_time"):
                    spike_index = int(spike_time_input * fs)  # 转换时间到索引
                    if 0 <= spike_index < len(data):
                        # 添加到手动spikes列表
                        manual_spikes_list = st.session_state.dat_channel_results[ch_idx].get('manual_spikes', [])
                        if spike_index not in manual_spikes_list:
                            manual_spikes_list.append(spike_index)
                            st.session_state.dat_channel_results[ch_idx]['manual_spikes'] = sorted(manual_spikes_list)
                            st.success(f"Added spike at index {spike_index} (time: {spike_time_input}s)")
                        else:
                            st.warning("Spike already exists at this time")
                    else:
                        st.error(f"Spike time out of range. Valid range: 0 to {len(data)/fs:.3f}s")
        
        elif remove_spike_mode:
            st.subheader("Remove Spike Manually")
            col_remove1, col_remove2 = st.columns(2)
            with col_remove1:
                remove_spike_time_input = st.number_input("Spike Time to Remove (s)", value=0.0, step=0.001, format="%.3f", 
                                                         help="Enter the time in seconds of the spike you want to remove")
            with col_remove2:
                if st.button("Remove Spike at Time", key="remove_spike_at_time"):
                    remove_spike_index = int(remove_spike_time_input * fs)  # 转换时间到索引
                    # 检查是否在自动检测的spikes中
                    if remove_spike_index in spikes:
                        removed_spikes_list = st.session_state.dat_channel_results[ch_idx].get('removed_spikes', [])
                        if remove_spike_index not in removed_spikes_list:
                            removed_spikes_list.append(remove_spike_index)
                            st.session_state.dat_channel_results[ch_idx]['removed_spikes'] = sorted(removed_spikes_list)
                            st.success(f"Marked spike for removal at index {remove_spike_index} (time: {remove_spike_time_input}s)")
                        else:
                            st.warning("Spike already marked for removal")
                    else:
                        st.error(f"No auto-detected spike found at time {remove_spike_time_input}s")
        
        # 更新最终的spikes数量显示
        all_detected_spikes = set(spikes)
        manual_spikes_set = set(st.session_state.dat_channel_results[ch_idx].get('manual_spikes', []))
        removed_spikes_set = set(st.session_state.dat_channel_results[ch_idx].get('removed_spikes', []))
        final_spikes = sorted(list((all_detected_spikes | manual_spikes_set) - removed_spikes_set))
        final_spikes = np.array(final_spikes)
        
        st.info(f"Final spike count: {len(final_spikes)} total "
                f"(Auto: {len(spikes)}, Manual: {len(manual_spikes_set)}, Removed: {len(removed_spikes_set)})")
        
        # 添加导出/保存编辑结果的功能
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            # 矩形表：各列长度统一为 max(各列表长度, 1)，较短列用 None 填充
            auto_list = np.asarray(spikes).ravel().astype(int).tolist() if len(spikes) > 0 else []
            man_list = list(st.session_state.dat_channel_results[ch_idx].get("manual_spikes", []))
            rem_list = list(st.session_state.dat_channel_results[ch_idx].get("removed_spikes", []))
            fin_list = final_spikes.astype(int).tolist() if len(final_spikes) > 0 else []
            n_rows = max(len(auto_list), len(man_list), len(rem_list), len(fin_list), 1)
            edits_data = {
                "channel_index": [ch_idx] * n_rows,
                "channel_label": [st.session_state.dat_trace_labels[ch_idx]] * n_rows,
                "sampling_rate": [fs] * n_rows,
                "auto_detected_spikes": _pad_list_to_len(auto_list, n_rows),
                "manual_added_spikes": _pad_list_to_len(man_list, n_rows),
                "manually_removed_spikes": _pad_list_to_len(rem_list, n_rows),
                "final_spikes": _pad_list_to_len(fin_list, n_rows),
            }

            # 将数据转换为CSV格式
            df = pd.DataFrame(edits_data)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Export Manual Edits (CSV)",
                data=csv,
                file_name=f"spike_edits_channel_{ch_idx+1}.csv",
                mime="text/csv"
            )
        
        with col_export2:
            # 导入spike编辑的选项（简化版）
            uploaded_file = st.file_uploader("Import Spike Edits (CSV)", type=['csv'], key="import_edits")
            if uploaded_file is not None:
                try:
                    df_import = pd.read_csv(uploaded_file)
                    # 解析导入的数据
                    if 'manual_added_spikes' in df_import.columns:
                        imported_manual_spikes_series = df_import['manual_added_spikes'].dropna()
                        if not imported_manual_spikes_series.empty:
                            imported_manual_spikes = imported_manual_spikes_series.astype(int).tolist()
                            st.session_state.dat_channel_results[ch_idx]['manual_spikes'] = imported_manual_spikes
                            st.success(f"Imported {len(imported_manual_spikes)} manual spikes")
                    
                    if 'manually_removed_spikes' in df_import.columns:
                        imported_removed_spikes_series = df_import['manually_removed_spikes'].dropna()
                        if not imported_removed_spikes_series.empty:
                            imported_removed_spikes = imported_removed_spikes_series.astype(int).tolist()
                            st.session_state.dat_channel_results[ch_idx]['removed_spikes'] = imported_removed_spikes
                            st.success(f"Imported {len(imported_removed_spikes)} removed spikes")
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Error importing spike edits: {str(e)}")
        
        # 单独显示波形图
        st.subheader("Spike Waveforms")
        
        # 获取最终的spikes列表（包括手动添加和删除的）
        all_detected_spikes = set(spikes)
        manual_spikes_set = set(st.session_state.dat_channel_results[ch_idx].get('manual_spikes', []))
        removed_spikes_set = set(st.session_state.dat_channel_results[ch_idx].get('removed_spikes', []))
        final_spikes = sorted(list((all_detected_spikes | manual_spikes_set) - removed_spikes_set))
        final_spikes = np.array(final_spikes)
        
        # 添加一个按钮用于手动刷新波形图
        refresh_waveforms = st.button("Refresh Waveforms", key="refresh_waveforms")
        
        if len(final_spikes) > 0:
            # 使用Streamlit的缓存功能来优化波形提取和绘图
            @st.cache_data
            def get_waveforms_and_plot(_corrected, _spikes, _window_size, _fs):
                waveforms, _ = extract_spike_waveforms(_corrected, _spikes, _window_size, return_indices=True)
                
                if len(waveforms) > 0:
                    fig2, ax2 = plt.subplots(figsize=(12, 4))
                    time_per_sample = 1000 / _fs
                    waveform_time = np.arange(_window_size) * time_per_sample - (_window_size // 2) * time_per_sample
                    
                    for waveform in waveforms[:50]:  # Show only first 50 to avoid clutter
                        ax2.plot(waveform_time, waveform, 'b-', alpha=0.3, linewidth=0.5)
                    
                    avg_waveform = np.mean(waveforms, axis=0)
                    ax2.plot(waveform_time, avg_waveform, 'r-', linewidth=2, label='Average Waveform')
                    
                    ax2.set_title(f"Spike Waveforms (Total: {len(waveforms)}, Window: {_window_size})")
                    ax2.set_xlabel("Time (ms)")
                    ax2.set_ylabel("Amplitude")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    return fig2
                return None
            
            # 当waveform参数改变或用户点击刷新按钮时重新计算和绘图
            if 'last_window_size' not in st.session_state:
                st.session_state.last_window_size = window_size
            
            if st.session_state.last_window_size != window_size or refresh_waveforms:
                st.session_state.last_window_size = window_size
                # 清除缓存以强制重新计算
                get_waveforms_and_plot.clear()
            
            fig2 = get_waveforms_and_plot(corrected, tuple(final_spikes), window_size, fs)
            if fig2:
                st.pyplot(fig2, width="stretch")
        else:
            st.info("No spikes detected yet. Please run spike detection first or add manual spikes.")


def analysis_tab():
    """Analysis tab"""
    st.header("Results Analysis")
    
    if not st.session_state.dat_channels:
        st.info("Please upload a DAT file first")
        return
    
    # Check if there are detection results
    has_results = any('spikes' in res for res in st.session_state.dat_channel_results.values())
    if not has_results:
        st.info("Please perform spike detection first")
        return
    
    # Channel selection
    channel_options = []
    for i, label in enumerate(st.session_state.dat_trace_labels):
        if i in st.session_state.dat_channel_results and 'spikes' in st.session_state.dat_channel_results[i]:
            spike_count = len(st.session_state.dat_channel_results[i]['spikes'])
            channel_options.append(f"{i+1}: {label} ({spike_count} spikes)")
        else:
            channel_options.append(f"{i+1}: {label}")
    
    selected_channel = st.selectbox("Select Analysis Channel", channel_options, 
                                   index=st.session_state.current_channel, key="analysis_channel")
    ch_idx = int(selected_channel.split(':')[0]) - 1
    
    if ch_idx not in st.session_state.dat_channel_results or 'spikes' not in st.session_state.dat_channel_results[ch_idx]:
        st.warning("No spike detection results for this channel")
        return
    
    res = st.session_state.dat_channel_results[ch_idx]
    spikes = res['spikes']
    fs = res['fs']
    signal = res.get('detected_signal', res.get('raw'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Frequency Analysis")
        interval = st.slider("Analysis Interval (s)", 0.1, 5.0, 0.5, 0.1)
        
        if st.button("Calculate Frequency"):
            signal_duration = len(signal) / fs
            spike_times = spikes / fs
            
            time_bins = np.arange(0, signal_duration, interval)
            if time_bins[-1] < signal_duration:
                time_bins = np.append(time_bins, signal_duration)
            
            frequencies = np.zeros(len(time_bins) - 1)
            for i in range(len(time_bins) - 1):
                bin_start = time_bins[i]
                bin_end = time_bins[i + 1]
                spike_count = np.sum((spike_times >= bin_start) & (spike_times < bin_end))
                frequencies[i] = spike_count / interval
            
            # Plot frequency
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(time_bins[:-1], frequencies, width=interval*0.8, color='skyblue', alpha=0.8)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Spikes/s")
            ax.set_title("Spike Frequency Distribution")
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
    
    with col2:
        st.subheader("Spike Classification")
        classification_method = st.selectbox("Classification Method", 
                                           ["K-means", "PCA+KMeans", "GaussianMixture"], index=0)
        n_clusters = st.slider("Number of Clusters", 2, 10, 3, 1)
        
        if st.button("Perform Classification"):
            try:
                # 提取波形用于分类
                waveforms = extract_spike_waveforms(res['corrected'], spikes, window_size)
                
                if len(waveforms) == 0:
                    st.warning("No waveforms to classify")
                    return
                
                # 特征提取 - 使用PCA降维或直接使用波形峰值作为特征
                if classification_method == "PCA+KMeans":
                    # PCA降维
                    pca = PCA(n_components=min(10, waveforms.shape[1]))
                    features = pca.fit_transform(waveforms.reshape(len(waveforms), -1))
                    
                    # K-means聚类
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(features)
                    
                elif classification_method == "GaussianMixture":
                    # 高斯混合模型
                    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
                    features_flat = waveforms.reshape(len(waveforms), -1)
                    labels = gmm.fit_predict(features_flat)
                    
                else:  # K-means
                    # 直接使用波形作为特征
                    features_flat = waveforms.reshape(len(waveforms), -1)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(features_flat)
                
                # 保存分类结果
                st.session_state.spike_classifications[ch_idx] = labels
                
                # 绘制分类结果
                fig, ax = plt.subplots(figsize=(10, 6))
                
                colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
                for cluster_id in range(n_clusters):
                    cluster_spikes = waveforms[labels == cluster_id]
                    if len(cluster_spikes) > 0:
                        avg_waveform = np.mean(cluster_spikes, axis=0)
                        std_waveform = np.std(cluster_spikes, axis=0)
                        
                        time_axis = np.arange(len(avg_waveform)) * (1000/fs) - (window_size//2) * (1000/fs)
                        
                        ax.plot(time_axis, avg_waveform, 
                               label=f'Cluster {cluster_id+1} (n={np.sum(labels==cluster_id)})',
                               color=colors[cluster_id], linewidth=2)
                        ax.fill_between(time_axis, 
                                      avg_waveform - std_waveform, 
                                      avg_waveform + std_waveform,
                                      color=colors[cluster_id], alpha=0.2)
                
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Amplitude")
                ax.set_title("Spike Classification Results")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                st.success(f"Classification completed with {classification_method}")
                
            except Exception as e:
                st.error(f"Classification failed: {str(e)}")
            with st.spinner("Classifying..."):
                # Extract waveforms
                window_size = 30
                waveforms, valid_indices = extract_spike_waveforms(res['corrected'], spikes, window_size, return_indices=True)
                
                if len(waveforms) < 2:
                    st.warning("Too few spikes for classification")
                    return
                
                # Perform classification
                if classification_method == "K-means":
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(waveforms)
                elif classification_method == "PCA+KMeans":
                    pca = PCA(n_components=min(5, waveforms.shape[1]))
                    pca_result = pca.fit_transform(waveforms)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(pca_result)
                else:  # GaussianMixture
                    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
                    labels = gmm.fit_predict(waveforms)
                
                # Save classification results
                labels_full = np.full(len(spikes), -1, dtype=int)
                for lbl, spike_idx in zip(labels, valid_indices):
                    if 0 <= spike_idx < len(labels_full):
                        labels_full[spike_idx] = int(lbl)
                
                unique_labels = sorted(l for l in np.unique(labels_full) if l >= 0)
                counts = {int(label): int(np.sum(labels_full == label)) for label in unique_labels}
                
                st.session_state.spike_classifications[ch_idx] = {
                    'method': classification_method,
                    'labels_full': labels_full,
                    'counts': counts
                }
                
                # Display classification results
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Waveform plot
                time_per_sample = 1000 / fs
                waveform_time = np.arange(window_size) * time_per_sample - (window_size // 2) * time_per_sample
                
                if len(unique_labels) > 0:
                    try:
                        color_map = plt.cm.get_cmap('tab10', max(len(unique_labels), 1))
                    except:
                        color_map = plt.cm.viridis
                    
                    for label in unique_labels:
                        mask = labels == label
                        if np.any(mask):
                            class_waveforms = waveforms[mask]
                            for waveform in class_waveforms[:20]:  # Show 20 per class
                                ax1.plot(waveform_time, waveform, color=color_map(label), alpha=0.2, linewidth=0.5)
                            avg_waveform = np.mean(class_waveforms, axis=0)
                            ax1.plot(waveform_time, avg_waveform, color=color_map(label), 
                                    linewidth=2, label=f'Class {label}')
                
                ax1.set_title(f"Spike Classification ({classification_method})")
                ax1.set_xlabel("Time (ms)")
                ax1.set_ylabel("Amplitude")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Statistics plot
                if counts:
                    labels_list = list(counts.keys())
                    values = [counts[label] for label in labels_list]
                    if len(unique_labels) > 0:
                        try:
                            colors = [color_map(label) for label in labels_list]
                        except:
                            colors = 'skyblue'
                    else:
                        colors = 'skyblue'
                    ax2.bar(labels_list, values, color=colors, alpha=0.8)
                    ax2.set_title("Spike Count by Class")
                    ax2.set_xlabel("Class")
                    ax2.set_ylabel("Count")
                    ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show statistics
                st.write("### Classification Statistics")
                for label, count in counts.items():
                    percentage = count / len(waveforms) * 100
                    st.write(f"Class {label}: {count} spikes ({percentage:.1f}%)")
    
    # Results export
    st.subheader("Results Export")
    if st.button("Export to Excel"):
        try:
            # Create Excel file
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Create summary sheet
                summary_data = []
                for i, label in enumerate(st.session_state.dat_trace_labels):
                    if i in st.session_state.dat_channel_results and 'spikes' in st.session_state.dat_channel_results[i]:
                        res_i = st.session_state.dat_channel_results[i]
                        auto_spikes_i = set(np.asarray(res_i.get('spikes', []), dtype=int).ravel().tolist())
                        manual_spikes_i = set(np.asarray(res_i.get('manual_spikes', []), dtype=int).ravel().tolist())
                        removed_spikes_i = set(np.asarray(res_i.get('removed_spikes', []), dtype=int).ravel().tolist())
                        final_spikes_i = (auto_spikes_i | manual_spikes_i) - removed_spikes_i
                        spike_count = len(final_spikes_i)
                        preprocessed = "Yes" if 'preprocessed' in st.session_state.dat_channel_results[i] else "No"
                        
                        if i in st.session_state.spike_classifications:
                            cls_info = st.session_state.spike_classifications[i]
                            class_info = f"{cls_info['method']}, {len(cls_info['counts'])} classes"
                        else:
                            class_info = "Not classified"
                    else:
                        spike_count = 0
                        preprocessed = "No"
                        class_info = "Not detected"
                    
                    summary_data.append({
                        'Channel': i+1,
                        'Label': label,
                        'Preprocessed': preprocessed,
                        'Spike Count': spike_count,
                        'Classification Status': class_info
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Channel Summary', index=False)
                
                # Create sheet for each channel with spikes
                for ch_idx, res in st.session_state.dat_channel_results.items():
                    if 'spikes' not in res:
                        continue
                    
                    fs = res.get('fs', st.session_state.dat_sampling_rate)
                    signal = res.get('detected_signal')
                    if signal is None:
                        signal = res.get('preprocessed')
                    if signal is None:
                        signal = res.get('raw')
                    if signal is None and ch_idx < len(st.session_state.dat_channels):
                        signal = st.session_state.dat_channels[ch_idx]
                    if signal is None:
                        continue

                    signal = np.asarray(signal, dtype=float).ravel()
                    n = len(signal)
                    times = np.arange(n, dtype=float) / fs

                    auto_spikes = set(np.asarray(res.get('spikes', []), dtype=int).ravel().tolist())
                    manual_spikes = set(np.asarray(res.get('manual_spikes', []), dtype=int).ravel().tolist())
                    removed_spikes = set(np.asarray(res.get('removed_spikes', []), dtype=int).ravel().tolist())
                    final_spikes = sorted((auto_spikes | manual_spikes) - removed_spikes)
                    final_spikes = [idx for idx in final_spikes if 0 <= idx < n]

                    spike_flag = np.zeros(n, dtype=np.int8)
                    if final_spikes:
                        spike_flag[np.asarray(final_spikes, dtype=int)] = 1

                    # Required export format: Time / signal / spike(0|1), after manual add/remove edits
                    df = pd.DataFrame(
                        {
                            'Time': times,
                            'signal': signal,
                            'spike': spike_flag,
                        }
                    )
                    sheet_name = f'Channel{ch_idx+1}_Spikes'
                    if len(sheet_name) > 31:
                        sheet_name = sheet_name[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            output.seek(0)
            
            # Provide download
            st.download_button(
                label="Download Excel File",
                data=output,
                file_name="spike_analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        except Exception as e:
            st.error(f"Export failed: {e}")


def visualize_artifact_removal(original, corrected, mask, fs, regions, method):
    """Visualize artifact removal (for Streamlit display)"""
    t = np.arange(len(original)) / fs
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, original, 'b-', label='Original Signal', alpha=0.6, linewidth=1)
    ax1.plot(t, corrected, 'r-', label='Corrected', alpha=0.8, linewidth=1)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(regions)))
    for i, region in enumerate(regions):
        trans_start, trans_end = region['transition']
        core_start, core_end = region['core']
        
        ax1.axvspan(t[trans_start], t[trans_end], alpha=0.2, color=colors[i])
        ax1.axvspan(t[core_start], t[core_end], alpha=0.4, color=colors[i])
        
        for bound in [core_start, core_end]:
            ax1.axvline(t[bound], color=colors[i], linestyle='--', alpha=0.7, linewidth=1)
    
    ax1.set_title(f'Artifact Removal Comparison - Method: {method}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1])
    diff = corrected - original
    ax2.plot(t, diff, 'g-', linewidth=0.8)
    ax2.fill_between(t, diff, alpha=0.3, color='green')
    ax2.set_title('Correction Amount (Corrected - Original)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Δ Amplitude')
    ax2.grid(alpha=0.3)
    
    ax3 = fig.add_subplot(gs[2])
    orig_diff = np.diff(original)
    corr_diff = np.diff(corrected)
    ax3.plot(t[1:], orig_diff, 'b-', label='Original', alpha=0.5, linewidth=0.8)
    ax3.plot(t[1:], corr_diff, 'r-', label='Corrected', alpha=0.8, linewidth=0.8)
    
    for region in regions:
        for bound in [region['core'][0], region['core'][1]]:
            if 0 < bound < len(t) - 1:
                ax3.axvline(t[bound], color='k', linestyle='--', alpha=0.5)
    
    ax3.set_title('First Derivative (Check Boundary Jumps)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('d(signal)/dt')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


# Run Streamlit application
if __name__ == "__main__":
    main()
