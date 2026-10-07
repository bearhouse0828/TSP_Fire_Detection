
#!/usr/bin/env python3
"""
Site data loader for aa-44-16-29 using FireDetectionDataLoader.
Loads PM1, PM4, S1-S12, T, RH for the specified site and provides
plot utilities for time-series (line and bar) with safe downsampling/resampling.
"""

import matplotlib
matplotlib.use('Agg')  # Safe for headless environments; remove if running locally with GUI

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, Union
from sklearn.ensemble import IsolationForest
from scipy.signal import detrend, welch

from pathlib import Path
from fire_detection_data_loader import FireDetectionDataLoader

SITE_ID = 'aa-44-16-29'
FIG_DIR = Path('figures') / SITE_ID
OUT_DIR = Path('outputs') / SITE_ID

class SiteDataLoader:
    """Load per-site time-series for selected sensors using FireDetectionDataLoader."""

    def __init__(self, data_folder: str = 'data', site_id: str = SITE_ID, loader: Optional[FireDetectionDataLoader] = None) -> None:
        self.data_folder = data_folder
        self.site_id = site_id
        # Allow injecting a shared loader to avoid reloading for each site
        self.loader = loader if loader is not None else FireDetectionDataLoader(data_folder)
        self.loaded: Dict[str, pd.Series] = {}

    def load(self) -> Dict[str, pd.Series]:
        """Load PM1, PM4, S1-S12, T, RH for the site as pandas Series keyed by sensor name."""
        # Use cached data if already loaded to avoid redundant I/O
        data = self.loader.data if getattr(self.loader, 'data', None) else self.loader.load_all_data()
        wanted = ['PM1', 'PM4', 'Temperature', 'Relative_Humidity'] + [f'SPH_S{i}' for i in range(1, 13)]
        out: Dict[str, pd.Series] = {}
        for sensor in wanted:
            if sensor not in data:
                continue
            df = data[sensor]
            if self.site_id not in df.columns:
                continue
            s = df[self.site_id].copy()
            s.name = sensor
            out[sensor] = s
        self.loaded = out
        return out


def _to_dataframe(input_data: Union[pd.Series, pd.DataFrame, Dict[str, pd.Series]]) -> pd.DataFrame:
    if isinstance(input_data, dict):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.Series):
        df = input_data.to_frame()
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError('input_data must be Series, DataFrame, or dict of Series')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _reduce_points(df: pd.DataFrame,
                   resample_freq: Optional[str] = None,
                   agg: str = 'mean',
                   max_points: int = 5000) -> pd.DataFrame:
    if resample_freq:
        df_res = getattr(df.resample(resample_freq), agg)()
        return df_res.dropna(how='all')
    n = len(df)
    if n <= max_points:
        return df
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    return df.iloc[idx]


def plot_time_series(input_data: Union[pd.Series, pd.DataFrame, Dict[str, pd.Series]],
                     y_min: Optional[float] = None,
                     y_max: Optional[float] = None,
                     title: str = 'Time Series',
                     save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (14, 6),
                     resample_freq: Optional[str] = '30S',
                     agg: str = 'mean',
                     max_points: int = 5000,
                     rasterized: bool = True) -> plt.Figure:
    """
    Line plot with optional time-based resampling (default 30 seconds) or decimation.
    """
    df = _to_dataframe(input_data)
    df = _reduce_points(df, resample_freq=resample_freq, agg=agg, max_points=max_points)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for col in df.columns:
        ax.plot(df.index, df[col], label=col, alpha=0.9, rasterized=rasterized, linewidth=1.0)

    ax.set_title(title)
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Value')
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_bar_series(input_data: Union[pd.Series, pd.DataFrame, Dict[str, pd.Series]],
                    y_min: Optional[float] = None,
                    y_max: Optional[float] = None,
                    title: str = 'Bar Time Series',
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (14, 6),
                    resample_freq: str = '30S',
                    agg: str = 'mean') -> plt.Figure:
    """
    Bar plot on time axis; always resamples to bin size (default 30 seconds) before plotting.
    """
    df = _to_dataframe(input_data)
    df = getattr(df.resample(resample_freq), agg)().dropna(how='all')

    if len(df.index) >= 2:
        bin_seconds = (df.index[1] - df.index[0]).total_seconds()
    else:
        bin_seconds = 30.0
    width_days = 0.8 * bin_seconds / 86400.0

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    cols = list(df.columns)
    n = len(cols)

    if n == 1:
        ax.bar(df.index, df[cols[0]], width=width_days, alpha=0.9, label=cols[0])
    else:
        offsets = np.linspace(-width_days*(n-1)/2, width_days*(n-1)/2, n)
        for i, col in enumerate(cols):
            ax.bar(df.index + pd.to_timedelta(offsets[i], unit='D'), df[col],
                   width=width_days, alpha=0.85, label=col)

    ax.set_title(title)
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Value')
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_pm_ratio(pm1: pd.Series,
                  pm4: pd.Series,
                  y_min: Optional[float] = None,
                  y_max: Optional[float] = None,
                  title: str = 'PM1/PM4 Ratio (30s mean)',
                  save_path: Optional[str] = None,
                  figsize: Tuple[int, int] = (14, 5),
                  resample_freq: str = '30S',
                  agg: str = 'mean') -> plt.Figure:
    """
    Compute and plot PM1/PM4 ratio as a time series (with resampling).
    Handles division safely and clips extreme values.
    """
    df = _to_dataframe({'PM1': pm1, 'PM4': pm4})
    df = getattr(df.resample(resample_freq), agg)().dropna(how='all')
    # Avoid division by zero
    ratio = df['PM1'] / df['PM4'].replace(0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    # Optional clipping to reduce spikes impact (e.g., 0-10)
    # ratio = ratio.clip(lower=0, upper=10)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(ratio.index, ratio.values, color='tab:purple', alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('PM1 / PM4')
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_pca_sensors(s_dict: Dict[str, pd.Series],
                    sensor_prefix: str = 'SPH_S',
                    n_components: int = 5,
                    resample_freq: str = '30S',
                    agg: str = 'mean',
                    dropna: str = 'any') -> Dict[str, object]:
    """
    Run PCA on S1..S12 (columns) after time alignment and resampling.

    Returns dict with keys: df, df_resampled, X, scaler, pca, components, explained_variance_ratio,
    scores (DataFrame indexed by time), loadings (DataFrame sensors x components)
    """
    # Collect S1..S12 into DataFrame
    sensors = {k: v for k, v in s_dict.items() if k.startswith(sensor_prefix)}
    if not sensors:
        raise ValueError('No SPH_S* sensors found')
    df = pd.DataFrame(sensors)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Resample
    df_res = getattr(df.resample(resample_freq), agg)()
    # Drop rows with too many NaNs
    if dropna == 'any':
        df_res = df_res.dropna(how='any')
    elif dropna == 'all':
        df_res = df_res.dropna(how='all')
    else:
        # drop rows with more than half NaNs
        thresh = int(np.ceil(df_res.shape[1] / 2))
        df_res = df_res.dropna(thresh=thresh)

    # Standardize features (per sensor)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_res.values)

    # PCA
    pca = PCA(n_components=min(n_components, X.shape[1]))
    scores = pca.fit_transform(X)  # time x components

    # Build outputs
    comp_names = [f'PC{i+1}' for i in range(pca.n_components_)]
    score_df = pd.DataFrame(scores, index=df_res.index, columns=comp_names)
    loadings = pd.DataFrame(pca.components_.T,
                            index=df_res.columns,
                            columns=comp_names)

    return {
        'df': df,
        'df_resampled': df_res,
        'X': X,
        'scaler': scaler,
        'pca': pca,
        'components': pca.components_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'scores': score_df,
        'loadings': loadings,
    }


def plot_pca_results(pca_out: Dict[str, object],
                     prefix: str = 'aa-44-16-29_SPH_PCA',
                     save_dir: str = 'figures') -> None:
    """Save scree plot, loadings heatmap, and PC1/PC2 time series plots."""
    import seaborn as sns
    import os
    os.makedirs(save_dir, exist_ok=True)

    evr = pca_out['explained_variance_ratio']
    scores = pca_out['scores']
    loadings = pca_out['loadings']

    # Scree plot
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(range(1, len(evr)+1), evr*100, color='tab:blue')
    ax.plot(range(1, len(evr)+1), np.cumsum(evr)*100, marker='o', color='tab:orange', label='Cumulative')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('Scree Plot')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{save_dir}/{prefix}_scree.png", dpi=150)

    # Loadings heatmap
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(loadings, annot=False, cmap='coolwarm', center=0)
    ax.set_title('PCA Loadings (S1..S12 x PCs)')
    plt.tight_layout()
    fig.savefig(f"{save_dir}/{prefix}_loadings.png", dpi=150)

    # PC1 & PC2 time series
    fig, ax = plt.subplots(2,1, figsize=(12,6), sharex=True)
    if 'PC1' in scores.columns:
        ax[0].plot(scores.index, scores['PC1'], color='tab:green')
        ax[0].set_title('PC1 Scores over Time')
        ax[0].grid(True, alpha=0.3)
    if 'PC2' in scores.columns:
        ax[1].plot(scores.index, scores['PC2'], color='tab:red')
        ax[1].set_title('PC2 Scores over Time')
        ax[1].grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(f"{save_dir}/{prefix}_scores.png", dpi=150)


import seaborn as sns

def compute_site_correlation_30s(data_folder: str = 'data', agg: str = 'mean'):
    """Compute 30s-mean correlation matrices across sites for PM1 and PM4.
    Returns (corr_pm1, corr_pm4) as DataFrames (stations x stations).
    """
    loader = FireDetectionDataLoader(data_folder)
    data = loader.load_all_data()
    pm1 = data.get('PM1')
    pm4 = data.get('PM4')
    if pm1 is None or pm4 is None:
        raise ValueError('PM1/PM4 not loaded')
    # Ensure datetime index and sort
    pm1 = pm1.copy(); pm4 = pm4.copy()
    if not isinstance(pm1.index, pd.DatetimeIndex):
        pm1.index = pd.to_datetime(pm1.index)
    if not isinstance(pm4.index, pd.DatetimeIndex):
        pm4.index = pd.to_datetime(pm4.index)
    pm1 = pm1.sort_index(); pm4 = pm4.sort_index()
    # Resample to 30s mean across all station columns
    pm1_30s = getattr(pm1.resample('30S'), agg)()
    pm4_30s = getattr(pm4.resample('30S'), agg)()
    # Correlation across stations
    corr_pm1 = pm1_30s.corr()
    corr_pm4 = pm4_30s.corr()
    return corr_pm1, corr_pm4


def plot_site_correlation_heatmaps(corr_pm1, corr_pm4, save_dir: str = 'figures', prefix: str = 'sites_PM_corr_30s'):
    """Plot and save heatmaps for PM1 and PM4 site correlation matrices."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    # PM1 heatmap
    fig1, ax1 = plt.subplots(figsize=(8,6))
    sns.heatmap(corr_pm1, vmin=-1, vmax=1, cmap='coolwarm', square=True, cbar=True, ax=ax1)
    ax1.set_title('Site Correlation (PM1, 30s mean)')
    plt.tight_layout()
    fig1.savefig(f"{save_dir}/{prefix}_PM1.png", dpi=150)
    # PM4 heatmap
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.heatmap(corr_pm4, vmin=-1, vmax=1, cmap='coolwarm', square=True, cbar=True, ax=ax2)
    ax2.set_title('Site Correlation (PM4, 30s mean)')
    plt.tight_layout()
    fig2.savefig(f"{save_dir}/{prefix}_PM4.png", dpi=150)



def compute_derivative(series: pd.Series,
                        resample_freq: str = '30S',
                        agg: str = 'mean',
                        method: str = 'diff') -> pd.Series:
    """
    Compute time derivative d(series)/dt after resampling.
    Returns derivative in units per second.
    method: 'diff' (forward diff) or 'central' (central diff where possible).
    """
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    if resample_freq is not None:
        s = getattr(s.resample(resample_freq), agg)()
    if len(s) < 2:
        return s * 0.0
    # convert timedelta to seconds using index spacing
    dt = s.index.to_series().diff().dt.total_seconds()
    if method == 'central' and len(s) >= 3:
        # central difference for interior points
        sp = s.shift(1)
        sn = s.shift(-1)
        dt_c = (s.index.to_series().shift(-1) - s.index.to_series().shift(1)).dt.total_seconds()
        deriv = (sn - sp) / dt_c
    else:
        deriv = s.diff() / dt
    return deriv


def plot_dpm1_dt(pm1: pd.Series,
                  resample_freq: str = '30S',
                  agg: str = 'mean',
                  method: str = 'diff',
                  y_min: Optional[float] = None,
                  y_max: Optional[float] = None,
                  title: str = 'dPM1/dt (30s)',
                  save_path: Optional[str] = None,
                  figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
    """
    Compute and plot dPM1/dt with chosen resampling and difference scheme.
    """
    dpm1 = compute_derivative(pm1, resample_freq=resample_freq, agg=agg, method=method)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(dpm1.index, dpm1.values, color='tab:blue', alpha=0.9)
    ax.axhline(0.0, color='#888', linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('dPM1/dt (µg/m³ per second)')
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig



def compute_pm_ratio(pm1: pd.Series,
                     pm4: pd.Series,
                     resample_freq: Optional[str] = '30S',
                     agg: str = 'mean') -> pd.Series:
    """Return PM1/PM4 ratio time series after resampling."""
    df = _to_dataframe({'PM1': pm1, 'PM4': pm4})
    if resample_freq is not None:
        df = getattr(df.resample(resample_freq), agg)().dropna(how='all')
    ratio = df['PM1'] / df['PM4'].replace(0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan)


def plot_d_ratio_dt(pm1: pd.Series,
                    pm4: pd.Series,
                    resample_freq: str = '30S',
                    agg: str = 'mean',
                    method: str = 'central',
                    y_min: Optional[float] = None,
                    y_max: Optional[float] = None,
                    title: str = 'd(PM1/PM4)/dt (30s)',
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
    """Compute PM1/PM4 then plot its time derivative."""
    ratio = compute_pm_ratio(pm1, pm4, resample_freq=resample_freq, agg=agg)
    d_ratio = compute_derivative(ratio, resample_freq=None, agg=agg, method=method)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(d_ratio.index, d_ratio.values, color='tab:purple', alpha=0.9)
    ax.axhline(0.0, color='#888', linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('d(PM1/PM4)/dt (per second)')
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig

# Z baseline functions
EPS = 1e-3

# ========== Helper: Robust baseline (hourly median + MAD) ==========
def _robust_hourly_baseline(series: pd.Series, clean_mask: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mu, sigma, z) mapped to each timestamp using hourly median + MAD.
    clean_mask=True marks points to exclude when computing baseline.
    """
    s = series.copy()
    # Align mask to series index
    if not isinstance(clean_mask, pd.Series):
        clean_mask = pd.Series(clean_mask, index=s.index)
    else:
        clean_mask = clean_mask.reindex(s.index, fill_value=False)
    # Hour-of-day vectors
    hod_full = s.index.hour
    sel = ~clean_mask
    hod_sel = hod_full[sel]
    # Group by hour-of-day on the filtered selection
    med = s[sel].groupby(hod_sel).median()
    mad = s[sel].groupby(hod_sel).apply(lambda x: (x - x.median()).abs().median())
    # Map back to full length
    mu = pd.Series(hod_full).map(med).to_numpy()
    sig = (1.4826 * pd.Series(hod_full).map(mad).fillna(mad.median())).to_numpy()
    z = (s.to_numpy() - mu) / (sig + 1e-6)
    return mu, sig, z

def compute_iterative_baseline(pm1: pd.Series, pm4: pd.Series, R:pd.Series, iters: int = 3) -> dict:
    """Iteratively remove outliers to get mu/sigma/z for PM1/PM4/R"""
    
    # Initial mask (coarse): high percentiles & short-term increments
    d5 = pm1 - pm1.shift(10)  # At 30s frequency, 10 points ≈ 5 minutes
    mask = (
        (pm1 > pm1.quantile(0.995)) |
        (pm4 > pm4.quantile(0.995)) |
        (d5  > d5.quantile(0.99))   |
        (R   > R.quantile(0.99))
    ).fillna(False)

    for _ in range(iters):
        mu1, s1, z1 = _robust_hourly_baseline(pm1, mask)
        mu4, s4, z4 = _robust_hourly_baseline(pm4, mask)
        muR, sR, zR = _robust_hourly_baseline(R,   mask)
        # Expand mask
        mask = mask | (np.abs(z1) > 4) | (np.abs(z4) > 4) | (zR > 3)

    out = {
        "R": R,
        "mu1": mu1, "s1": s1, "z1": z1,
        "mu4": mu4, "s4": s4, "z4": z4,
        "muR": muR, "sR": sR, "zR": zR,
    }
    return out

# ========== Helper: Fire event candidate detection (rule-based + hysteresis) ==========
def detect_fire_events(index: pd.DatetimeIndex,
                       z1: np.ndarray,
                       z4: np.ndarray,
                       R: pd.Series,
                       zR: np.ndarray,
                       pm1: pd.Series,
                       min_run: int = 3,   # Continuous trigger minutes (30s frequency => 3*2=6 points)
                       start_th_z1: float = 3.0,
                       start_th_z4: float = 2.5,
                       ratio_min: float = 0.6,
                       zR_min: float = 2.0,
                       end_th_z1: float = 1.5,
                       end_hold: int = 4   # End requires continuous end_hold*30s below threshold
                      ) -> tuple[pd.Series, list[dict]]:
    """
    Returns fire_flag(0/1, aligned to index) and events list
    """
    # 5-minute cumulative change (10 points at 30s frequency)
    d5 = pm1 - pm1.shift(10)
    q95_d5 = d5.quantile(0.95)

    start_cond = (
        ((z1 >= start_th_z1) | (z4 >= start_th_z4)) &
        (R.to_numpy() >= ratio_min) & (zR >= zR_min) &
        (d5.to_numpy() >= q95_d5)
    )
    # Require continuous >= min_run minutes (30s frequency => min_run*2 points)
    need = min_run * 2
    run = pd.Series(start_cond, index=index).rolling(need, min_periods=1).sum().to_numpy() >= need

    # Hysteresis end
    in_ev = False
    low_cnt = 0
    flag = np.zeros(len(index), dtype=bool)
    for i, (trig, z) in enumerate(zip(run, z1)):
        if not in_ev and trig:
            in_ev = True
            low_cnt = 0
        elif in_ev:
            if z < end_th_z1:
                low_cnt += 1
                if low_cnt >= end_hold:
                    in_ev = False
                    low_cnt = 0
            else:
                low_cnt = 0
        flag[i] = in_ev

    fire_flag = pd.Series(flag, index=index)

    # Event extraction
    events = []
    if fire_flag.any():
        groups = (fire_flag != fire_flag.shift()).cumsum()
        for g, sub in fire_flag[fire_flag].groupby(groups):
            t0, t1 = sub.index[0], sub.index[-1]
            seg = slice(t0, t1)
            peak_pm1 = float(pm1.loc[seg].max())
            peak_R   = float(R.loc[seg].max())
            # Simple event confidence (0-1 normalized)
            z1_seg_max = np.nanmax(z1[(index >= t0) & (index <= t1)])
            d5_seg_max = float(d5.loc[seg].max())
            # Scale based on historical percentiles
            def norm(val, qlo, qhi):
                return float(np.clip((val - qlo) / (qhi - qlo + 1e-6), 0, 1))
            z1_q05, z1_q95 = np.nanquantile(z1, [0.05, 0.95])
            d5_q05, d5_q95 = d5.quantile(0.05), d5.quantile(0.95)
            R_q05,  R_q95  = R.quantile(0.05),  R.quantile(0.95)
            S = (0.5*norm(z1_seg_max, z1_q05, z1_q95)
                 + 0.3*norm(d5_seg_max, d5_q05, d5_q95)
                 + 0.2*norm(peak_R, R_q05, R_q95))
            level = "Confirmed Fire" if S >= 0.7 else ("Potential Smoke" if S >= 0.4 else "Weak")
            events.append({
                "start": t0, "end": t1, "peak_PM1": peak_pm1, "peak_R": peak_R,
                "score": round(S, 3), "level": level
            })
    return fire_flag, events


def _robust_norm_series(series: pd.Series, lo=0.10, hi=0.90):
    """Returns (norm_func, (plo, phi)), scales to 0-1 based on given percentiles"""
    plo = np.nanpercentile(series, lo*100)
    phi = np.nanpercentile(series, hi*100)
    def _norm(x):
        return float(np.clip((x - plo) / (phi - plo + 1e-6), 0, 1))
    return _norm, (plo, phi)

def score_events(events: list[dict],
                 z_df: pd.DataFrame,      # Contains z_PM1, z_R, R, PM1
                 win_pts_5min: int = 10,  # 30s frequency → 10 points = 5min
                 use_zR: bool = False,    # True uses z_R_peak; False uses R_peak
                 weights: dict = None):
    """
    Calculate scores based on z_df and events, update events:
    Add keys: z1_peak, d5_peak, R_peak/zR_peak, dur_min, score, level
    """
    if weights is None:
        weights = dict(z1=0.50, d5=0.25, r=0.20, dur=0.05)

    # Pre-compute 5min increment series (recommend using smoothed PM1)
    d5 = z_df["PM1"] - z_df["PM1"].shift(win_pts_5min)

    # Build site-scale normalizers for each feature (using background or all data)
    # Background: non-event periods; if no fire_flag temporarily, can use all data directly
    if "fire_flag" in z_df.columns:
        bg_mask = z_df["fire_flag"] == 0
    else:
        bg_mask = pd.Series(True, index=z_df.index)

    nz1, _ = _robust_norm_series(z_df.loc[bg_mask, "z_PM1"])
    nd5, _ = _robust_norm_series(d5.loc[bg_mask].dropna())
    if use_zR:
        nr, _  = _robust_norm_series(z_df.loc[bg_mask, "z_R"])
    else:
        nr, _  = _robust_norm_series(z_df.loc[bg_mask, "R"])
    ndur, _ = _robust_norm_series(
        pd.Series([ (e["end"]-e["start"]).total_seconds()/60.0 for e in events ]) if events else pd.Series([0])
    )

    for e in events:
        t0, t1 = e["start"], e["end"]
        seg = slice(t0, t1)

        z1_peak  = float(z_df.loc[seg, "z_PM1"].max())
        d5_peak  = float(d5.loc[seg].max())
        r_peak   = float(z_df.loc[seg, "z_R"].max() if use_zR else z_df.loc[seg, "R"].max())
        dur_min  = float((t1 - t0).total_seconds()/60.0)

        s = (weights["z1"] * nz1(z1_peak) +
             weights["d5"] * nd5(d5_peak) +
             weights["r"]  * nr(r_peak)   +
             weights["dur"]* ndur(dur_min))

        e.update(dict(
            z1_peak=round(z1_peak,3),
            d5_peak=round(d5_peak,3),
            **({"zR_peak": round(r_peak,3)} if use_zR else {"R_peak": round(r_peak,3)}),
            dur_min=round(dur_min,2),
            score=round(float(s),3),
            level=("Confirmed Fire" if s>=0.70 else "Potential Smoke" if s>=0.40 else "Weak")
        ))
    return events
def main_single(shared_loader: Optional[FireDetectionDataLoader] = None) -> None:

    #########################################################
    # loading data
    #########################################################
    loader = SiteDataLoader('data', SITE_ID, loader=shared_loader)
    series = loader.load()
    # Ensure per-site output directories exist
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    #########################################################
    # input data visualization  
    #########################################################
    pm1 = series.get('PM1')
    pm4 = series.get('PM4')

    if pm1 is None or pm4 is None:
        raise SystemExit('PM1/PM4 not loaded')


    fig = plot_pm_ratio(pm1, pm4, title=f'{SITE_ID} - PM1/PM4 (30s mean)',
                        save_path=str(FIG_DIR / f"{SITE_ID}_PM1_div_PM4_30s.png"),
                        resample_freq='30S')
    pca_out = run_pca_sensors(series, n_components=3, resample_freq='30S', agg='mean', dropna='any')
    plot_pca_results(pca_out, prefix=f'{SITE_ID}_SPH_PCA', save_dir=str(FIG_DIR))
    pc1 = pca_out['scores']['PC1']
    pc2 = pca_out['scores']['PC2']

    plot_time_series({'PC1': pc1, 'PC2': pc2}, resample_freq='30S',
                    title=f'{SITE_ID} - PC1/PC2 (30s mean)',
                    save_path=str(FIG_DIR / f"{SITE_ID}_PC1_PC2_30s.png"))
    
    PM4minusPM1 = pm4 - pm1
    plot_time_series({'PM4minusPM1': PM4minusPM1}, resample_freq='30S',
                    title=f'{SITE_ID} - PM4-PM1 (30s mean)',
                    save_path=str(FIG_DIR / f"{SITE_ID}_PM4_minus_PM1_30s.png"))
    plot_time_series({'PM1': pm1}, resample_freq='30S',
                    title=f'{SITE_ID} - PM1 (30s mean)',y_min=0, y_max=20,
                    save_path=str(FIG_DIR / f"{SITE_ID}_PM1_compare_30s.png"))
    '''
    s1 = series.get('SPH_S1')
    s2 = series.get('SPH_S2')
    s3 = series.get('SPH_S3')
    s4 = series.get('SPH_S4')
    s5 = series.get('SPH_S5')
    s6 = series.get('SPH_S6')
    s7 = series.get('SPH_S7')
    s8 = series.get('SPH_S8')
    s9 = series.get('SPH_S9')
    s10 = series.get('SPH_S10')
    s11 = series.get('SPH_S11')
    s12 = series.get('SPH_S12')
    
    plot_time_series({'SPH_S1': s1, 'SPH_S2': s2, 'SPH_S3': s3, 'SPH_S4': s4, 'SPH_S5': s5, 'SPH_S6': s6, 'SPH_S7': s7, 'SPH_S8': s8, 'SPH_S9': s9, 'SPH_S10': s10, 'SPH_S11': s11, 'SPH_S12': s12}, resample_freq='30S',
                    title=f'{SITE_ID} - S1..S12 (30s mean)',
                    save_path=str(FIG_DIR / f"{SITE_ID}_S1_S12_30s.png"))

    corr_pm1, corr_pm4 = compute_site_correlation_30s('data', agg='mean')
    plot_site_correlation_heatmaps(corr_pm1, corr_pm4, save_dir=str(FIG_DIR), prefix='sites_PM_corr_30s')

    print('Saved:')
    print(' - figures/sites_PM_corr_30s_PM1.png')
    print(' - figures/sites_PM_corr_30s_PM4.png')
    '''

    #########################################################
    # input data derivative
    #########################################################
    plot_dpm1_dt(
    pm1, resample_freq='30S', method='diff',
    y_min=-1.0, y_max=1.0,
    title=f'{SITE_ID} - dPM1/dt (30s diff)',
    save_path=str(FIG_DIR / f"{SITE_ID}_dPM1_dt_30s.png"))
    plot_dpm1_dt(
    pm4, resample_freq='30S', method='diff',
    y_min=-1.0, y_max=1.0,
    title=f'{SITE_ID} - dPM4/dt (30s diff)',
    save_path=str(FIG_DIR / f"{SITE_ID}_dPM4_dt_30s.png"))
    fig = plot_dpm1_dt(pc1, resample_freq=None, method='central',
             title=f'{SITE_ID} - d(PC1)/dt', save_path=str(FIG_DIR / 'PC1_derivative.png'))

    fig = plot_dpm1_dt(pc2, resample_freq=None, method='central',
                   title=f'{SITE_ID} - d(PC2)/dt', save_path=str(FIG_DIR / 'PC2_derivative.png'))


    #########################################################
    # data alignment and preprocessing
    #########################################################
    def _to_30s(s: pd.Series) -> pd.Series:
        s = s.copy()
        s = s.sort_index().resample('30S').mean()
        s[s < 0] = np.nan
        s = s.ffill(limit=3)
        return s

    #########################################################
    # median-based smoothing, removeing small spikes
    #########################################################
    pm1_30 = pm1.sort_index().resample('30S').median().clip(lower=0).ffill(limit=3)
    pm4_30 = pm4.sort_index().resample('30S').median().clip(lower=0).ffill(limit=3)


    def smooth_online_ema(x, ema_span=6, pre_med_win='1min'):
        y = x.rolling(pre_med_win, min_periods=1).median()
        return y.ewm(span=ema_span, adjust=False).mean()

    pm1_s = smooth_online_ema(pm1_30, ema_span=6, pre_med_win='1min')   # ≈3min effective window
    pm4_s = smooth_online_ema(pm4_30, ema_span=6, pre_med_win='1min')

    
    #########################################################
    # build baseline and z-scores to remove diurnal changes and to normalize the data
    #########################################################

    R_s = compute_pm_ratio(pm1_s, pm4_s, resample_freq=None, agg='mean')
    base = compute_iterative_baseline(pm1_s, pm4_s, R=R_s, iters=3)

    #########################################################
     #detect the periods of the smoothed data
    #########################################################
    # ---- Input ----
    # pm is a 1D numpy array of your PM1_30s values (uniform 30 s sampling)
    # Align mu1 with time index for plotting
    pm1_baseline_series = pd.Series(base["mu1"], index=pm1_s.index, name="PM1_baseline")
    plot_time_series({'PM1_baseline': pm1_baseline_series}, resample_freq=None,
                    title=f'{SITE_ID} - PM1 baseline',
                    save_path=str(FIG_DIR / f"{SITE_ID}_PM1_baseline_30s.png"))    
    #########################################################
    
    z_df = pd.DataFrame({
        "z_PM1": base["z1"],
        "z_PM4": base["z4"],
        "z_R":   base["zR"],
        "R":     R_s.to_numpy(),        # Use smoothed R
        "PM1":   pm1_s.to_numpy(),      # Use smoothed PM
        "PM4":   pm4_s.to_numpy(),
    }, index=pm1_s.index)

    #########################################################
    # fire event detection
    # Metrics include PM1, PM4, R, and their z-scores  
    #########################################################
    # ---- Fire event candidate detection & event table (thresholds slightly relaxed due to smoothing) ----
    fire_flag, events = detect_fire_events(
        index=z_df.index,
        z1=z_df["z_PM1"].to_numpy(),
        z4=z_df["z_PM4"].to_numpy(),
        R=z_df["R"],
        zR=z_df["z_R"].to_numpy(),
        pm1=z_df["PM1"],
        # Can be adjusted per station
        min_run=2,           # Continuous 2 minutes (30S frequency => 4 points)
        start_th_z1=2.5,     # After smoothing, fluctuations reduced → slightly lower threshold
        start_th_z4=2.0,
        ratio_min=0.55,
        zR_min=1.0,
        end_th_z1=1.2,
        end_hold=4           # End when continuously 4×30s below threshold
    )
    z_df["fire_flag"] = fire_flag.astype(int)

    #########################################################
    # plot the z-scores and fire_flag
    #########################################################
    plot_time_series(
        {"z_PM1": z_df["z_PM1"], "z_PM4": z_df["z_PM4"], "z_R": z_df["z_R"]},
        resample_freq=None,  # Already 30s
        title=f"{SITE_ID} - Robust z-scores (30s)",
        save_path=str(FIG_DIR / f"{SITE_ID}_z_scores_30s.png")
    )
    plot_time_series(
        {"fire_flag": z_df["fire_flag"]},
        resample_freq=None,
        title=f"{SITE_ID} - Fire Flag (30s, 1=True)",
        save_path=str(FIG_DIR / f"{SITE_ID}_fire_flag_30s.png"),
        y_min=-0.1, y_max=1.1
    )

    ########################################################
    # exporting event list
    #######################################################
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ev_df = pd.DataFrame(events)
    if not ev_df.empty:
        ev_df.to_csv(str(OUT_DIR / f"{SITE_ID}_pm_fire_events.csv"), index=False)
    else:
        # Still output empty table header for process integration
        pd.DataFrame(columns=["start","end","peak_PM1","peak_R","score","level"])\
          .to_csv(str(OUT_DIR / f"{SITE_ID}_pm_fire_events.csv"), index=False)
    #########################################################
    #calculate score for each event
    #########################################################
    z_df["fire_flag"] = fire_flag.astype(int)

    events = score_events(events, z_df, win_pts_5min=10, use_zR=False)

    # export
    pd.DataFrame(events).to_csv(str(OUT_DIR / f"{SITE_ID}_pm_fire_events_scored.csv"), index=False)
    '''
#########################################################
# an IsolationForest method for anormally detection 
#########################################################


    # 1) Build features (use smoothed series & z-scores already in z_df)
    win_1m  = 2    # 30s × 2 = 1min
    win_5m  = 10   # 30s × 10 = 5min

    feat = pd.DataFrame(index=z_df.index)
    feat["PM1"] = z_df["PM1"]
    feat["PM4"] = z_df["PM4"]
    feat["R"]   = z_df["R"]
    feat["z1"]  = z_df["z_PM1"]
    feat["z4"]  = z_df["z_PM4"]
    feat["zR"]  = z_df["z_R"]

    # short & medium changes
    feat["dPM1_1"] = z_df["PM1"].diff()                       # 30s step change
    feat["dPM1_5"] = z_df["PM1"] - z_df["PM1"].shift(win_5m)  # 5min change

    # robust local range (captures bursts)
    feat["PM1_roll_max1m"] = z_df["PM1"].rolling(win_1m, min_periods=1).max()
    feat["PM1_roll_min1m"] = z_df["PM1"].rolling(win_1m, min_periods=1).min()
    feat["PM1_range1m"]    = feat["PM1_roll_max1m"] - feat["PM1_roll_min1m"]

    feat = (feat.replace([np.inf, -np.inf], np.nan)
                .fillna(method="ffill")
                .fillna(method="bfill"))

    # 2) Define conservative "background" (no labels needed)
    bg_mask = (
        (feat["z1"].abs() < 1.5) &
        (feat["z4"].abs() < 1.5) &
        (feat["zR"] < 1.0) &
        (feat["R"] < 0.65)
    ).fillna(False)
    if bg_mask.sum() < 200:  # fallback if too strict
        bg_mask = ((feat["z1"].abs() < 2.0) & (feat["z4"].abs() < 2.0)).fillna(False)

    X_bg = feat[bg_mask].values
    X_all = feat.values

    # 3) Train IF on background only; score all minutes
    iso = IsolationForest(
        n_estimators=300, max_samples="auto",
        contamination=0.01, random_state=0, n_jobs=-1
    )
    iso.fit(X_bg)

    # scikit-learn IF: higher anomaly -> more negative score; invert so bigger=more anomalous
    raw = -iso.score_samples(X_all)

    # 4) Normalize to 0–1 using background quantiles (stable)
    p_lo = np.percentile(raw[bg_mask.to_numpy()], 90)     # 90th of background
    p_hi = np.percentile(raw[bg_mask.to_numpy()], 99.5)   # 99.5th of background
    ml_score = np.clip((raw - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)

    z_df["if_score"] = ml_score  # minute-level IF score (0–1)

    # 5) Minute IF flag with continuity + hysteresis (independent of rule-based)
    IF_THR_ENTER = 0.70     # enter threshold (tune 0.65–0.8)
    IF_THR_EXIT  = 0.55     # exit threshold to add hysteresis
    MIN_RUN_MIN  = 2        # require >= 2 minutes continuous to form an event
    PTS_PER_MIN  = 2        # because your series is 30s
    NEED_POINTS  = MIN_RUN_MIN * PTS_PER_MIN

    cand = (z_df["if_score"] >= IF_THR_ENTER).astype(int)
    cand_run = cand.rolling(NEED_POINTS, min_periods=1).sum() >= NEED_POINTS

    # hysteresis on the score itself
    in_ev = False
    flag  = np.zeros(len(z_df), dtype=bool)
    for i, s in enumerate(z_df["if_score"].to_numpy()):
        if not in_ev and cand_run.iloc[i]:
            in_ev = True
        elif in_ev and s < IF_THR_EXIT:
            in_ev = False
        flag[i] = in_ev

    z_df["if_flag"] = flag.astype(int)

    # 6) Extract IF events from if_flag
    def _events_from_flag(ts_index: pd.DatetimeIndex, flag_series: pd.Series) -> list[dict]:
        evs = []
        if flag_series.any():
            groups = (flag_series != flag_series.shift()).cumsum()
            for g, sub in flag_series[flag_series == 1].groupby(groups):
                t0, t1 = sub.index[0], sub.index[-1]
                seg = slice(t0, t1)
                # a few basic metrics for comparison
                evs.append({
                    "start": t0, "end": t1,
                    "if_score_median": float(z_df.loc[seg, "if_score"].median()),
                    "if_score_peak":   float(z_df.loc[seg, "if_score"].max()),
                    "PM1_peak":        float(z_df.loc[seg, "PM1"].max()),
                    "R_peak":          float(z_df.loc[seg, "R"].max()),
                    "dur_min":         float((t1 - t0).total_seconds()/60.0),
                })
        return evs

    if_events = _events_from_flag(z_df.index, z_df["if_flag"])

    # 7) Export IF-only outputs (separate from rule-based)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # per-minute: include IF score/flag for side-by-side comparison
    z_df[["if_score", "if_flag"]].to_csv(str(OUT_DIR / f"{SITE_ID}_if_per_minute.csv"))

    # event list (IF)
    pd.DataFrame(if_events).to_csv(str(OUT_DIR / f"{SITE_ID}_if_events.csv"), index=False)

    # 8) Optional quick plots for comparison (reuse your plot_time_series)
    plot_time_series(
        {"if_score": z_df["if_score"]},
        resample_freq=None,
        title=f"{SITE_ID} - IsolationForest score (0–1, 30s)",
        save_path=str(FIG_DIR / f"{SITE_ID}_if_score_30s.png"),
    )
    plot_time_series(
        {"if_flag": z_df["if_flag"]},
        resample_freq=None,
        title=f"{SITE_ID} - IsolationForest Flag (30s, 1=True)",
        save_path=str(FIG_DIR / f"{SITE_ID}_if_flag_30s.png"),
        y_min=-0.1, y_max=1.1
    )
    '''

    '''
    #########################################################
    # blend the PM metrics and the ML metrics
    #########################################################
    # This fills e["ml_score_median"] and e["ml_score_peak"] from z_df["if_score"]
    for e in events:
        t0, t1 = pd.to_datetime(e["start"]), pd.to_datetime(e["end"])
        seg = slice(t0, t1)
        seg_if = z_df.loc[seg, "if_score"] if "if_score" in z_df.columns else pd.Series(dtype=float)

        if seg_if.size > 0 and seg_if.notna().any():
            e["ml_score_median"] = float(seg_if.median())
            e["ml_score_peak"]   = float(seg_if.max())
        else:
            # safe defaults if no overlap or missing scores
            e["ml_score_median"] = 0.0
            e["ml_score_peak"]   = 0.0
    
    for e in events:
        s_pm = e.get("score", 0.0)
        s_ml = e["ml_score_median"]
        # blend (weights are tunable)
        e["score_blend"] = round(0.7*s_pm + 0.3*s_ml, 3)
        e["level"] = ("Confirmed Fire" if e["score_blend"] >= 0.70
                    else "Potential Smoke" if e["score_blend"] >= 0.40
                    else "Weak")
    ev_df = pd.DataFrame(events)

    if not ev_df.empty and "score_blend" in ev_df:
        plt.figure(figsize=(10,5))
        plt.bar(ev_df.index.astype(str), ev_df["score_blend"], color="orange")
        plt.axhline(0.70, color="red", linestyle="--", label="Confirmed threshold")
        plt.axhline(0.40, color="blue", linestyle="--", label="Potential threshold")
        plt.ylim(0,1.05)
        plt.xlabel("Event ID")
        plt.ylabel("Score Blend")
        plt.title(f"{SITE_ID} - Event Score Blend")
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(FIG_DIR / f"{SITE_ID}_events_score_blend.png"))
        plt.close()
    #########################################################
    # exporting event list
    #########################################################
    pd.DataFrame(events).to_csv(str(OUT_DIR / f"{SITE_ID}_events_blended.csv"), index=False)
    '''
def main(site_ids: Optional[list] = None) -> None:
    """Run processing for all sites (or a provided subset)."""
    global SITE_ID, FIG_DIR, OUT_DIR
    # Determine sites
    if site_ids is None:
        loader_global = FireDetectionDataLoader('data')
        # Preload all data once for reuse across sites
        loader_global.load_all_data()
        loc = loader_global.location_info
        # Filter ACTIVE stations only; drop NaN/invalid rows
        if loc is not None and 'StationID' in loc.columns and 'Status' in loc.columns:
            active = loc[loc['Status'].astype(str).str.strip().str.lower() == 'active']
            site_ids = (
                active['StationID']
                .dropna()
                .astype(str)
                .map(lambda s: s.strip())
                .tolist()
            )
        else:
            site_ids = []
    # Fallback to current SITE_ID if list is empty
    if not site_ids:
        site_ids = [SITE_ID]

    for sid in site_ids:
        print(f"\n===== Processing site: {sid} =====")
        # Update globals used throughout the pipeline
        SITE_ID = sid
        FIG_DIR = Path('figures') / SITE_ID
        OUT_DIR = Path('outputs') / SITE_ID
        main_single(shared_loader=loader_global)

if __name__ == '__main__':
    main()