# fire_detection_pipeline_tspulse_v2.py
# -*- coding: utf-8 -*-
"""
Fire detection with rule-based pipeline (+30s preprocessing) blended with
TSPulse reconstruction anomaly. TSPulse scoring code is kept intact; the
rest (preprocess, features, detection, scoring) is from your latest pipeline.

Inputs (wide format):
  - PM1.csv: columns [timestamp_utc, <station1>, <station2>, ...]
  - PM4.csv: columns [timestamp_utc, <station1>, <station2>, ...]
  - ENV.csv: (optional; used for RH veto if contains 'RH'/'Humidity')
  - Location.csv: (optional; StationID, Location="lat,lon")

Outputs (./output):
  - events_out_tspulse.csv                 (rule-only metrics + TSPulse stats)
  - events_out_with_tspulse_blend.csv     (blended score/level)
  - timeline_out_tspulse.csv              (timeline: cand, Z, slopeZ)
  - tspulse_per_minute.csv                (per-station TSPulse time series)
  - tspulse_thresholds.csv                (per-station enter/exit thresholds)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.optimize import least_squares

# -----------------------------
# Config dataclasses
# -----------------------------

@dataclass
class ColumnMap:
    time: str = "timestamp_utc"

@dataclass
class PreprocessCfg:
    # 30s median resample + EMA smoothing (≈3 min eff.)
    resample_rule: str = "30s"
    ema_span: int = 6
    pre_med_win: str = "1min"
    min_valid_frac: float = 0.5
    ffill_limit: int = 3

@dataclass
class BackgroundCfg:
    roll_minutes: int = 90
    lowpass_minutes: int = 15

@dataclass
class DetectCfg:
    z_trigger: float = 5.0
    z_reset: float = 3.0
    slope_q: float = 0.90
    mad_win_min: int = 60
    min_duration_min: int = 2
    z4_trigger: float = 5.0
    pmratio_min: float = 1.05
    pmratio_z_min: float = 0.5
    max_time_lag_min: int = 15

@dataclass
class VetoCfg:
    rh_high: float = 85.0
    all_station_sync_span_min: int = 10

@dataclass
class ScoreWeights:
    w_z: float = 0.6
    w_slope: float = 0.4
    penalty_veto: float = 0.5

# ---- TSPulse config (unchanged) ----
@dataclass
class TSPulseCfg:
    win_min: int = 512
    stride_min: int = 4
    use_quantile_thr: bool = True
    enter_q: float = 0.99
    exit_q: float = 0.95
    enter_thr_fixed: float = 0.70
    exit_thr_fixed: float = 0.55
    min_run: int = 10
    use_multi_channel: bool = True

# -----------------------------
# Utilities
# -----------------------------

def _to_30s_median_ffill(s: pd.Series, ffill_limit: int) -> pd.Series:
    s = s.sort_index().resample("30s").median()
    s[s < 0] = np.nan
    return s.ffill(limit=ffill_limit)

def _smooth_online_ema(x: pd.Series, ema_span=6, pre_med_win="1min") -> pd.Series:
    y = x.rolling(pre_med_win, min_periods=1).median()
    return y.ewm(span=ema_span, adjust=False).mean()

def _rolling_mad(x: pd.Series, win: int) -> pd.Series:
    def mad(a: np.ndarray) -> float:
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return np.nan
        return np.median(np.abs(a - np.median(a))) * 1.4826
    win = max(win, 3)
    return x.rolling(win, min_periods=max(3, win // 5)).apply(mad, raw=False)

def _infer_dt_seconds(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2:
        return 60.0
    diffs = np.diff(idx.view("i8")) / 1e9
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 60.0
    return float(np.median(diffs))

def _cross_corr_lag(x: np.ndarray, y: np.ndarray, dt_sec: float, max_lag_min: int) -> Optional[float]:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 5:
        return None
    x0, y0 = x[mask], y[mask]
    x0 = (x0 - x0.mean()) / (x0.std() + 1e-9)
    y0 = (y0 - y0.mean()) / (y0.std() + 1e-9)
    max_lag = int(round(max_lag_min * 60 / dt_sec))
    corr = correlate(y0, x0, mode="full")
    lags = np.arange(-len(y0)+1, len(x0))
    mask2 = (lags >= -max_lag) & (lags <= max_lag)
    if mask2.sum() == 0:
        return None
    lag_samples = lags[mask2][np.argmax(corr[mask2])]
    return lag_samples * dt_sec / 60.0

# -----------------------------
# TSPulse adapter (kept intact)
# -----------------------------

TSPULSE_REPO = "ibm-granite/granite-timeseries-tspulse-r1"
TSPULSE_REV  = "main"

def _load_tspulse_model():
    try:
        import torch
        from tsfm_public.models.tspulse import TSPulseForReconstruction
        model = TSPulseForReconstruction.from_pretrained(TSPULSE_REPO, revision=TSPULSE_REV)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = model.to(device).eval()
        return model, device, False
    except ImportError as e:
        raise ImportError(
            f"TSPulse model loading failed: {e}\n"
            'Install in py3.9–3.12:\n  pip install "granite-tsfm @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.3.1"'
        )
    except Exception as e:
        raise RuntimeError(
            f"TSPulse model loading failed: {e}\nCheck your granite-tsfm installation."
        )

def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.median()
    mad = (df - mu).abs().median() * 1.4826
    mad = mad.replace(0, 1.0)
    return (df - mu) / (mad + 1e-6)

def _make_windows(x: np.ndarray, win: int, stride: int) -> np.ndarray:
    T = len(x)
    idx = list(range(0, max(T - win + 1, 0), stride))
    if not idx:
        return np.zeros((0, win, x.shape[1]), dtype=np.float32)
    return np.stack([x[i:i+win] for i in idx], axis=0)

def _per_station_tspulse_scores(
    st_df: pd.DataFrame,
    model,
    device,
    cfg: TSPulseCfg
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    import torch
    X_full = _standardize_cols(st_df).astype("float32").to_numpy()  # (T, F)
    T, F = X_full.shape
    if T < cfg.win_min:
        out = pd.DataFrame(index=st_df.index, data={
            "tspulse_score": np.zeros(T),
            "tspulse_flag":  np.zeros(T, dtype=int)
        })
        return out, {"enter": np.nan, "exit": np.nan}

    # (kept single-channel to match TSPulse API expectations)
    Xw = _make_windows(X_full[:, 0:1], cfg.win_min, cfg.stride_min)  # (B,W,1)
    x = torch.from_numpy(Xw).to(device)

    with torch.no_grad():
        out = model(x)
        if isinstance(out, dict):
            x_hat = out.get("reconstruction_outputs", out.get("reconstruction", None))
        else:
            x_hat = out
        if x_hat is None:
            raise RuntimeError("TSPulse forward() returned no reconstruction tensor")
        if x_hat.dim() == 2:
            x_hat = x_hat.unsqueeze(-1)
        err = torch.nn.functional.l1_loss(x_hat, x, reduction="none").mean(dim=2).cpu().numpy()  # (B,W)

    err_ts = np.zeros(T, dtype=np.float32); cnt = np.zeros(T, dtype=np.float32)
    B, W = err.shape
    for b in range(B):
        s = b * cfg.stride_min
        e = s + W
        err_ts[s:e] += err[b]
        cnt[s:e]    += 1
    err_ts = err_ts / np.maximum(cnt, 1.0)

    p_lo = np.nanpercentile(err_ts, 90)
    p_hi = np.nanpercentile(err_ts, 99.5)
    score = np.clip((err_ts - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)

    if cfg.use_quantile_thr:
        enter = float(np.nanquantile(score, cfg.enter_q))
        exit_ = float(np.nanquantile(score, cfg.exit_q))
    else:
        enter = cfg.enter_thr_fixed
        exit_ = cfg.exit_thr_fixed

    in_ev = False; flag = np.zeros(T, dtype=int)
    for i, s in enumerate(score):
        if (not in_ev) and (s >= enter):
            in_ev = True
        elif in_ev and (s < exit_):
            in_ev = False
        flag[i] = int(in_ev)
    run = pd.Series(flag, index=st_df.index).rolling(cfg.min_run, min_periods=1).sum() >= cfg.min_run
    flag = run.astype(int).to_numpy()

    out = pd.DataFrame(index=st_df.index, data={"tspulse_score": score, "tspulse_flag": flag})
    return out, {"enter": enter, "exit": exit_}

def _features_for_tspulse(pm_wide: pd.DataFrame,
                          feats: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out = {}
    pm1 = pm_wide["PM1"]
    pm4 = pm_wide["PM4"]
    ratio = feats["pmratio"]
    Z1 = feats["Z"]["PM1"]
    Z4 = feats["Z"]["PM4"]
    slopeZ1 = feats["slopeZ"]["PM1"]

    for st in pm1.columns:
        df = pd.DataFrame(index=pm1.index)
        df["PM1"] = pm1[st]
        df["PM4"] = pm4[st] if st in pm4.columns else np.nan
        df["R"]   = ratio[st] if st in ratio.columns else np.nan
        df["z_PM1"] = Z1[st]
        df["z_PM4"] = Z4[st]
        df["slopeZ_PM1"] = slopeZ1[st]
        df = df.replace([np.inf, -np.inf], np.nan).interpolate(limit=3).ffill().bfill()
        out[st] = df
    return out

def compute_tspulse_for_all(
    pm_wide: pd.DataFrame,
    feats: Dict[str, pd.DataFrame],
    tsp_cfg: TSPulseCfg
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model, device, _ = _load_tspulse_model()
    st_feats = _features_for_tspulse(pm_wide, feats)
    out_list = []; thr_rows = []
    for st, df in st_feats.items():
        st_res, thr = _per_station_tspulse_scores(df, model, device, tsp_cfg)
        st_res.columns = pd.MultiIndex.from_product([[st], st_res.columns], names=["station", "var"])
        out_list.append(st_res)
        thr_rows.append({"station": st, "enter": thr["enter"], "exit": thr["exit"]})
    if not out_list:
        return pd.DataFrame(index=pm_wide.index), pd.DataFrame(columns=["station","enter","exit"])
    tspulse_wide = pd.concat(out_list, axis=1).reindex(index=pm_wide.index)
    thr_df = pd.DataFrame(thr_rows).sort_values("station")
    return tspulse_wide, thr_df

# -----------------------------
# Rule pipeline (from your latest code)
# -----------------------------

class RuleFireDetector:
    def __init__(self,
                 pre: PreprocessCfg = PreprocessCfg(),
                 bg: BackgroundCfg = BackgroundCfg(),
                 det: DetectCfg = DetectCfg(),
                 veto: VetoCfg = VetoCfg(),
                 weights: ScoreWeights = ScoreWeights()):
        self.pre = pre; self.bg = bg; self.det = det
        self.veto = veto; self.weights = weights

    def load_and_prep(self,
                      pm1_df: pd.DataFrame,
                      pm4_df: pd.DataFrame,
                      env_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        ctime = "timestamp_utc"
        pm1_df[ctime] = pd.to_datetime(pm1_df[ctime], errors="coerce", utc=True)
        pm4_df[ctime] = pd.to_datetime(pm4_df[ctime], errors="coerce", utc=True)
        pm1_df = pm1_df.dropna(subset=[ctime]).set_index(ctime).sort_index()
        pm4_df = pm4_df.dropna(subset=[ctime]).set_index(ctime).sort_index()

        if env_df is not None:
            env_df[ctime] = pd.to_datetime(env_df[ctime], errors="coerce", utc=True)
            env_df = env_df.dropna(subset=[ctime]).set_index(ctime).sort_index()

        pm1_proc = {st: _smooth_online_ema(_to_30s_median_ffill(pm1_df[st], self.pre.ffill_limit),
                                           self.pre.ema_span, self.pre.pre_med_win)
                    for st in pm1_df.columns}
        pm4_proc = {st: _smooth_online_ema(_to_30s_median_ffill(pm4_df[st], self.pre.ffill_limit),
                                           self.pre.ema_span, self.pre.pre_med_win)
                    for st in pm4_df.columns}

        pm1_df = pd.concat(pm1_proc, axis=1) if pm1_proc else pd.DataFrame()
        pm4_df = pd.concat(pm4_proc, axis=1) if pm4_proc else pd.DataFrame()

        common_stations = sorted(set(pm1_df.columns).intersection(pm4_df.columns))
        pm1_df = pm1_df[common_stations]; pm4_df = pm4_df[common_stations]
        common_idx = pm1_df.index.intersection(pm4_df.index)
        pm1_df = pm1_df.loc[common_idx]; pm4_df = pm4_df.loc[common_idx]

        if not pm1_df.empty:
            keep = pm1_df.notna().mean()
            keep = keep[keep >= self.pre.min_valid_frac].index
            pm1_df = pm1_df[keep]; pm4_df = pm4_df[keep]

        pm_wide = pd.concat({"PM1": pm1_df, "PM4": pm4_df}, axis=1)

        # pass env wide-through (optional)
        env_wide = env_df.copy() if env_df is not None and len(env_df.columns) > 0 else None
        return pm_wide, env_wide

    def estimate_background(self, pm_wide: pd.DataFrame) -> pd.DataFrame:
        dt_sec = _infer_dt_seconds(pm_wide.index)
        k_lp = max(1, int(round(self.bg.lowpass_minutes * 60 / dt_sec)))
        roll = f"{self.bg.roll_minutes}min"
        BG = {}
        for metric in ["PM1", "PM4"]:
            W = pm_wide[metric]
            bg = W.median(axis=1, skipna=True).rolling(roll, min_periods=5).median()
            bg = bg.rolling(k_lp, min_periods=1, center=True).mean()
            BG[metric] = bg
        return pd.DataFrame(BG, index=pm_wide.index)

    def compute_features(self, pm_wide: pd.DataFrame, bg_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        dt_sec = _infer_dt_seconds(pm_wide.index)
        dt_min = dt_sec / 60.0
        feats: Dict[str, pd.DataFrame] = {}
        resid = {}; Z = {}; slope = {}; slopeZ = {}
        mad_win_samples = max(3, int(round(self.det.mad_win_min * 60 / dt_sec)))

        for metric in ["PM1", "PM4"]:
            W = pm_wide[metric]
            r = (W.T - bg_df[metric].values).T
            resid[metric] = r
            mad_r = r.apply(lambda s: _rolling_mad(s, mad_win_samples), axis=0)
            Z[metric] = r / (mad_r + 1e-6)
            dW = W.diff().fillna(0) / max(dt_min, 1e-9)
            slope[metric] = dW
            mad_slope = dW.apply(lambda s: _rolling_mad(s, mad_win_samples), axis=0)
            slopeZ[metric] = dW / (mad_slope + 1e-6)

        feats["resid"]  = pd.concat(resid, axis=1)
        feats["Z"]      = pd.concat(Z, axis=1)
        feats["slope"]  = pd.concat(slope, axis=1)
        feats["slopeZ"] = pd.concat(slopeZ, axis=1)
        feats["pmratio"] = pm_wide["PM1"] / pm_wide["PM4"].replace(0, np.nan)
        return feats

    def detect_events(self,
                      pm_wide: pd.DataFrame,
                      feats: Dict[str, pd.DataFrame],
                      env_wide: Optional[pd.DataFrame] = None
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dt_sec = _infer_dt_seconds(pm_wide.index)
        dt_min = dt_sec / 60.0
        idx = pm_wide.index
        stations_list = list(pm_wide["PM1"].columns)

        Z1 = feats["Z"]["PM1"]
        Z4 = feats["Z"]["PM4"]
        slopeZ1 = feats["slopeZ"]["PM1"]
        pmratio = feats["pmratio"]

        def _rz(s: pd.Series) -> pd.Series:
            med = s.median()
            mad = np.median(np.abs(s - med)) * 1.4826
            return (s - med) / (mad + 1e-6)
        pmratio_z = pmratio.apply(_rz)

        slope_thr = slopeZ1.quantile(self.det.slope_q)
        cand = (
            (Z1.ge(self.det.z_trigger) | Z4.ge(self.det.z4_trigger)) &
            (slopeZ1.ge(slope_thr)) &
            (pmratio.ge(self.det.pmratio_min) | pmratio_z.ge(self.det.pmratio_z_min))
        )

        min_run_samples = max(1, int(round(self.det.min_duration_min * 60 / dt_sec)))

        def consec_persist(s: pd.Series) -> pd.Series:
            arr = s.astype(bool).to_numpy()
            out = np.zeros_like(arr, dtype=bool)
            run = 0
            for i, v in enumerate(arr):
                if v:
                    run += 1
                else:
                    if run >= min_run_samples:
                        out[i-run:i] = True
                    run = 0
            if run >= min_run_samples:
                out[len(arr)-run:] = True
            return pd.Series(out, index=s.index)

        cand = cand.apply(consec_persist)

        # timeline
        timeline = pd.DataFrame(index=idx)
        for s in stations_list:
            timeline[(s, "cand")] = cand[s].astype(int)
            timeline[(s, "Z")] = Z1[s]
            timeline[(s, "slopeZ")] = slopeZ1[s]
        timeline.columns = pd.MultiIndex.from_tuples(timeline.columns, names=["station","var"])

        # group contiguous segments (any station active)
        any_cand = cand.any(axis=1)
        events: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        active = False; start_t = None
        for t, flag in any_cand.items():
            if flag and not active:
                active = True; start_t = t
            elif active and not flag:
                events.append((start_t, prev_t))
                active = False
            prev_t = t
        if active:
            events.append((start_t, prev_t))

        # RH veto (if ENV has an RH-like column)
        rh_bg = None
        if env_wide is not None:
            rh_like = [c for c in env_wide.columns if str(c).strip().upper() in ["RH", "HUMIDITY", "RELHUM"]]
            if rh_like:
                rh_bg = env_wide[rh_like[0]].median(axis=1, skipna=True)

        rows = []
        for eid, (t0, t1) in enumerate(events, start=1):
            win = slice(t0, t1)
            act = [s for s in stations_list if cand.loc[win, s].any()]
            if len(act) == 0:
                rows.append({"event_id": eid, "t_start": t0, "t_end": t1,
                             "duration_min": (t1-t0).total_seconds()/60.0,
                             "stations_involved": "", "Z_peak": np.nan,
                             "slopeZ_peak": np.nan, "pmratio_delta_med": np.nan,
                             "veto_rh": False, "veto_sync": False,
                             "score": 0.0, "level": "Weak"})
                continue

            Zp = Z1.loc[win, act].max().max()
            slopep = slopeZ1.loc[win, act].max().max()

            pmr_window = pmratio.loc[win, act].median()
            start = t0 - pd.Timedelta("180min")
            pmr_hist = pmratio.loc[start:t0, act]
            pmr_baseline = pmr_hist.median() if not pmr_hist.empty else np.nan
            pmr_delta = (pmr_window - pmr_baseline).median() if isinstance(pmr_baseline, pd.Series) else np.nan

            rh_high = False
            if rh_bg is not None:
                rh_win = rh_bg.loc[win]
                if not rh_win.empty and rh_win.median() >= self.veto.rh_high:
                    rh_high = True

            onsets = []
            for s in act:
                ts = cand[s].loc[win]
                if ts.any():
                    onsets.append(ts.idxmax())
            sync_veto = False
            if len(onsets) >= max(3, len(stations_list)//2):
                if (max(onsets) - min(onsets)).total_seconds()/60.0 <= self.veto.all_station_sync_span_min:
                    sync_veto = True

            veto_flag = rh_high or sync_veto

            score = (self.weights.w_z * np.tanh((Zp or 0)/10.0) +
                     self.weights.w_slope * np.tanh((slopep or 0)/10.0))
            if veto_flag:
                score *= self.weights.penalty_veto
            score = float(np.clip(score, 0, 1))
            level = "Level-1" if score < 0.6 or veto_flag else "Level-2"

            rows.append({
                "event_id": eid, "t_start": t0, "t_end": t1,
                "duration_min": (t1 - t0).total_seconds()/60.0,
                "stations_involved": ",".join(act),
                "Z_peak": float(Zp) if pd.notna(Zp) else np.nan,
                "slopeZ_peak": float(slopep) if pd.notna(slopep) else np.nan,
                "pmratio_delta_med": float(pmr_delta) if pd.notna(pmr_delta) else np.nan,
                "veto_rh": rh_high, "veto_sync": sync_veto,
                "score": score, "level": level
            })

        events_df = (pd.DataFrame(rows)
                     .sort_values("t_start")
                     .reset_index(drop=True)
                     if rows else
                     pd.DataFrame(columns=["event_id","t_start","t_end","duration_min",
                                           "stations_involved","Z_peak","slopeZ_peak",
                                           "pmratio_delta_med","veto_rh","veto_sync",
                                           "score","level"]))
        return events_df, timeline

# -----------------------------
# Runner (merge + blend)
# -----------------------------

from pathlib import Path
import argparse

def run_pipeline_with_tspulse_v2(pm1_csv: str,
                                 pm4_csv: str,
                                 env_csv: Optional[str] = None,
                                 stations_csv: Optional[str] = None,
                                 tsp_cfg: Optional[TSPulseCfg] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tsp_cfg = tsp_cfg or TSPulseCfg()

    # load CSVs
    pm1_df = pd.read_csv(pm1_csv, comment="#")
    pm4_df = pd.read_csv(pm4_csv, comment="#")
    env_df = pd.read_csv(env_csv, comment="#") if env_csv else None
    stations_df = pd.read_csv(stations_csv) if stations_csv else None

    # rule pipeline pieces
    rule = RuleFireDetector()
    pm_wide, env_wide = rule.load_and_prep(pm1_df, pm4_df, env_df)
    BG = rule.estimate_background(pm_wide)
    feats = rule.compute_features(pm_wide, BG)
    events, timeline = rule.detect_events(pm_wide, feats, env_wide)

    # TSPulse per-station scores (unchanged code-path)
    tspulse_wide, thr_df = compute_tspulse_for_all(pm_wide, feats, tsp_cfg)

    Path("output").mkdir(parents=True, exist_ok=True)

    # save tspulse per-minute & thresholds
    if not tspulse_wide.empty:
        tspulse_flat = tspulse_wide.copy()
        tspulse_flat.columns = [f"{a}_{b}" for a,b in tspulse_flat.columns.to_list()]
        tspulse_flat.to_csv("output/tspulse_per_minute.csv")
    else:
        tspulse_flat = pd.DataFrame(index=pm_wide.index)
        tspulse_flat.to_csv("output/tspulse_per_minute.csv")
    thr_df.to_csv("output/tspulse_thresholds.csv", index=False)

    # fuse tspulse into event table
    if not events.empty and not tspulse_wide.empty:
        sc_cols = [c for c in tspulse_wide.columns if c[1] == "tspulse_score"]
        scores_by_station = tspulse_wide[sc_cols]
        scores_by_station.columns = [c[0] for c in sc_cols]  # station names

        for i, row in events.iterrows():
            t0, t1 = row["t_start"], row["t_end"]
            act = str(row.get("stations_involved","")).split(",") if pd.notna(row.get("stations_involved")) else []
            act = [s for s in act if s in scores_by_station.columns]
            win = slice(t0, t1)
            if len(act) == 0:
                s_med = np.nan; s_pk = np.nan
            else:
                sub = scores_by_station.loc[win, act]
                s_med = float(np.nanmedian(sub.values)) if sub.size else np.nan
                s_pk  = float(np.nanmax(sub.values)) if sub.size else np.nan
            events.loc[i, "tspulse_score_median"] = round(s_med, 3) if not np.isnan(s_med) else np.nan
            events.loc[i, "tspulse_score_peak"]   = round(s_pk, 3)  if not np.isnan(s_pk)  else np.nan

        # Blend (rule 0.7 + TSPulse 0.3)
        events["score_blend"] = (0.7 * events["score"].fillna(0.0) +
                                 0.3 * events["tspulse_score_median"].fillna(0.0)).clip(0,1)
        events["level_blend"] = events["score_blend"].apply(lambda s: "Level-2" if s >= 0.70
                                                            else ("Level-1" if s >= 0.40 else "Weak"))

    # Save outputs
    events.to_csv("output/events_out_tspulse.csv", index=False)
    events.to_csv("output/events_out_with_tspulse_blend.csv", index=False)
    timeline.to_csv("output/timeline_out_tspulse.csv")

    # optional: basic localization using rule features & stations (if provided)
    if stations_csv:
        try:
            sdf = stations_df.copy().set_index("StationID")
            def parse_loc(loc):
                try:
                    lat, lon = map(float, str(loc).split(","))
                    return lat, lon
                except:
                    return np.nan, np.nan
            latlon = sdf["Location"].apply(parse_loc)
            sdf["lat"] = latlon.apply(lambda t: t[0])
            sdf["lon"] = latlon.apply(lambda t: t[1])
            stations = sdf[["lat", "lon"]].dropna()
        except Exception:
            stations = None
    else:
        stations = None

    if stations is not None and not stations.empty and not events.empty:
        # simple localization reusing rule features
        idx = feats["resid"]["PM1"].index
        results = []
        for _, row in events.iterrows():
            t0, t1 = row["t_start"], row["t_end"]
            win = slice(t0, t1)
            resid = feats["resid"]["PM1"].loc[win, stations.index.intersection(feats["resid"]["PM1"].columns)]
            slopeZ = feats["slopeZ"]["PM1"].loc[win, resid.columns]
            arrivals = {}
            for s in resid.columns:
                s_ser = slopeZ[s]
                if s_ser.notna().any():
                    arrivals[s] = s_ser.idxmax()
            if len(arrivals) < 3:
                results.append({**row, "x0": np.nan, "y0": np.nan})
                continue
            t0min = {s: (t - t0).total_seconds()/60.0 for s, t in arrivals.items()}

            XY = stations.loc[list(t0min.keys()), ["lat","lon"]].to_numpy()
            lat0 = XY[:,0].mean()
            km_per_deg_lat = 110.574
            km_per_deg_lon = 111.320 * np.cos(np.deg2rad(lat0))
            XY_km = np.column_stack([(XY[:,0]-lat0)*km_per_deg_lat,
                                     (XY[:,1]-stations["lon"].mean())*km_per_deg_lon])
            t_vec = np.array([t0min[s] for s in t0min.keys()])

            def residuals(p):
                x,y,t_src,c = p
                d = np.sqrt((XY_km[:,0]-x)**2 + (XY_km[:,1]-y)**2)
                return t_vec - (t_src + d/np.maximum(c,1e-3))

            p0 = np.array([0.0, 0.0, 0.0, 0.5])
            res = least_squares(residuals, p0, bounds=([-5,-5,-10,0.05],[5,5,10,5]))
            x,y,t_src,c = res.x
            lat_est = lat0 + x/km_per_deg_lat
            lon_est = stations["lon"].mean() + y/km_per_deg_lon
            results.append({**row, "x0": lat_est, "y0": lon_est,
                            "t_src_min": t_src, "c_km_per_min": c, "loc_cost": res.cost})
        loc_df = pd.DataFrame(results)
        if not loc_df.empty:
            loc_df.to_csv("output/events_located.csv", index=False)

    return events, timeline, tspulse_wide if 'tspulse_wide' in locals() else pd.DataFrame(index=pm_wide.index)

# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm1", required=True, help="Path to PM1 wide CSV",default="./data/PM1.csv")
    ap.add_argument("--pm4", required=True, help="Path to PM4 wide CSV",default="./data/PM4.csv")
    ap.add_argument("--env", default=None, help="Optional ENV CSV")
    ap.add_argument("--loc", default=None, help="Optional Location.csv (StationID, Location='lat,lon')")
    args = ap.parse_args()

    try:
        events, timeline, tsp = run_pipeline_with_tspulse_v2(
            pm1_csv=args.pm1, pm4_csv=args.pm4, env_csv=args.env, stations_csv=args.loc
        )
        print("Saved:")
        print(" - output/events_out_tspulse.csv")
        print(" - output/events_out_with_tspulse_blend.csv")
        print(" - output/timeline_out_tspulse.csv")
        print(" - output/tspulse_per_minute.csv")
        print(" - output/tspulse_thresholds.csv")
        if Path("output/events_located.csv").exists():
            print(" - output/events_located.csv")
    except Exception as e:
        print("Run failed:", e)
