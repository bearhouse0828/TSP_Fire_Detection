# fire_detection_pipeline.py
# -*- coding: utf-8 -*-
"""
Fire Detection from PM sensor network (30s preprocessing, PM1+PM4)
Steps:
B. Preprocessing (30s median + EMA)
C. Common background estimation
D. Local anomaly features per station
E. Spatial timing checks (kept simple)
F. Alert levels & scoring
G. Optional rough localization

Inputs (wide format):
  - input/PM1.csv: columns [timestamp_utc, <station1>, <station2>, ...]
  - input/PM4.csv: columns [timestamp_utc, <station1>, <station2>, ...]
  - input/ENV.csv: (optional; same shape; used only for RH veto if present)
  - input/Location.csv: (optional; StationID, Location="lat,lon")

Outputs (in ./multi_output):
  - events_out.csv
  - timeline_out.csv
  - events_located.csv (if stations provided)

Author: ChatGPT for Jingjing
Date: 2025-10-03
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.optimize import least_squares

# -----------------------------
# Configuration dataclasses
# -----------------------------

@dataclass
class ColumnMap:
    time: str = "timestamp_utc"      # timestamp column (parseable to datetime)
    station: str = "station"
    pm1: str = "PM1"
    pm4: str = "PM4"
    rh: Optional[str] = "RH"
    temp: Optional[str] = "T"

@dataclass
class PreprocessCfg:
    # You resample to 30s using median, then EMA smoothing (≈3min effective)
    resample_rule: str = "30s"
    ema_span: int = 6                 # EMA span (samples at 30s) ~3 minutes
    pre_med_win: str = "1min"         # rolling median before EMA
    min_valid_frac: float = 0.5       # drop station if too sparse
    ffill_limit: int = 3              # fill gaps up to 3 samples (90s)

@dataclass
class BackgroundCfg:
    roll_minutes: int = 90
    lowpass_minutes: int = 15

@dataclass
class DetectCfg:
    # intensity & persistence
    z_trigger: float = 5.0
    z_reset: float = 3.0
    slope_q: float = 0.90              # per-station slopeZ quantile
    mad_win_min: int = 60              # minutes used to estimate robust σ
    min_duration_min: int = 2          # minutes of persistence required
    # ratio constraints
    z4_trigger: float = 5.0
    pmratio_min: float = 1.05
    pmratio_z_min: float = 0.5
    # spatial (kept simple here)
    neighbor_radius_km: float = 1.0
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

# -----------------------------
# Utilities
# -----------------------------

def _to_30s_median_ffill(s: pd.Series, ffill_limit: int) -> pd.Series:
    """Resample to 30s median, remove negatives, fill short gaps."""
    s = s.sort_index().resample("30s").median()
    s[s < 0] = np.nan
    return s.ffill(limit=ffill_limit)

def _smooth_online_ema(x: pd.Series, ema_span=6, pre_med_win="1min") -> pd.Series:
    """Median(1min) + EMA(span samples) smoothing."""
    y = x.rolling(pre_med_win, min_periods=1).median()
    return y.ewm(span=ema_span, adjust=False).mean()

def _rolling_mad(x: pd.Series, win: int) -> pd.Series:
    def mad(a: np.ndarray) -> float:
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return np.nan
        return np.median(np.abs(a - np.median(a))) * 1.4826
    # guard against tiny windows
    win = max(win, 3)
    return x.rolling(win, min_periods=max(3, win // 5)).apply(mad, raw=False)

def _clip_by_quantile(s: pd.Series, qlo: float, qhi: float) -> pd.Series:
    lo, hi = s.quantile([qlo, qhi])
    return s.clip(lo, hi)

def _infer_dt_seconds(idx: pd.DatetimeIndex) -> float:
    """Infer cadence (seconds) from index (median diff)."""
    if len(idx) < 2:
        return 60.0
    diffs = np.diff(idx.view("i8")) / 1e9  # ns -> s
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
# Core pipeline
# -----------------------------

class FireDetector:
    def __init__(self,
                 col: ColumnMap = ColumnMap(),
                 pre: PreprocessCfg = PreprocessCfg(),
                 bg: BackgroundCfg = BackgroundCfg(),
                 det: DetectCfg = DetectCfg(),
                 veto: VetoCfg = VetoCfg(),
                 weights: ScoreWeights = ScoreWeights(),
                 debug: bool = False):
        self.col = col
        self.pre = pre
        self.bg = bg
        self.det = det
        self.veto = veto
        self.weights = weights
        self.debug = debug

    # -------- B. Preprocessing --------
    def load_and_prep(self,
                      pm1_df: pd.DataFrame,
                      pm4_df: pd.DataFrame,
                      env_df: Optional[pd.DataFrame] = None,
                      stations_df: Optional[pd.DataFrame] = None
                      ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        c = self.col
        # parse times
        pm1_df[c.time] = pd.to_datetime(pm1_df[c.time], errors="coerce", utc=True)
        pm4_df[c.time] = pd.to_datetime(pm4_df[c.time], errors="coerce", utc=True)
        pm1_df = pm1_df.dropna(subset=[c.time]).set_index(c.time).sort_index()
        pm4_df = pm4_df.dropna(subset=[c.time]).set_index(c.time).sort_index()

        if env_df is not None:
            env_df[c.time] = pd.to_datetime(env_df[c.time], errors="coerce", utc=True)
            env_df = env_df.dropna(subset=[c.time]).set_index(c.time).sort_index()

        # station columns
        pm1_stations = list(pm1_df.columns)
        pm4_stations = list(pm4_df.columns)

        # 30s median + EMA preprocessing
        pm1_proc = {}
        pm4_proc = {}
        for st in pm1_stations:
            s = _to_30s_median_ffill(pm1_df[st], self.pre.ffill_limit)
            s = _smooth_online_ema(s, self.pre.ema_span, self.pre.pre_med_win)
            pm1_proc[st] = s
        for st in pm4_stations:
            s = _to_30s_median_ffill(pm4_df[st], self.pre.ffill_limit)
            s = _smooth_online_ema(s, self.pre.ema_span, self.pre.pre_med_win)
            pm4_proc[st] = s

        pm1_df = pd.concat(pm1_proc, axis=1) if pm1_proc else pd.DataFrame()
        pm4_df = pd.concat(pm4_proc, axis=1) if pm4_proc else pd.DataFrame()

        # align by time + common stations
        common_stations = sorted(set(pm1_df.columns).intersection(pm4_df.columns))
        pm1_df = pm1_df[common_stations]
        pm4_df = pm4_df[common_stations]
        common_idx = pm1_df.index.intersection(pm4_df.index)
        pm1_df = pm1_df.loc[common_idx]
        pm4_df = pm4_df.loc[common_idx]

        # drop sparse stations (based on PM1)
        if not pm1_df.empty:
            valid_frac = pm1_df.notna().mean()
            keep = valid_frac[valid_frac >= self.pre.min_valid_frac].index
            pm1_df = pm1_df[keep]
            pm4_df = pm4_df[keep]

        if self.debug:
            print(f"PM1 shape: {pm1_df.shape}, PM4 shape: {pm4_df.shape}")

        # build MultiIndex
        pm_wide = pd.concat({"PM1": pm1_df, "PM4": pm4_df}, axis=1)

        # ENV wide (optional, just pass-through; RH used for veto if provided)
        env_wide = None
        if env_df is not None and len(env_df.columns) > 0:
            env_wide = env_df.copy()

        # stations info (optional)
        stations = None
        if stations_df is not None and {"StationID", "Location"}.issubset(stations_df.columns):
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

        return pm_wide, env_wide, stations

    # -------- C. Background estimation --------
    def estimate_background(self, pm_wide: pd.DataFrame) -> pd.DataFrame:
        # station-median then rolling median over time (minutes)
        dt_sec = _infer_dt_seconds(pm_wide.index)
        k_lp = max(1, int(round(self.bg.lowpass_minutes * 60 / dt_sec)))
        roll = f"{self.bg.roll_minutes}min"
        BG = {}
        for metric in ["PM1", "PM4"]:
            W = pm_wide[metric]
            bg = W.median(axis=1, skipna=True).rolling(roll, min_periods=5).median()
            # extra smoothing (moving average with k_lp samples)
            bg = bg.rolling(k_lp, min_periods=1, center=True).mean()
            BG[metric] = bg
        return pd.DataFrame(BG, index=pm_wide.index)

    # -------- D. Local anomaly features --------
    def compute_features(self, pm_wide: pd.DataFrame, bg_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        dt_sec = _infer_dt_seconds(pm_wide.index)
        dt_min = dt_sec / 60.0
        feats: Dict[str, pd.DataFrame] = {}
        resid = {}; Z = {}; slope = {}; slopeZ = {}

        # rolling-MAD window in samples
        mad_win_samples = max(3, int(round(self.det.mad_win_min * 60 / dt_sec)))

        for metric in ["PM1", "PM4"]:
            W = pm_wide[metric]
            r = (W.T - bg_df[metric].values).T  # residuals
            resid[metric] = r

            mad_r = r.apply(lambda s: _rolling_mad(s, mad_win_samples), axis=0)
            Z[metric] = r / (mad_r + 1e-6)

            dW = W.diff().fillna(0) / max(dt_min, 1e-9)  # per minute
            slope[metric] = dW
            mad_slope = dW.apply(lambda s: _rolling_mad(s, mad_win_samples), axis=0)
            slopeZ[metric] = dW / (mad_slope + 1e-6)

        feats["resid"]  = pd.concat(resid, axis=1)
        feats["Z"]      = pd.concat(Z, axis=1)
        feats["slope"]  = pd.concat(slope, axis=1)
        feats["slopeZ"] = pd.concat(slopeZ, axis=1)

        pmratio = pm_wide["PM1"] / pm_wide["PM4"].replace(0, np.nan)
        feats["pmratio"] = pmratio
        return feats

    # -------- E+F. Detection & scoring --------
    def detect_events(self,
                      pm_wide: pd.DataFrame,
                      feats: Dict[str, pd.DataFrame],
                      env_wide: Optional[pd.DataFrame] = None,
                      stations: Optional[pd.DataFrame] = None
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dt_sec = _infer_dt_seconds(pm_wide.index)
        dt_min = dt_sec / 60.0
        idx = pm_wide.index
        stations_list = list(pm_wide["PM1"].columns)

        Z1 = feats["Z"]["PM1"]
        Z4 = feats["Z"]["PM4"]
        slopeZ1 = feats["slopeZ"]["PM1"]
        pmratio = feats["pmratio"]

        # robust z of ratio per station
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

        # persistence in *samples*
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

        # timeline output
        timeline = pd.DataFrame(index=idx)
        for s in stations_list:
            timeline[(s, "cand")] = cand[s].astype(int)
            timeline[(s, "Z")] = Z1[s]
            timeline[(s, "slopeZ")] = slopeZ1[s]
        timeline.columns = pd.MultiIndex.from_tuples(timeline.columns, names=["station", "var"])

        # group continuous segments where any station is candidate
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

        # RH veto (median across env if RH column exists)
        rh_bg = None
        if env_wide is not None:
            # try to find any RH-like column name
            rh_like = [c for c in env_wide.columns if str(c).strip().upper() in ["RH", "HUMIDITY", "RELHUM"]]
            if rh_like:
                rh_name = rh_like[0]
                rh_bg = env_wide[rh_name].median(axis=1, skipna=True)

        # summarize each event
        rows = []
        for eid, (t0, t1) in enumerate(events, start=1):
            win = slice(t0, t1)
            act_stations = [s for s in stations_list if cand.loc[win, s].any()]

            if len(act_stations) == 0:
                # should be rare, keep a placeholder row
                rows.append({
                    "event_id": eid, "t_start": t0, "t_end": t1,
                    "duration_min": (t1 - t0).total_seconds()/60.0,
                    "stations_involved": "",
                    "Z_peak": np.nan, "slopeZ_peak": np.nan,
                    "pmratio_delta_med": np.nan,
                    "veto_rh": False, "veto_sync": False,
                    "score": 0.0, "level": "Weak"
                })
                continue

            Zp = Z1.loc[win, act_stations].max().max()
            slopep = slopeZ1.loc[win, act_stations].max().max()

            # pmratio delta: window median vs past 180min median before t0
            pmr_window = pmratio.loc[win, act_stations].median()

            start = t0 - pd.Timedelta("180min")
            pmr_hist = pmratio.loc[start:t0, act_stations]
            if not pmr_hist.empty:
                pmr_baseline = pmr_hist.median()
            else:
                pmr_baseline = np.nan
                print(f"Debug: pmr_hist is empty")
            if pd.notna(pmr_baseline).any():
                pmr_delta = (pmr_window - pmr_baseline).median()
            else:
                pmr_delta = np.nan

            # vetoes
            rh_high = False
            if rh_bg is not None:
                rh_win = rh_bg.loc[win]
                if not rh_win.empty and rh_win.median() >= 85:
                    rh_high = True

            # sync veto: if many stations turn on within short span
            onsets = []
            for s in act_stations:
                ts = cand[s].loc[win]
                if ts.any():
                    onsets.append(ts.idxmax())
            sync_veto = False
            #if len(onsets) >= max(3, len(stations_list)//2) and (max(onsets) - min(onsets)).total_seconds()/60.0 <= self.veto.all_station_sync_span_min:
            #    sync_veto = True

            veto_flag = rh_high or sync_veto

            # score and level
            score = (
                self.weights.w_z * np.tanh((Zp or 0)/10.0) +
                self.weights.w_slope * np.tanh((slopep or 0)/10.0)
            )
            if veto_flag:
                score *= self.weights.penalty_veto
            score = float(np.clip(score, 0, 1))

            level = "Level-1"
            if score >= 0.6 and not veto_flag:
                level = "Level-2"

            rows.append({
                "event_id": eid,
                "t_start": t0, "t_end": t1,
                "duration_min": (t1 - t0).total_seconds()/60.0,
                "stations_involved": ",".join(act_stations),
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

    # -------- G. Optional: Rough localization (unchanged logic) --------
    def locate_source(self,
                      events_df: pd.DataFrame,
                      feats: Dict[str, pd.DataFrame],
                      stations: pd.DataFrame) -> pd.DataFrame:
        if events_df.empty or stations is None or stations.empty:
            return events_df

        idx = feats["resid"]["PM1"].index
        results = []
        for _, row in events_df.iterrows():
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

            # mins since t0
            t0min = {s: (t - t0).total_seconds()/60.0 for s, t in arrivals.items()}

            # coords -> local km
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
        return pd.DataFrame(results)

# -----------------------------
# Convenience runner
# -----------------------------

from pathlib import Path

def run_pipeline(pm1_csv: str,
                 pm4_csv: str,
                 temp_csv: Optional[str] = None,
                 rh_csv: Optional[str] = None,
                 env_csv: Optional[str] = None,  # Keep for backward compatibility
                 stations_csv: Optional[str] = None,
                 column_map: Optional[ColumnMap] = None,
                 debug: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    col = column_map or ColumnMap()
    det = FireDetector(col=col, debug=debug)

    pm1_df = pd.read_csv(pm1_csv, comment="#")
    pm4_df = pd.read_csv(pm4_csv, comment="#")
    
    # Load environment data - prioritize separate T and RH files over combined ENV file
    env_df = None
    if temp_csv is not None and rh_csv is not None:
        # Load separate temperature and humidity files
        temp_df = pd.read_csv(temp_csv, comment="#")
        rh_df = pd.read_csv(rh_csv, comment="#")
        
        # Combine them into a single env_df with proper column names
        # Temperature columns will have '_T' suffix, RH columns will have '_RH' suffix
        temp_renamed = temp_df.copy()
        rh_renamed = rh_df.copy()
        
        # Rename columns (excluding timestamp_utc)
        for col in temp_renamed.columns:
            if col != 'timestamp_utc':
                temp_renamed = temp_renamed.rename(columns={col: f"{col}_T"})
        
        for col in rh_renamed.columns:
            if col != 'timestamp_utc':
                rh_renamed = rh_renamed.rename(columns={col: f"{col}_RH"})
        
        # Merge on timestamp
        env_df = pd.merge(temp_renamed, rh_renamed, on='timestamp_utc', how='outer')
        
        if debug:
            print(f"Combined env data shape: {env_df.shape}")
            print(f"Temperature stations: {[col for col in env_df.columns if col.endswith('_T')]}")
            print(f"Humidity stations: {[col for col in env_df.columns if col.endswith('_RH')]}")
            
    elif env_csv is not None:
        # Fallback to original ENV.csv approach
        env_df = pd.read_csv(env_csv, comment="#")
    
    stations_df = pd.read_csv(stations_csv) if stations_csv else None

    pm_wide, env_wide, stations = det.load_and_prep(pm1_df, pm4_df, env_df, stations_df)
    BG = det.estimate_background(pm_wide)
    feats = det.compute_features(pm_wide, BG)
    events, timeline = det.detect_events(pm_wide, feats, env_wide, stations)
    loc_df = None
    if stations is not None and not stations.empty and not events.empty:
        loc_df = det.locate_source(events, feats, stations)

    Path("multi_output").mkdir(parents=True, exist_ok=True)
    events.to_csv("multi_output/events_out.csv", index=False)
    timeline.to_csv("multi_output/timeline_out.csv")
    if loc_df is not None:
        loc_df.to_csv("multi_output/events_located.csv", index=False)
    return events, timeline, loc_df

# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    try:
        events, timeline, loc = run_pipeline(
            pm1_csv="data/PM1.csv",
            pm4_csv="data/PM4.csv",
            temp_csv="data/T.csv",             # Temperature data
            rh_csv="data/RH.csv",              # Relative humidity data
            stations_csv="data/Location.csv",  # optional
            debug=False
        )
        print("Saved:")
        print(" - multi_output/events_out.csv")
        print(" - multi_output/timeline_out.csv")
        if loc is not None:
            print(" - multi_output/events_located.csv")
    except Exception as e:
        print("Run failed:", e)
