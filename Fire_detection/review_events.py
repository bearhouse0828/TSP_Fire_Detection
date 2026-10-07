# Reload and run the review + plotting pipeline for the three uploaded CSVs.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
try:
    from caas_jupyter_tools import display_dataframe_to_user
except Exception:
    def display_dataframe_to_user(title, df):
        # Fallback: save a preview CSV for manual inspection when caas_jupyter_tools is unavailable
        preview_path = Path("/Users/jpeng/Downloads/fire_detection") / "_preview_review_head.csv"
        try:
            df.head(100).to_csv(preview_path, index=False)
        except Exception:
            pass

base = Path("/Users/jpeng/Downloads/fire_detection")
events = pd.read_csv(base / "events_out.csv", parse_dates=["t_start","t_end"])
timeline = pd.read_csv(base / "timeline_out.csv", header=[0,1], index_col=0, parse_dates=True)
loc_df = None
if (base / "events_located.csv").exists():
    loc_df = pd.read_csv(base / "events_located.csv")

def bool_str(x: bool) -> str:
    return "Yes" if bool(x) else "No"

def mk_reason(row) -> str:
    bits = []
    if row["Z_peak"] >= 6: bits.append("strong Z")
    elif row["Z_peak"] >= 5: bits.append("moderate Z")
    else: bits.append("weak Z")
    if row["slopeZ_peak"] >= 6: bits.append("steep rise")
    elif row["slopeZ_peak"] >= 5: bits.append("moderate rise")
    if "pmratio_delta_med" in row and pd.notna(row["pmratio_delta_med"]):
        if row["pmratio_delta_med"] > 0.1: bits.append("PM1/PM4 up")
        elif row["pmratio_delta_med"] < -0.05: bits.append("PM1/PM4 down")
        else: bits.append("PM1/PM4 flat")
    if row["neigh_ok"] >= 0.5: bits.append("neighbors ok")
    if row["lag_ok"] >= 0.5: bits.append("lags ok")
    if row["decay_ok"] >= 0.5: bits.append("decay ok")
    if row["veto_rh"]: bits.append("high RH veto")
    if row["veto_sync"]: bits.append("sync-rise veto")
    return ", ".join(bits)

def recommend(row) -> str:
    veto = bool(row["veto_rh"]) or bool(row["veto_sync"])
    cond_strength = (row["Z_peak"] >= 6) and (row["slopeZ_peak"] >= 5)
    cond_spatial = (row["neigh_ok"] >= 0.5) + (row["lag_ok"] >= 0.5) + (row["decay_ok"] >= 0.5) >= 2
    cond_score = row["score"] >= 0.6
    pmratio_helpful = ("pmratio_delta_med" in row) and pd.notna(row["pmratio_delta_med"]) and (row["pmratio_delta_med"] > 0.05)
    if (not veto) and cond_strength and cond_spatial and cond_score:
        return "Confirm Level-2 (recommended)"
    if (not veto) and (cond_strength or cond_spatial) and (row["score"] >= 0.5 or pmratio_helpful):
        return "Likely Level-1 (keep as potential)"
    return "Background/Reject (not recommended)"

review = events.copy()
review["reason"] = review.apply(mk_reason, axis=1)
review["recommendation"] = review.apply(recommend, axis=1)
review["fire_yes_no"] = review["recommendation"].str.startswith("Confirm").map({True:"Yes", False:"No"})
review = review.sort_values(["level","score","t_start"], ascending=[True, False, True]).reset_index(drop=True)

review_path = base / "events_review.csv"
review.to_csv(review_path, index=False)
display_dataframe_to_user("Event Review (recommendations)", review)

# Quick-look plots for top events
plot_dir = base / "event_plots"
plot_dir.mkdir(exist_ok=True)
topN = len(review)
top_events = review.sort_values("score", ascending=False).head(topN)

generated = []
for _, row in top_events.iterrows():
    eid = int(row["event_id"])
    t0, t1 = row["t_start"], row["t_end"]
    stations = []
    if isinstance(row["stations_involved"], str) and row["stations_involved"].strip():
        stations = [s.strip() for s in row["stations_involved"].split(",")]
    if len(stations) == 0:
        win = (timeline.index>=t0) & (timeline.index<=t1)
        zs = [(col[0], timeline[col][win].max()) for col in timeline.columns if col[1]=="Z"]
        zs_df = pd.DataFrame(zs, columns=["station","Zmax"]).groupby("station")["Zmax"].max().sort_values(ascending=False)
        stations = list(zs_df.head(4).index)

    # Z over time plot
    win = (timeline.index>=t0) & (timeline.index<=t1)
    fig = plt.figure(figsize=(10,4))
    ax = plt.gca()
    for s in stations:
        col = (s,"Z")
        if col in timeline.columns:
            ax.plot(timeline.index[win], timeline[col][win], label=s)
    ax.axvspan(t0, t1, alpha=0.08)
    ax.set_ylabel("Z")
    ax.set_title(f"Event {eid} (Z over time)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    outp = plot_dir / f"event_{eid}_Z.png"
    fig.savefig(outp, dpi=150)
    plt.close(fig)
    generated.append(outp.as_posix())

# Save a per-event one-line text summary
summ_lines = []
for _, r in review.iterrows():
    pmr = "NA" if ("pmratio_delta_med" not in r or pd.isna(r["pmratio_delta_med"])) else f"{r['pmratio_delta_med']:.3f}"
    line = (
        f"Event {int(r['event_id'])}: {r['t_start']} – {r['t_end']} "
        f"({r['duration_min']:.1f} min), score={r['score']:.2f}, level={r['level']}; "
        f"neighbors={r['neigh_ok']:.2f}, lag_ok={r['lag_ok']:.2f}, decay_ok={r['decay_ok']:.2f}; "
        f"pmratioΔ={pmr}; veto_rh={bool_str(r['veto_rh'])}, veto_sync={bool_str(r['veto_sync'])}; "
        f"→ {r['recommendation']} ({r['reason']})"
    )
    summ_lines.append(line)


print("Artifacts:")
print("[CSV] ", review_path.as_posix())
print("[PLOT previews]")
for p in generated:
    print(" -", p)
