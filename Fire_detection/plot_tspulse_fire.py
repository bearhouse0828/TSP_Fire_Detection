# plot_tspulse_timeline.py
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==== 1. Read file ====
df = pd.read_csv("tsp_output/tspulse_per_minute.csv", index_col=0, parse_dates=True)

# ==== 2. Detect stations ====
stations = sorted(set(c.split("_")[0] for c in df.columns if "_tspulse_score" in c))

# ==== 3. Create output folder ====
os.makedirs("plots", exist_ok=True)

# ==== 4. Plot for each station ====
for st in stations:
    sc_col = f"{st}_tspulse_score"
    fl_col = f"{st}_tspulse_flag"
    if sc_col not in df.columns or fl_col not in df.columns:
        continue

    fig, ax = plt.subplots(figsize=(10, 3))
    #ax.plot(df.index, df[sc_col], label="TSPulse Score", color="blue", lw=1.5)
    ax.fill_between(df.index, 0, df[fl_col]*1.0, color="red", alpha=0.3, label="Fire Flag")

    ax.set_title(f"TSPulse Fire Detection Timeline – {st}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"plots/{st}_tspulse_timeline.png", dpi=150)
    plt.close()

print(f"✅ Saved plots for {len(stations)} stations in ./plots/")
