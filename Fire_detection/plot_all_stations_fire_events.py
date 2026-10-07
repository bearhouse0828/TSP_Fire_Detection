# plot_all_stations_fire_events.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from datetime import datetime, timedelta

def get_global_time_range():
    """Get global time range from PM1 data"""
    try:
        df = pd.read_csv('data/PM1.csv', skiprows=1)
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        time_min = df['timestamp_utc'].min()
        time_max = df['timestamp_utc'].max()
        print(f"📅 Global time range: {time_min} to {time_max}")
        return time_min, time_max
    except Exception as e:
        print(f"⚠️  Could not read PM1 data: {e}")
        return None, None

def get_active_stations():
    """Get active stations from Location.csv"""
    try:
        df = pd.read_csv('data/Location.csv')
        active_stations = df[df['Status'] == 'Active']['StationID'].tolist()
        print(f"📍 Active stations: {active_stations}")
        return active_stations
    except Exception as e:
        print(f"⚠️  Could not read Location.csv: {e}")
        return []

def get_real_fire_events():
    """Get real fire events for marking on plots (KST times)"""
    real_fires = [
        {
            'start': '2025-09-11 19:00:00',  # KST
            'end': '2025-09-11 19:25:00',    # KST
            'location': 'Dae-Cheong Lake',
            'description': 'Sep 11, 19:00-19:25 KST (Dae-Cheong Lake)'
        },
        {
            'start': '2025-09-12 08:15:00',  # KST
            'end': '2025-09-12 08:45:00',    # KST
            'location': 'Dae-Cheong Lake',
            'description': 'Sep 12, 08:15-08:45 KST (Dae-Cheong Lake)'
        },
        {
            'start': '2025-09-16 19:10:00',  # KST
            'end': '2025-09-16 20:00:00',    # KST
            'location': 'Dae-Cheong Lake',
            'description': 'Sep 16, 19:10-20:00 KST (Dae-Cheong Lake)'
        },
        {
            'start': '2025-09-17 09:20:00',  # KST
            'end': '2025-09-17 10:25:00',    # KST
            'location': 'Dae-Cheong Lake',
            'description': 'Sep 17, 09:20-10:25 KST (Dae-Cheong Lake)'
        },
        {
            'start': '2025-09-11 19:40:00',  # KST
            'end': '2025-09-11 20:00:00',    # KST
            'location': 'Cherry Blossom Road',
            'description': 'Sep 11, 19:40-20:00 KST (Cherry Blossom Road)'
        },
        {
            'start': '2025-09-16 18:00:00',  # KST
            'end': '2025-09-16 18:30:00',    # KST
            'location': 'Cherry Blossom Road',
            'description': 'Sep 16, 18:00-18:30 KST (Cherry Blossom Road)'
        }
    ]
    
    # Convert KST to UTC (subtract 9 hours)
    utc_fires = []
    for fire in real_fires:
        start_kst = pd.to_datetime(fire['start'])
        end_kst = pd.to_datetime(fire['end'])
        
        start_utc = start_kst - timedelta(hours=9)
        end_utc = end_kst - timedelta(hours=9)
        
        # Ensure UTC timezone awareness
        start_utc = start_utc.tz_localize('UTC') if start_utc.tz is None else start_utc
        end_utc = end_utc.tz_localize('UTC') if end_utc.tz is None else end_utc
        
        utc_fires.append({
            'start': start_utc,
            'end': end_utc,
            'location': fire['location'],
            'description': fire['description']
        })
    
    return utc_fires

def add_real_fire_markers(axes, real_fires):
    """Add vertical lines for real fire events across all subplots"""
    for fire in real_fires:
        for ax in axes:
            # Add start and end lines
            ax.axvline(x=fire['start'], color='black', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.axvline(x=fire['end'], color='black', linestyle='--', alpha=0.7, linewidth=1.5)
            
            # Add fill between start and end for better visibility
            ax.axvspan(fire['start'], fire['end'], alpha=0.1, color='black')

def plot_tspulse_all_stations(active_stations, global_time_min, global_time_max, real_fires):
    """Plot TSPulse fire events for all active stations using fill_between (same as single station plots)"""
    print("📊 Plotting TSPulse events for all active stations...")
    
    # Create output folder
    os.makedirs("plots/all_stations", exist_ok=True)
    
    # Read TSPulse per minute data (same as single station plots)
    try:
        df = pd.read_csv("tsp_output/tspulse_per_minute.csv", index_col=0, parse_dates=True)
    except Exception as e:
        print(f"❌ Error reading TSPulse per minute data: {e}")
        return
    
    # Filter for active stations only
    available_stations = [st for st in active_stations if f"{st}_tspulse_flag" in df.columns]
    
    if not available_stations:
        print("⚠️  No TSPulse data found for active stations")
        return
    
    # Create figure with subplots for each station
    n_stations = len(available_stations)
    fig, axes = plt.subplots(n_stations, 1, figsize=(15, 2*n_stations), sharex=True)
    
    if n_stations == 1:
        axes = [axes]
    
    for i, station in enumerate(available_stations):
        ax = axes[i]
        
        # Get the flag column for this station
        fl_col = f"{station}_tspulse_flag"
        
        # Plot using fill_between (same as single station plots)
        ax.fill_between(df.index, 0, df[fl_col]*1.0, color="red", alpha=0.3, label="Fire Flag")
        
        # Set station-specific formatting
        ax.set_title(f"TSPulse Fire Events - {station}", fontsize=10, pad=10)
        ax.set_ylabel("Fire Flag", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlim(global_time_min, global_time_max)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Add real fire event markers
    add_real_fire_markers(axes, real_fires)
    
    # Set common x-axis label and formatting
    fig.suptitle("TSPulse Fire Events - All Active Stations (Real Fire Events Marked)", fontsize=14, y=0.98)
    axes[-1].set_xlabel("Time", fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig("plots/all_stations/tspulse_all_active_stations.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved TSPulse plot for {n_stations} active stations")

def plot_outputs_all_stations(active_stations, global_time_min, global_time_max, real_fires):
    """Plot outputs fire events for all active stations using fill_between"""
    print("📊 Plotting outputs events for all active stations...")
    
    # Create output folder
    os.makedirs("plots/all_stations", exist_ok=True)
    
    # Collect all fire events data
    all_events = []
    
    for station in active_stations:
        fire_files = glob.glob(f"outputs/{station}/{station}_pm_fire_events.csv")
        fire_files.extend(glob.glob(f"outputs/{station}_pm_fire_events.csv"))
        
        for fire_file in fire_files:
            try:
                events = pd.read_csv(fire_file)
                if not events.empty:
                    events['station'] = station
                    events['start'] = pd.to_datetime(events['start'])
                    events['end'] = pd.to_datetime(events['end'])
                    all_events.append(events)
            except Exception as e:
                print(f"⚠️  Error reading {fire_file}: {e}")
    
    if not all_events:
        print("⚠️  No outputs events found for active stations")
        return
    
    # Combine all events
    combined_events = pd.concat(all_events, ignore_index=True)
    
    # Create figure with subplots for each station
    n_stations = len(active_stations)
    fig, axes = plt.subplots(n_stations, 1, figsize=(15, 2*n_stations), sharex=True)
    
    if n_stations == 1:
        axes = [axes]
    
    # Create time series for the entire period (every 5 minutes)
    time_series = pd.date_range(start=global_time_min, end=global_time_max, freq='5min')
    
    for i, station in enumerate(active_stations):
        ax = axes[i]
        
        # Get events for this station
        station_events = combined_events[combined_events['station'] == station]
        
        # Create fire flag time series for this station
        fire_flag = pd.Series(0, index=time_series)
        
        if not station_events.empty:
            # Mark fire events in the time series
            for idx, event in station_events.iterrows():
                mask = (time_series >= event['start']) & (time_series <= event['end'])
                fire_flag[mask] = 1
        
        # Plot using fill_between
        ax.fill_between(time_series, 0, fire_flag, color="red", alpha=0.3, label="Single-station Events")
        
        # Set station-specific formatting
        ax.set_title(f"Single-station Fire Events - {station}", fontsize=10, pad=10)
        ax.set_ylabel("Fire Flag", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlim(global_time_min, global_time_max)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Add real fire event markers
    add_real_fire_markers(axes, real_fires)
    
    # Set common x-axis label and formatting
    fig.suptitle("Single-station Fire Events - All Active Stations (Real Fire Events Marked)", fontsize=14, y=0.98)
    axes[-1].set_xlabel("Time", fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig("plots/all_stations/outputs_all_active_stations.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved outputs plot for {n_stations} active stations")

def plot_combined_comparison(active_stations, global_time_min, global_time_max, real_fires):
    """Plot combined comparison of TSPulse vs Outputs for all stations using fill_between"""
    print("📊 Creating combined comparison plot...")
    
    # Create output folder
    os.makedirs("plots/all_stations", exist_ok=True)
    
    # Read TSPulse per minute data (same as single station plots)
    try:
        tsp_df = pd.read_csv("tsp_output/tspulse_per_minute.csv", index_col=0, parse_dates=True)
    except:
        tsp_df = pd.DataFrame()
    
    # Collect outputs events
    all_outputs_events = []
    for station in active_stations:
        fire_files = glob.glob(f"outputs/{station}/{station}_pm_fire_events.csv")
        fire_files.extend(glob.glob(f"outputs/{station}_pm_fire_events.csv"))
        
        for fire_file in fire_files:
            try:
                events = pd.read_csv(fire_file)
                if not events.empty:
                    events['station'] = station
                    events['start'] = pd.to_datetime(events['start'])
                    events['end'] = pd.to_datetime(events['end'])
                    all_outputs_events.append(events)
            except:
                continue
    
    outputs_combined = pd.concat(all_outputs_events, ignore_index=True) if all_outputs_events else pd.DataFrame()
    
    # Create figure with subplots for each station
    n_stations = len(active_stations)
    fig, axes = plt.subplots(n_stations, 1, figsize=(16, 2*n_stations), sharex=True)
    
    if n_stations == 1:
        axes = [axes]
    
    # Create time series for the entire period (every 5 minutes)
    time_series = pd.date_range(start=global_time_min, end=global_time_max, freq='5min')
    
    for i, station in enumerate(active_stations):
        ax = axes[i]
        
        # Create fire flag time series for Outputs (Single-station)
        outputs_fire_flag = pd.Series(0, index=time_series)
        outputs_station = outputs_combined[outputs_combined['station'] == station] if not outputs_combined.empty else pd.DataFrame()
        if not outputs_station.empty:
            for idx, event in outputs_station.iterrows():
                mask = (time_series >= event['start']) & (time_series <= event['end'])
                outputs_fire_flag[mask] = 1

        # Plot Outputs events (top half)
        ax.fill_between(time_series, 0.5, 0.5 + outputs_fire_flag*0.5, color="blue", alpha=0.4, label="Single-station Events")

        # Plot TSPulse events (bottom half) using per-minute data
        fl_col = f"{station}_tspulse_flag"
        if not tsp_df.empty and fl_col in tsp_df.columns:
            ax.fill_between(tsp_df.index, 0, tsp_df[fl_col]*0.5, color="red", alpha=0.4, label="TSPulse Events")
        
        # Set station-specific formatting
        ax.set_title(f"Fire Events Comparison - {station}", fontsize=10, pad=10)
        ax.set_ylabel("Fire Flag", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlim(global_time_min, global_time_max)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Add horizontal line to separate Single-station and TSPulse
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add legend for this subplot
        ax.text(0.02, 0.85, "Single-station", transform=ax.transAxes, fontsize=8, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.4))
        ax.text(0.02, 0.15, "TSPulse", transform=ax.transAxes, fontsize=8,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.4))
    
    # Add real fire event markers
    add_real_fire_markers(axes, real_fires)
    
    # Set common x-axis label and formatting
    fig.suptitle("Fire Events Comparison - Single-station vs TSPulse (All Active Stations) (Real Fire Events Marked)", fontsize=14, y=0.98)
    axes[-1].set_xlabel("Time", fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig("plots/all_stations/combined_comparison_all_stations.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved combined comparison plot for {n_stations} active stations")

def plot_multi_output_all_stations(active_stations, global_time_min, global_time_max, real_fires):
    """Plot multi_output fire events for all active stations using fill_between"""
    print("📊 Plotting multi_output events for all active stations...")
    
    # Create output folder
    os.makedirs("plots/all_stations", exist_ok=True)
    
    # Read multi_output events
    try:
        multi_events = pd.read_csv("multi_output/events_out.csv")
        multi_events['t_start'] = pd.to_datetime(multi_events['t_start'])
        multi_events['t_end'] = pd.to_datetime(multi_events['t_end'])
    except Exception as e:
        print(f"❌ Error reading multi_output events: {e}")
        return
    
    # Filter events for active stations only
    active_events = multi_events[multi_events['stations_involved'].isin(active_stations)].copy()
    
    if active_events.empty:
        print("⚠️  No multi_output events found for active stations")
        return
    
    # Create figure with subplots for each station
    n_stations = len(active_stations)
    fig, axes = plt.subplots(n_stations, 1, figsize=(15, 2*n_stations), sharex=True)
    
    if n_stations == 1:
        axes = [axes]
    
    # Create time series for the entire period (every 5 minutes)
    time_series = pd.date_range(start=global_time_min, end=global_time_max, freq='5min')
    
    for i, station in enumerate(active_stations):
        ax = axes[i]
        
        # Get events for this station
        station_events = active_events[active_events['stations_involved'] == station]
        
        # Create fire flag time series for this station
        fire_flag = pd.Series(0, index=time_series)
        
        if not station_events.empty:
            # Mark fire events in the time series
            for idx, event in station_events.iterrows():
                mask = (time_series >= event['t_start']) & (time_series <= event['t_end'])
                fire_flag[mask] = 1
        
        # Plot using fill_between
        ax.fill_between(time_series, 0, fire_flag, color="red", alpha=0.3, label="Multi-station Events")
        
        # Set station-specific formatting
        ax.set_title(f"Multi-station Fire Events - {station}", fontsize=10, pad=10)
        ax.set_ylabel("Fire Flag", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlim(global_time_min, global_time_max)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Add real fire event markers
    add_real_fire_markers(axes, real_fires)
    
    # Set common x-axis label and formatting
    fig.suptitle("Multi-station Fire Events - All Active Stations (Real Fire Events Marked)", fontsize=14, y=0.98)
    axes[-1].set_xlabel("Time", fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig("plots/all_stations/multi_output_all_active_stations.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved multi_output plot for {n_stations} active stations")

def plot_all_methods_comparison(active_stations, global_time_min, global_time_max, real_fires):
    """Plot comparison of all three methods: TSPulse, Outputs, and Multi Output"""
    print("📊 Creating all methods comparison plot...")
    
    # Create output folder
    os.makedirs("plots/all_stations", exist_ok=True)
    
    # Read TSPulse per minute data
    try:
        tsp_df = pd.read_csv("tsp_output/tspulse_per_minute.csv", index_col=0, parse_dates=True)
    except:
        tsp_df = pd.DataFrame()
    
    # Collect outputs events
    all_outputs_events = []
    for station in active_stations:
        fire_files = glob.glob(f"outputs/{station}/{station}_pm_fire_events.csv")
        fire_files.extend(glob.glob(f"outputs/{station}_pm_fire_events.csv"))
        
        for fire_file in fire_files:
            try:
                events = pd.read_csv(fire_file)
                if not events.empty:
                    events['station'] = station
                    events['start'] = pd.to_datetime(events['start'])
                    events['end'] = pd.to_datetime(events['end'])
                    all_outputs_events.append(events)
            except:
                continue
    
    outputs_combined = pd.concat(all_outputs_events, ignore_index=True) if all_outputs_events else pd.DataFrame()
    
    # Read multi_output events
    try:
        multi_events = pd.read_csv("multi_output/events_out.csv")
        multi_events['t_start'] = pd.to_datetime(multi_events['t_start'])
        multi_events['t_end'] = pd.to_datetime(multi_events['t_end'])
        multi_active = multi_events[multi_events['stations_involved'].isin(active_stations)].copy()
    except:
        multi_active = pd.DataFrame()
    
    # Create figure with subplots for each station
    n_stations = len(active_stations)
    fig, axes = plt.subplots(n_stations, 1, figsize=(16, 2*n_stations), sharex=True)
    
    if n_stations == 1:
        axes = [axes]
    
    # Create time series for the entire period (every 5 minutes)
    time_series = pd.date_range(start=global_time_min, end=global_time_max, freq='5min')
    
    for i, station in enumerate(active_stations):
        ax = axes[i]
        
        # Plot Outputs events (top third)
        outputs_fire_flag = pd.Series(0, index=time_series)
        outputs_station = outputs_combined[outputs_combined['station'] == station] if not outputs_combined.empty else pd.DataFrame()
        if not outputs_station.empty:
            for idx, event in outputs_station.iterrows():
                mask = (time_series >= event['start']) & (time_series <= event['end'])
                outputs_fire_flag[mask] = 1
        ax.fill_between(time_series, 0.67, 0.67 + outputs_fire_flag*0.33, color="blue", alpha=0.4, label="Single-station Events")
        
        # Plot Multi Output events (middle third)
        multi_fire_flag = pd.Series(0, index=time_series)
        multi_station = multi_active[multi_active['stations_involved'] == station] if not multi_active.empty else pd.DataFrame()
        if not multi_station.empty:
            for idx, event in multi_station.iterrows():
                mask = (time_series >= event['t_start']) & (time_series <= event['t_end'])
                multi_fire_flag[mask] = 1
        ax.fill_between(time_series, 0.33, 0.33 + multi_fire_flag*0.34, color="green", alpha=0.4, label="Multi-station Events")
        
        # Plot TSPulse events (bottom third)
        fl_col = f"{station}_tspulse_flag"
        if not tsp_df.empty and fl_col in tsp_df.columns:
            ax.fill_between(tsp_df.index, 0, tsp_df[fl_col]*0.33, color="red", alpha=0.4, label="TSPulse Events")
        
        # Set station-specific formatting
        ax.set_title(f"All Methods Fire Events Comparison - {station}", fontsize=10, pad=10)
        ax.set_ylabel("Fire Flag", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlim(global_time_min, global_time_max)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Add horizontal lines to separate methods
        ax.axhline(y=0.67, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=0.33, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add legend for this subplot
        ax.text(0.02, 0.85, "Single-station", transform=ax.transAxes, fontsize=8, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.4))
        ax.text(0.02, 0.5, "Multi-station", transform=ax.transAxes, fontsize=8,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.4))
        ax.text(0.02, 0.15, "TSPulse", transform=ax.transAxes, fontsize=8,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.4))
    
    # Add real fire event markers
    add_real_fire_markers(axes, real_fires)
    
    # Set common x-axis label and formatting
    fig.suptitle("All Methods Fire Events Comparison - Single-station vs Multi-station vs TSPulse (All Active Stations) (Real Fire Events Marked)", fontsize=12, y=0.98)
    axes[-1].set_xlabel("Time", fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig("plots/all_stations/all_methods_comparison_all_stations.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved all methods comparison plot for {n_stations} active stations")

def main():
    """Main function to create all-stations fire events plots"""
    print("🔥 All Stations Fire Events Plotter")
    print("=" * 50)
    
    # Get global time range and active stations
    global_time_min, global_time_max = get_global_time_range()
    active_stations = get_active_stations()
    real_fires = get_real_fire_events()
    
    if not active_stations:
        print("❌ No active stations found")
        return
    
    if global_time_min is None or global_time_max is None:
        print("❌ Could not determine global time range")
        return
    
    print(f"📍 Real fire events to mark: {len(real_fires)}")
    for fire in real_fires:
        print(f"   - {fire['description']}")
    
    # Create all-stations plots
    plot_tspulse_all_stations(active_stations, global_time_min, global_time_max, real_fires)
    plot_outputs_all_stations(active_stations, global_time_min, global_time_max, real_fires)
    plot_multi_output_all_stations(active_stations, global_time_min, global_time_max, real_fires)
    plot_combined_comparison(active_stations, global_time_min, global_time_max, real_fires)
    plot_all_methods_comparison(active_stations, global_time_min, global_time_max, real_fires)
    
    print("\n🎉 All all-stations plotting completed!")
    print("📁 Check the 'plots/all_stations' directory for generated visualizations:")
    print("   - tspulse_all_active_stations.png        : TSPulse events for all stations")
    print("   - outputs_all_active_stations.png        : Single-station events for all stations")
    print("   - multi_output_all_active_stations.png   : Multi-station events for all stations")
    print("   - combined_comparison_all_stations.png   : TSPulse vs Single-station comparison")
    print("   - all_methods_comparison_all_stations.png : All three methods comparison")

if __name__ == "__main__":
    main()
