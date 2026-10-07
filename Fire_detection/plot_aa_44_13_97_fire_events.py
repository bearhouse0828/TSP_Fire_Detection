#!/usr/bin/env python3
"""
Plot fire events for aa-44-13-97 station with all methods and real fire labels
Created for Dae-Cheong Lake fire events visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import os

def get_real_fire_labels():
    """Define real fire labels for Dae-Cheong Lake (KST times)"""
    # Real fire events in KST (Korea Standard Time)
    # Note: KST is UTC+9, so we need to convert to UTC for comparison
    real_fires = [
        {
            'start': '2025-09-11 19:00:00',  # KST
            'end': '2025-09-11 19:25:00',    # KST
            'description': 'Sep 11, 19:00-19:25 KST (Dae-Cheong Lake)'
        },
        {
            'start': '2025-09-12 08:15:00',  # KST
            'end': '2025-09-12 08:45:00',    # KST
            'description': 'Sep 12, 08:15-08:45 KST (Dae-Cheong Lake)'
        },
        {
            'start': '2025-09-16 19:10:00',  # KST
            'end': '2025-09-16 20:00:00',    # KST
            'description': 'Sep 16, 19:10-20:00 KST (Dae-Cheong Lake)'
        },
        {
            'start': '2025-09-17 09:20:00',  # KST
            'end': '2025-09-17 10:25:00',    # KST
            'description': 'Sep 17, 09:20-10:25 KST (Dae-Cheong Lake)'
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
            'description': fire['description'],
            'start_kst': start_kst,
            'end_kst': end_kst
        })
    
    return utc_fires

def load_tspulse_data(station_id='aa-44-13-97'):
    """Load TSPulse data for the specified station"""
    try:
        df = pd.read_csv("tsp_output/tspulse_per_minute.csv", index_col=0, parse_dates=True)
        
        # Get columns for the specific station
        score_col = f"{station_id}_tspulse_score"
        flag_col = f"{station_id}_tspulse_flag"
        
        if score_col in df.columns and flag_col in df.columns:
            return df[[score_col, flag_col]]
        else:
            print(f"⚠️  TSPulse data not found for station {station_id}")
            return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading TSPulse data: {e}")
        return pd.DataFrame()

def load_single_station_data(station_id='aa-44-13-97'):
    """Load single-station fire events data"""
    try:
        fire_file = f"outputs/{station_id}/{station_id}_pm_fire_events.csv"
        if os.path.exists(fire_file):
            df = pd.read_csv(fire_file)
            if not df.empty:
                df['start'] = pd.to_datetime(df['start'])
                df['end'] = pd.to_datetime(df['end'])
                return df
        else:
            print(f"⚠️  Single-station fire events file not found: {fire_file}")
            return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading single-station data: {e}")
        return pd.DataFrame()

def load_multi_station_data(station_id='aa-44-13-97'):
    """Load multi-station fire events data"""
    try:
        df = pd.read_csv("multi_output/events_out.csv")
        if not df.empty:
            df['t_start'] = pd.to_datetime(df['t_start'])
            df['t_end'] = pd.to_datetime(df['t_end'])
            # Filter for events involving this station
            station_events = df[df['stations_involved'] == station_id].copy()
            return station_events
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading multi-station data: {e}")
        return pd.DataFrame()

def create_fire_flag_series(time_index, events_df, start_col='start', end_col='end'):
    """Create a fire flag time series from events dataframe"""
    fire_flag = pd.Series(0, index=time_index)
    
    if not events_df.empty:
        for idx, event in events_df.iterrows():
            mask = (time_index >= event[start_col]) & (time_index <= event[end_col])
            fire_flag[mask] = 1
    
    return fire_flag

def plot_aa_44_13_97_fire_events():
    """Plot comprehensive fire events for aa-44-13-97 station"""
    print("🔥 Plotting aa-44-13-97 Fire Events with All Methods and Real Labels")
    print("=" * 70)
    
    station_id = 'aa-44-13-97'
    
    # Create output directory
    os.makedirs("plots/aa-44-13-97", exist_ok=True)
    
    # Load data from all methods
    print("📊 Loading data from all methods...")
    tspulse_data = load_tspulse_data(station_id)
    single_station_data = load_single_station_data(station_id)
    multi_station_data = load_multi_station_data(station_id)
    real_fires = get_real_fire_labels()
    
    # Determine time range (focus on September 2025)
    start_date = pd.to_datetime('2025-09-01 00:00:00+00:00')
    end_date = pd.to_datetime('2025-09-30 23:59:59+00:00')
    
    # Ensure timezone awareness
    if start_date.tz is None:
        start_date = start_date.tz_localize('UTC')
    if end_date.tz is None:
        end_date = end_date.tz_localize('UTC')
    
    print(f"📅 Time range: {start_date} to {end_date}")
    print(f"📍 Real fires: {len(real_fires)} events")
    print(f"🔍 Single-station events: {len(single_station_data)} events")
    print(f"🔍 Multi-station events: {len(multi_station_data)} events")
    
    # Create time series for the entire period (every 5 minutes)
    time_series = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # Create fire flag time series for each method
    single_fire_flag = create_fire_flag_series(time_series, single_station_data)
    multi_fire_flag = create_fire_flag_series(time_series, multi_station_data, 't_start', 't_end')
    
    # Create real fire flag time series
    real_fire_flag = pd.Series(0, index=time_series)
    for fire in real_fires:
        mask = (time_series >= fire['start']) & (time_series <= fire['end'])
        real_fire_flag[mask] = 1
    
    # Create the main comparison plot (3 subplots instead of 4)
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    
    # Plot 1: Single-station Detection
    ax1 = axes[0]
    ax1.fill_between(time_series, 0, single_fire_flag, color="blue", alpha=0.6, label="Single-station Events")
    ax1.set_title("Single-station Fire Detection - aa-44-13-97", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Single-station\nFire Flag", fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(start_date, end_date)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    
    # Plot 2: Multi-station Detection
    ax2 = axes[1]
    ax2.fill_between(time_series, 0, multi_fire_flag, color="green", alpha=0.6, label="Multi-station Events")
    ax2.set_title("Multi-station Fire Detection - aa-44-13-97", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Multi-station\nFire Flag", fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(start_date, end_date)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    
    # Plot 3: TSPulse Detection
    ax3 = axes[2]
    if not tspulse_data.empty:
        # Filter TSPulse data to our time range
        tspulse_filtered = tspulse_data[(tspulse_data.index >= start_date) & 
                                       (tspulse_data.index <= end_date)]
        
        if not tspulse_filtered.empty:
            flag_col = f"{station_id}_tspulse_flag"
            ax3.fill_between(tspulse_filtered.index, 0, tspulse_filtered[flag_col], 
                           color="red", alpha=0.6, label="TSPulse Events")
    else:
        ax3.text(0.5, 0.5, "No TSPulse data available", transform=ax3.transAxes, 
                ha='center', va='center', fontsize=12)
    
    ax3.set_title("TSPulse Fire Detection - aa-44-13-97", fontsize=12, fontweight='bold')
    ax3.set_ylabel("TSPulse\nFire Flag", fontsize=10)
    ax3.set_xlabel("Time (UTC)", fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.set_xlim(start_date, end_date)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=8)
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    
    # Add vertical lines for real fire events across all subplots
    for fire in real_fires:
        for ax in axes:
            ax.axvline(x=fire['start'], color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(x=fire['end'], color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Set overall title and layout
    fig.suptitle("Fire Events Detection Comparison - aa-44-13-97 (Dae-Cheong Lake Area)\n" +
                "All Detection Methods (Real Fire Events Marked with Vertical Lines)", fontsize=14, fontweight='bold', y=0.98)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    output_file = "plots/aa-44-13-97/aa-44-13-97_all_methods_with_real_labels.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comprehensive plot: {output_file}")
    
    # Create a summary statistics plot
    create_summary_statistics_plot(station_id, real_fires, single_station_data, 
                                  multi_station_data, tspulse_data, start_date, end_date)

def create_summary_statistics_plot(station_id, real_fires, single_station_data, 
                                  multi_station_data, tspulse_data, start_date, end_date):
    """Create a summary statistics plot showing detection performance"""
    print("📊 Creating summary statistics plot...")
    
    # Calculate detection statistics
    real_fire_periods = []
    for fire in real_fires:
        real_fire_periods.append({
            'start': fire['start'],
            'end': fire['end'],
            'description': fire['description']
        })
    
    # Check detections within real fire periods (with some tolerance)
    tolerance_minutes = 30  # 30-minute tolerance window
    
    single_detections = []
    multi_detections = []
    tspulse_detections = []
    
    for real_fire in real_fire_periods:
        fire_start = real_fire['start'] - timedelta(minutes=tolerance_minutes)
        fire_end = real_fire['end'] + timedelta(minutes=tolerance_minutes)
        
        # Check single-station detections
        single_in_window = single_station_data[
            (single_station_data['start'] <= fire_end) & 
            (single_station_data['end'] >= fire_start)
        ]
        single_detections.append(len(single_in_window) > 0)
        
        # Check multi-station detections
        multi_in_window = multi_station_data[
            (multi_station_data['t_start'] <= fire_end) & 
            (multi_station_data['t_end'] >= fire_start)
        ]
        multi_detections.append(len(multi_in_window) > 0)
        
        # Check TSPulse detections
        if not tspulse_data.empty:
            tspulse_in_window = tspulse_data[
                (tspulse_data.index >= fire_start) & 
                (tspulse_data.index <= fire_end)
            ]
            flag_col = f"{station_id}_tspulse_flag"
            if flag_col in tspulse_in_window.columns:
                tspulse_detections.append((tspulse_in_window[flag_col] > 0).any())
            else:
                tspulse_detections.append(False)
        else:
            tspulse_detections.append(False)
    
    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Detection Success Rate
    methods = ['Single-station', 'Multi-station', 'TSPulse']
    success_rates = [
        sum(single_detections) / len(single_detections) * 100,
        sum(multi_detections) / len(multi_detections) * 100,
        sum(tspulse_detections) / len(tspulse_detections) * 100
    ]
    
    colors = ['blue', 'green', 'red']
    bars = ax1.bar(methods, success_rates, color=colors, alpha=0.7)
    ax1.set_title("Detection Success Rate\n(Within 30-min tolerance of real fires)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Success Rate (%)", fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Detection Timeline
    ax2.set_title("Detection Timeline for Real Fire Events", fontsize=12, fontweight='bold')
    
    # Plot real fire periods
    for i, fire in enumerate(real_fire_periods):
        ax2.barh(i, (fire['end'] - fire['start']).total_seconds() / 3600, 
                left=(fire['start'] - start_date).total_seconds() / 3600,
                height=0.6, color='black', alpha=0.7, label='Real Fire' if i == 0 else "")
        
        # Plot detections
        if single_detections[i]:
            ax2.barh(i-0.2, 0.1, left=(fire['start'] - start_date).total_seconds() / 3600,
                    height=0.15, color='blue', alpha=0.8, label='Single-station' if i == 0 else "")
        
        if multi_detections[i]:
            ax2.barh(i, 0.1, left=(fire['start'] - start_date).total_seconds() / 3600,
                    height=0.15, color='green', alpha=0.8, label='Multi-station' if i == 0 else "")
        
        if tspulse_detections[i]:
            ax2.barh(i+0.2, 0.1, left=(fire['start'] - start_date).total_seconds() / 3600,
                    height=0.15, color='red', alpha=0.8, label='TSPulse' if i == 0 else "")
    
    ax2.set_ylabel("Fire Event", fontsize=10)
    ax2.set_xlabel("Hours from Sep 1, 2025", fontsize=10)
    ax2.set_yticks(range(len(real_fire_periods)))
    ax2.set_yticklabels([f"Event {i+1}" for i in range(len(real_fire_periods))])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save summary plot
    output_file = "plots/aa-44-13-97/aa-44-13-97_detection_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved summary plot: {output_file}")
    
    # Print summary statistics
    print("\n📊 Detection Summary for aa-44-13-97:")
    print(f"   Real Fire Events: {len(real_fire_periods)}")
    print(f"   Single-station Success Rate: {success_rates[0]:.1f}% ({sum(single_detections)}/{len(single_detections)})")
    print(f"   Multi-station Success Rate: {success_rates[1]:.1f}% ({sum(multi_detections)}/{len(multi_detections)})")
    print(f"   TSPulse Success Rate: {success_rates[2]:.1f}% ({sum(tspulse_detections)}/{len(tspulse_detections)})")

def main():
    """Main function"""
    try:
        plot_aa_44_13_97_fire_events()
        print("\n🎉 All plots completed successfully!")
        print("📁 Check the 'plots/aa-44-13-97' directory for generated visualizations:")
        print("   - aa-44-13-97_all_methods_with_real_labels.png : Comprehensive comparison")
        print("   - aa-44-13-97_detection_summary.png : Detection performance summary")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
