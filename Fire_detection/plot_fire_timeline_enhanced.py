# plot_fire_timeline_enhanced.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

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

def plot_tspulse_timeline(global_time_min=None, global_time_max=None):
    """Plot TSPulse timeline data from tsp_output directory"""
    print("📊 Plotting TSPulse timeline data...")
    
    # Read TSPulse per minute data
    df = pd.read_csv("tsp_output/tspulse_per_minute.csv", index_col=0, parse_dates=True)
    
    # Detect stations
    stations = sorted(set(c.split("_")[0] for c in df.columns if "_tspulse_score" in c))
    
    # Create output folder
    os.makedirs("plots/tspulse", exist_ok=True)
    
    # Plot for each station
    for st in stations:
        sc_col = f"{st}_tspulse_score"
        fl_col = f"{st}_tspulse_flag"
        if sc_col not in df.columns or fl_col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Plot score as line
        #ax.plot(df.index, df[sc_col], label="TSPulse Score", color="blue", lw=1.5, alpha=0.8)
        
        # Fill fire flags
        ax.fill_between(df.index, 0, df[fl_col]*1.0, color="red", alpha=0.3, label="Fire Flag")

        # Set global time range if provided
        if global_time_min is not None and global_time_max is not None:
            ax.set_xlim(global_time_min, global_time_max)

        ax.set_title(f"TSPulse Fire Detection Timeline – {st}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right")
        ax.grid(True, ls="--", alpha=0.5)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f"plots/tspulse/{st}_tspulse_timeline.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✅ Saved TSPulse plots for {len(stations)} stations in ./plots/tspulse/")

def plot_outputs_timeline(global_time_min=None, global_time_max=None):
    """Plot fire events timeline from outputs directory"""
    print("📊 Plotting outputs directory fire events...")
    
    # Create output folder
    os.makedirs("plots/outputs", exist_ok=True)
    
    # Find all fire_events.csv files
    fire_event_files = glob.glob("outputs/*/*fire_events.csv") 
    
    if not fire_event_files:
        print("⚠️  No fire_events.csv files found in outputs directory")
        return
    
    print(f"📁 Found {len(fire_event_files)} fire event files")
    
    for fire_file in fire_event_files:
        # Extract station name from file path
        station = os.path.basename(os.path.dirname(fire_file)) if os.path.dirname(fire_file) != "outputs" else fire_file.split('/')[-1].replace('_fire_events.csv', '').replace('_pm_fire_events.csv', '')
        
        try:
            # Read fire events data
            events = pd.read_csv(fire_file)
            
            if events.empty:
                print(f"⚠️  Empty data file: {fire_file}")
                continue
            
            # Convert time columns
            events['start'] = pd.to_datetime(events['start'])
            events['end'] = pd.to_datetime(events['end'])
            
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Create a timeline plot similar to TSPulse using fill_between
            # Use global time range if provided, otherwise use event time range
            if global_time_min is not None and global_time_max is not None:
                time_min = global_time_min
                time_max = global_time_max
            else:
                time_min = events['start'].min()
                time_max = events['end'].max()
            
            # Create a regular time series (e.g., every 5 minutes)
            time_series = pd.date_range(start=time_min, end=time_max, freq='5min')
            fire_flag = pd.Series(0, index=time_series)
            
            # Mark fire events in the time series
            for idx, event in events.iterrows():
                # Set fire flag to 1 during event duration
                mask = (time_series >= event['start']) & (time_series <= event['end'])
                fire_flag[mask] = 1
            
            # Plot using fill_between like TSPulse
            ax.fill_between(time_series, 0, fire_flag, color="red", alpha=0.3, label="Fire Events")
            
            # Set global time range
            ax.set_xlim(time_min, time_max)
            
            ax.set_title(f"Fire Events Timeline – {station}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Fire Flag")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"plots/outputs/{station}_fire_events_timeline.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Plotted fire events timeline for station: {station}")
            
        except Exception as e:
            print(f"❌ Error plotting {fire_file}: {e}")
    
    print("✅ Outputs timeline processing completed")

def plot_tspulse_events_summary():
    """Plot TSPulse fire events summary"""
    print("📊 Creating TSPulse fire events summary...")
    
    os.makedirs("plots/summary", exist_ok=True)
    
    try:
        tsp_events = pd.read_csv("tsp_output/events_out_tspulse.csv")
        if not tsp_events.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Convert time columns to datetime
            tsp_events['t_start'] = pd.to_datetime(tsp_events['t_start'])
            tsp_events['t_end'] = pd.to_datetime(tsp_events['t_end'])
            
            # Plot events as horizontal bars
            for idx, event in tsp_events.iterrows():
                duration = (event['t_end'] - event['t_start']).total_seconds() / 60  # duration in minutes
                y_pos = idx
                
                # Color by level
                color = 'red' if 'Level-2' in str(event.get('level_blend', '')) else 'orange'
                
                ax.barh(y_pos, duration, left=event['t_start'], height=0.8, 
                       color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Add event info as text
                ax.text(event['t_start'], y_pos, f"ID:{event['event_id']}", 
                       va='center', ha='left', fontsize=8)
            
            ax.set_title("TSPulse Fire Events Timeline")
            ax.set_xlabel("Time")
            ax.set_ylabel("Event ID")
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("plots/summary/tspulse_events_timeline.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print("✅ Created TSPulse events summary plot")
            
    except Exception as e:
        print(f"❌ Error plotting TSPulse events: {e}")

def plot_outputs_fire_events_summary():
    """Plot fire events summary from outputs directory"""
    print("📊 Creating outputs fire events summary...")
    
    os.makedirs("plots/summary", exist_ok=True)
    
    # Plot outputs fire events
    fire_event_files = glob.glob("outputs/*/*fire_events.csv") 
    
    for fire_file in fire_event_files:
        try:
            events = pd.read_csv(fire_file)
            if events.empty:
                continue
                
            station = os.path.basename(os.path.dirname(fire_file)) if os.path.dirname(fire_file) != "outputs" else fire_file.split('/')[-1].replace('_pm_fire_events.csv', '').replace('_pm_fire_events_scored.csv', '')
            
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Convert time columns
            events['start'] = pd.to_datetime(events['start'])
            events['end'] = pd.to_datetime(events['end'])
            
            # Plot events as horizontal bars showing duration range
            for idx, event in events.iterrows():
                y_pos = idx
                
                # Color by level
                color = 'red' if 'Confirmed Fire' in str(event.get('level', '')) else 'orange'
                
                # Draw horizontal bar from start to end time
                ax.barh(y_pos, 1, left=event['start'], height=0.8, 
                       color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Add event duration info
                duration_min = (event['end'] - event['start']).total_seconds() / 60
                ax.text(event['start'], y_pos, f"Event {idx+1} ({duration_min:.1f}min)", 
                       va='center', ha='left', fontsize=9)
            
            # Set proper time range
            if not events.empty:
                time_min = events['start'].min()
                time_max = events['end'].max()
                # Add some padding
                time_range = time_max - time_min
                ax.set_xlim(time_min - time_range*0.02, time_max + time_range*0.02)
            
            ax.set_title(f"Fire Events Duration Timeline – {station}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Event ID")
            if not events.empty:
                ax.set_ylim(-0.5, len(events)-0.5)
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"plots/summary/{station}_fire_events_timeline.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Created fire events summary for station: {station}")
            
        except Exception as e:
            print(f"❌ Error plotting {fire_file}: {e}")

def main():
    """Main function to run all plotting functions"""
    print("🔥 Fire Detection Timeline Plotter")
    print("=" * 50)
    
    # Get global time range from PM1 data
    global_time_min, global_time_max = get_global_time_range()
    
    # Create main plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Plot TSPulse timeline with global time range
    plot_tspulse_timeline(global_time_min, global_time_max)
    
    # Plot outputs timeline with global time range
    plot_outputs_timeline(global_time_min, global_time_max)
    
    # Plot fire events summary
    plot_tspulse_events_summary()
    plot_outputs_fire_events_summary()
    
    print("\n🎉 All plotting completed!")
    print("📁 Check the 'plots' directory for generated visualizations:")
    print("   - plots/tspulse/     : TSPulse timeline plots")
    print("   - plots/outputs/     : Fire events timeline plots")
    print("   - plots/summary/     : Fire events summary plots")

if __name__ == "__main__":
    main()
