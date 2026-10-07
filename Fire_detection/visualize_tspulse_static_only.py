#!/usr/bin/env python3
"""
Visualize TSPulse fire events on a map with real-fire labels (9/11–9/18)
Static map only - no animation to avoid complexity
"""

import argparse
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from datetime import datetime, timedelta

# Headless backend for CI/container
import matplotlib
matplotlib.use('Agg')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize TSPulse fire events (static map only)")
    parser.add_argument("--tspulse_csv", type=str, default="tsp_output/tspulse_per_minute.csv")
    parser.add_argument("--location_csv", type=str, default="data/Location.csv")
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--map_style", type=str, default="admin", choices=["admin", "terrain", "satellite", "osm", "google", "google_sat", "google_terrain", "google_road", "esri"])
    parser.add_argument("--tile_zoom", type=int, default=13)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--animation", action="store_true", help="Create animated map")
    parser.add_argument("--fps", type=int, default=2, help="Animation frames per second")
    parser.add_argument("--time_step_minutes", type=int, default=60, help="Time step for animation in minutes")
    parser.add_argument("--output_format", type=str, default="gif", choices=["gif", "mp4"], help="Output format for animation")
    return parser.parse_args()


def load_station_locations(location_csv: str, exclude_distant: bool = True) -> pd.DataFrame:
    df = pd.read_csv(location_csv)
    coords = df['Location'].str.extract(r'([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)')
    df['lat'] = pd.to_numeric(coords[0], errors='coerce')
    df['lon'] = pd.to_numeric(coords[1], errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])
    df = df[df['Status'].str.strip().str.lower() == 'active']
    
    # Fix negative longitude values (should be positive)
    df['lon'] = df['lon'].abs()
    
    if exclude_distant:
        df = df[df['StationID'] != 'aa-87-15-02']
    
    return df[['StationID', 'lat', 'lon']].copy()


def load_tspulse_data(tspulse_csv: str) -> pd.DataFrame:
    """Load TSPulse per-minute data"""
    df = pd.read_csv(tspulse_csv)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
    return df


def kst_to_utc(ts_str: str) -> pd.Timestamp:
    ts_kst = pd.to_datetime(ts_str)
    ts_utc = (ts_kst - timedelta(hours=9)).tz_localize('UTC') if ts_kst.tz is None else ts_kst.tz_convert('UTC')
    return ts_utc


def get_real_fire_windows_utc() -> List[Dict[str, pd.Timestamp]]:
    """Get real fire windows in UTC"""
    windows = [
        # Dae-Cheong Lake events
        ("2025-09-11 19:00 KST", "2025-09-11 19:25 KST", "Dae-Cheong Lake"),
        ("2025-09-12 08:15 KST", "2025-09-12 08:45 KST", "Dae-Cheong Lake"),
        ("2025-09-16 19:10 KST", "2025-09-16 20:00 KST", "Dae-Cheong Lake"),
        ("2025-09-17 09:20 KST", "2025-09-17 10:25 KST", "Dae-Cheong Lake"),
        # Cherry Blossom Road events
        ("2025-09-11 19:40 KST", "2025-09-11 20:00 KST", "Cherry Blossom Road"),
        ("2025-09-16 18:00 KST", "2025-09-16 18:30 KST", "Cherry Blossom Road"),
    ]
    
    out = []
    for start_str, end_str, location in windows:
        start_utc = kst_to_utc(start_str)
        end_utc = kst_to_utc(end_str)
        
        # Extend duration by 1 hour on each side for visibility
        start_utc = start_utc - timedelta(hours=1)
        end_utc = end_utc + timedelta(hours=1)
        
        out.append({
            "start": start_utc,
            "end": end_utc,
            "location": location,
        })
    return out


def filter_tspulse_by_timerange(df: pd.DataFrame, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    """Filter TSPulse data to time range"""
    mask = (df['timestamp_utc'] >= start_utc) & (df['timestamp_utc'] <= end_utc)
    return df.loc[mask].copy()


def get_station_color_at_time(station_id: str, current_time: pd.Timestamp, df_range: pd.DataFrame) -> str:
    """Get station color based on TSPulse detection at given timestamp"""
    flag_col = f"{station_id}_tspulse_flag"
    
    if flag_col not in df_range.columns:
        return '#1f77b4'  # Blue for no data
    
    # Find closest timestamp in data
    time_diff = abs(df_range['timestamp_utc'] - current_time)
    closest_idx = time_diff.idxmin()
    
    # Check if TSPulse flag is 1 (fire detected)
    if df_range.loc[closest_idx, flag_col] == 1:
        return '#d62728'  # Red for fire detected
    else:
        return '#1f77b4'  # Blue for no fire


def add_background_map(ax, map_style: str = "admin", tile_zoom: int = 13, offline: bool = False) -> None:
    ax.add_feature(cfeature.LAND, alpha=0.4, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, alpha=0.4, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5)
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    if offline:
        try:
            ax.stock_img()
        except Exception:
            pass
        return
    try:
        if map_style == "osm":
            osm = cimgt.OSM(); ax.add_image(osm, tile_zoom)
        elif map_style == "google":
            google = cimgt.GoogleTiles(); ax.add_image(google, tile_zoom)
        elif map_style == "google_sat":
            google = cimgt.GoogleTiles(url='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'); ax.add_image(google, tile_zoom)
        elif map_style == "google_terrain":
            google = cimgt.GoogleTiles(url='https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}'); ax.add_image(google, tile_zoom)
        elif map_style == "google_road":
            google = cimgt.GoogleTiles(url='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}'); ax.add_image(google, tile_zoom)
        elif map_style == "esri":
            esri = cimgt.GoogleTiles(url='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'); ax.add_image(esri, tile_zoom)
        elif map_style == "terrain":
            terrain = cimgt.Stamen('terrain'); ax.add_image(terrain, tile_zoom)
        elif map_style == "satellite":
            satellite = cimgt.Stamen('satellite'); ax.add_image(satellite, tile_zoom)
        else:
            ax.add_feature(cfeature.COASTLINE, linewidth=2)
    except Exception as e:
        print(f"Warning: Could not load {map_style} tiles: {e}")
        ax.add_feature(cfeature.COASTLINE, linewidth=2)
    ax.add_feature(cfeature.RIVERS, alpha=0.6, linewidth=0.8)
    ax.add_feature(cfeature.LAKES, alpha=0.6, facecolor='lightblue')


def calculate_map_extent(locations_df: pd.DataFrame, buffer: float = 0.005):
    """Calculate tighter map extent around stations"""
    if locations_df.empty:
        return 127.35, 127.55, 36.295, 36.40
    
    # Use smaller buffer for tighter view
    min_lon = locations_df['lon'].min() - buffer
    max_lon = locations_df['lon'].max() + buffer
    min_lat = locations_df['lat'].min() - buffer
    max_lat = locations_df['lat'].max() + buffer
    
    return min_lon, max_lon, min_lat, max_lat


def create_static_map(tspulse_df: pd.DataFrame, locations_df: pd.DataFrame,
                     real_windows: List[Dict[str, pd.Timestamp]],
                     start_utc: pd.Timestamp, end_utc: pd.Timestamp,
                     output_dir: Path, map_style: str = "admin",
                     tile_zoom: int = 13, offline: bool = False) -> None:
    """Create static map showing TSPulse fire detections vs real fire events"""
    
    # Filter TSPulse data to time range
    df_range = filter_tspulse_by_timerange(tspulse_df, start_utc, end_utc)
    
    # Determine which stations have TSPulse fire detections during the time range
    stations_with_detections = set()
    
    for _, station in locations_df.iterrows():
        station_id = station['StationID']
        flag_col = f"{station_id}_tspulse_flag"
        
        if flag_col in df_range.columns:
            # Check if any TSPulse flags are 1 (fire detected) in the time range
            fire_detections = df_range[flag_col].sum()
            if fire_detections > 0:
                stations_with_detections.add(station_id)
                print(f"Station {station_id}: {fire_detections} TSPulse fire detections")
    
    print(f"Stations with TSPulse fire detections: {sorted(stations_with_detections)}")
    
    # Create map
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    add_background_map(ax, map_style, tile_zoom, offline)

    min_lon, max_lon, min_lat, max_lat = calculate_map_extent(locations_df)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # Scatter all active stations with colors based on TSPulse fire detection
    for _, st in locations_df.iterrows():
        if st['StationID'] in stations_with_detections:
            # Red for stations that detected fire
            color = '#d62728'
            edge_color = '#8b0000'
        else:
            # Blue for stations with no fire detection
            color = '#1f77b4'
            edge_color = '#000080'
        
        ax.scatter(st['lon'], st['lat'], c=color, s=80, alpha=0.9,
                   edgecolors=edge_color, linewidth=1, transform=ccrs.PlateCarree(), marker='o')
        ax.text(st['lon'] + 0.001, st['lat'] + 0.001, st['StationID'], fontsize=7,
                transform=ccrs.PlateCarree())

    # Build set of real fire locations in our time range
    real_fire_locations = []
    
    for win in real_windows:
        # Only consider windows that intersect the map time range
        if (win['start'] <= end_utc) and (win['end'] >= start_utc):
            real_fire_locations.append(win['location'])
    
    print(f"Real fire locations in map range: {real_fire_locations}")

    # Place real fire event markers as triangles at exact locations
    real_fire_coords = {
        'Dae-Cheong Lake': (127.491808, 36.354333),  # Exact coordinates
        'Cherry Blossom Road': (127.497597, 36.354356)  # Exact coordinates
    }
    
    # Mark real fire events with triangles (no text labels)
    for location in real_fire_locations:
        if location in real_fire_coords:
            lon, lat = real_fire_coords[location]
            ax.scatter(lon, lat, c='orange', s=300, marker='^', 
                      alpha=0.9, edgecolors='darkorange', linewidth=3, 
                      transform=ccrs.PlateCarree(), zorder=10)

    # Title with time window in KST (for readability)
    start_kst = (start_utc.tz_convert('Asia/Seoul') if start_utc.tz else start_utc.tz_localize('UTC').tz_convert('Asia/Seoul'))
    end_kst = (end_utc.tz_convert('Asia/Seoul') if end_utc.tz else end_utc.tz_localize('UTC').tz_convert('Asia/Seoul'))
    ax.set_title(f"TSPulse Fire Events Map: Station Detections vs Real Fire Events\nTime Window: {start_kst.strftime('%Y-%m-%d %H:%M KST')} – {end_kst.strftime('%Y-%m-%d %H:%M KST')}",
                 fontsize=14, fontweight='bold', pad=20)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, label='Station (No TSPulse Fire Detected)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=10, label='Station (TSPulse Fire Detected)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=12, label='Real Fire Event'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Save
    output_path = output_dir / "tspulse_fire_events_map_static_working.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved static TSPulse map: {output_path}")


def create_animated_map(tspulse_df: pd.DataFrame, locations_df: pd.DataFrame,
                       real_windows: List[Dict[str, pd.Timestamp]],
                       start_utc: pd.Timestamp, end_utc: pd.Timestamp,
                       output_dir: Path, map_style: str = "admin",
                       tile_zoom: int = 13, offline: bool = False,
                       fps: int = 2, time_step_minutes: int = 60, 
                       output_format: str = "gif") -> None:
    """Create animated map showing TSPulse fire detections over time"""
    
    # Create time range for animation
    time_range = pd.date_range(start_utc, end_utc, freq=f'{time_step_minutes}min')
    
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    add_background_map(ax, map_style, tile_zoom, offline)
    
    min_lon, max_lon, min_lat, max_lat = calculate_map_extent(locations_df)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    # Filter TSPulse data to our time range
    df_range = filter_tspulse_by_timerange(tspulse_df, start_utc, end_utc)
    
    # Initialize scatter plot for stations
    scatter = ax.scatter([], [], s=80, alpha=0.9, edgecolors='black', linewidth=1,
                        transform=ccrs.PlateCarree())
    
    # Add station labels (static)
    for _, st in locations_df.iterrows():
        ax.text(st['lon'] + 0.001, st['lat'] + 0.001, st['StationID'], fontsize=7,
                transform=ccrs.PlateCarree())
    
    # Initialize real fire markers (will be animated)
    real_fire_coords = {
        'Dae-Cheong Lake': (127.491808, 36.354333),  # Exact coordinates
        'Cherry Blossom Road': (127.497597, 36.354356)  # Exact coordinates
    }
    
    # Find real fire locations in our time range
    real_fire_locations = []
    for win in real_windows:
        if (win['start'] <= end_utc) and (win['end'] >= start_utc):
            real_fire_locations.append(win['location'])
    
    # Initialize scatter plot for real fire markers
    real_fire_scatter = ax.scatter([], [], c='orange', s=300, marker='^', 
                                  alpha=0.9, edgecolors='darkorange', linewidth=3, 
                                  transform=ccrs.PlateCarree(), zorder=10)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, label='Station (No TSPulse Fire)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=10, label='Station (TSPulse Fire Detected)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=12, label='Real Fire Event'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    def animate(frame):
        current_time = time_range[frame]
        
        # Update station colors
        colors = []
        sizes = []
        lons = []
        lats = []
        
        for _, st in locations_df.iterrows():
            color = get_station_color_at_time(st['StationID'], current_time, df_range)
            colors.append(color)
            sizes.append(80)
            lons.append(st['lon'])
            lats.append(st['lat'])
        
        scatter.set_offsets(np.column_stack([lons, lats]))
        scatter.set_color(colors)
        scatter.set_sizes(sizes)
        
        # Update real fire markers
        fire_lons = []
        fire_lats = []
        fire_colors = []
        
        for location in real_fire_locations:
            if location in real_fire_coords:
                # Check if current time is within any real fire window for this location
                in_fire_window = False
                for win in real_windows:
                    if (win['location'] == location and 
                        win['start'] <= current_time <= win['end']):
                        in_fire_window = True
                        break
                
                if in_fire_window:
                    lon, lat = real_fire_coords[location]
                    fire_lons.append(lon)
                    fire_lats.append(lat)
                    fire_colors.append('orange')
        
        if fire_lons:
            real_fire_scatter.set_offsets(np.column_stack([fire_lons, fire_lats]))
            real_fire_scatter.set_color(fire_colors)
        else:
            real_fire_scatter.set_offsets(np.column_stack([[], []]))
        
        # Update title with current time
        current_kst = current_time.tz_convert('Asia/Seoul')
        ax.set_title(f"TSPulse Fire Events Map (Animated)\nCurrent Time: {current_kst.strftime('%Y-%m-%d %H:%M KST')}",
                     fontsize=14, fontweight='bold', pad=20)
        
        return scatter, real_fire_scatter

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(time_range), 
                                  interval=1000//fps, blit=False, repeat=True)
    
    # Save animation in specified format
    if output_format == "mp4":
        output_path = output_dir / "tspulse_fire_events_map_animated_working.mp4"
        try:
            # Try to use ffmpeg writer for MP4
            anim.save(output_path, writer='ffmpeg', fps=fps, bitrate=1800)
            print(f"Saved animated TSPulse map as MP4: {output_path}")
        except Exception as e:
            print(f"Failed to save as MP4: {e}")
            print("Falling back to GIF format...")
            output_path = output_dir / "tspulse_fire_events_map_animated_working.gif"
            anim.save(output_path, writer='pillow', fps=fps)
            print(f"Saved animated TSPulse map as GIF: {output_path}")
    else:
        output_path = output_dir / "tspulse_fire_events_map_animated_working.gif"
        anim.save(output_path, writer='pillow', fps=fps)
        print(f"Saved animated TSPulse map as GIF: {output_path}")
    
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading station locations...")
    locations_df = load_station_locations(args.location_csv, exclude_distant=True)
    print(f"Loaded {len(locations_df)} active stations")

    print("Loading TSPulse data...")
    tspulse_df = load_tspulse_data(args.tspulse_csv)
    print(f"Loaded {len(tspulse_df)} TSPulse records")

    # Date range: 9/11–9/18 in KST => UTC: 2025-09-11 00:00 to 2025-09-18 15:00
    start_utc = pd.Timestamp('2025-09-11 00:00:00', tz='UTC')
    end_utc = pd.Timestamp('2025-09-18 15:00:00', tz='UTC')

    # Real-fire windows in UTC
    real_windows = get_real_fire_windows_utc()

    print("Creating static TSPulse map...")
    create_static_map(tspulse_df, locations_df, real_windows, start_utc, end_utc,
                     output_dir, args.map_style, args.tile_zoom, args.offline)

    if args.animation:
        print("Creating animated TSPulse map...")
        create_animated_map(tspulse_df, locations_df, real_windows, start_utc, end_utc,
                           output_dir, args.map_style, args.tile_zoom, args.offline,
                           args.fps, args.time_step_minutes, args.output_format)

    print("Done.")


if __name__ == "__main__":
    main()
