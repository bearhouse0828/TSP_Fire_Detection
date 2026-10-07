#!/usr/bin/env python3
"""
Visualize TSPulse fire events on a map with dynamic station colors.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from datetime import datetime, timedelta

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize TSPulse fire events on station map")
    parser.add_argument("--tspulse_csv", type=str, default="tsp_output/tspulse_per_minute.csv")
    parser.add_argument("--location_csv", type=str, default="data/Location.csv")
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--animation", action="store_true")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--time_step_minutes", type=int, default=30)
    parser.add_argument("--exclude_distant", action="store_true", default=False, help="Exclude distant station aa-87-15-02")
    parser.add_argument("--map_style", type=str, default="admin", choices=["admin", "terrain", "satellite", "osm", "google", "google_sat", "google_terrain", "google_road", "esri"])
    parser.add_argument("--tile_zoom", type=int, default=13)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def load_station_locations(location_csv: str, exclude_distant: bool = True) -> pd.DataFrame:
    """Load station locations from CSV."""
    df = pd.read_csv(location_csv)
    coords = df['Location'].str.extract(r'([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)')
    df['lat'] = pd.to_numeric(coords[0], errors='coerce')
    df['lon'] = pd.to_numeric(coords[1], errors='coerce')
    
    df = df.dropna(subset=['lat', 'lon'])
    df = df[df['Status'].str.strip().str.lower() == 'active']
    
    if exclude_distant:
        df = df[df['StationID'] != 'aa-87-15-02']
    
    return df[['StationID', 'lat', 'lon']].copy()


def load_tspulse_data(tspulse_csv: str) -> pd.DataFrame:
    """Load TSPulse data from CSV."""
    try:
        df = pd.read_csv(tspulse_csv, index_col=0, parse_dates=True)
        print(f"Loaded TSPulse data with {len(df)} time points")
        print(f"Time range: {df.index.min()} to {df.index.max()}")
        
        # Get station columns (those ending with _tspulse_flag)
        flag_columns = [col for col in df.columns if col.endswith('_tspulse_flag')]
        score_columns = [col for col in df.columns if col.endswith('_tspulse_score')]
        
        print(f"Found {len(flag_columns)} stations with TSPulse data:")
        for col in flag_columns:
            station_id = col.replace('_tspulse_flag', '')
            print(f"  {station_id}")
        
        return df
        
    except Exception as e:
        print(f"Error loading TSPulse data: {e}")
        return pd.DataFrame()


def get_station_tspulse_status(station_id: str, current_time: datetime, tspulse_df: pd.DataFrame) -> Tuple[float, str]:
    """Get station TSPulse status based on current time."""
    if tspulse_df.empty:
        return 0.0, '#1f77b4'
    
    # Find the closest time point
    time_diff = abs(tspulse_df.index - current_time)
    closest_idx = tspulse_df.index[time_diff.argmin()]
    
    # Get flag and score columns for this station
    flag_col = f"{station_id}_tspulse_flag"
    score_col = f"{station_id}_tspulse_score"
    
    if flag_col not in tspulse_df.columns:
        return 0.0, '#1f77b4'
    
    flag_value = tspulse_df.loc[closest_idx, flag_col]
    score_value = tspulse_df.loc[closest_idx, score_col] if score_col in tspulse_df.columns else 0.0
    
    # Color mapping: blue (no fire) to red (fire)
    if flag_value == 0:
        return 0.0, '#1f77b4'
    else:
        # Use score as intensity (normalize to 0-1)
        intensity = min(score_value, 1.0) if score_value is not None else 1.0
        return intensity, '#d62728'


def create_google_tiles(style: str):
    """Create Google Maps tiles using the provided URL pattern."""
    if style == 'google_sat':
        url = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
    elif style == 'google_terrain':
        url = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}'
    elif style == 'google_road':
        url = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}'
    else:  # default to roadmap
        url = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}'
    
    return cimgt.GoogleTiles(url=url)


def add_background_map(ax, map_style: str = "admin", tile_zoom: int = 13, offline: bool = False) -> None:
    """Add background map layers based on style."""
    ax.add_feature(cfeature.LAND, alpha=0.4, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, alpha=0.4, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5)
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    
    if not offline:
        try:
            if map_style == "terrain":
                terrain = cimgt.GoogleTiles(style="terrain")
                ax.add_image(terrain, tile_zoom)
                print("Loaded terrain tiles successfully")
            elif map_style == "satellite":
                satellite = cimgt.GoogleTiles(style="satellite")
                ax.add_image(satellite, tile_zoom)
                print("Loaded satellite tiles successfully")
            elif map_style == "osm":
                osm = cimgt.OSM()
                ax.add_image(osm, tile_zoom)
                print("Loaded OSM tiles successfully")
            elif map_style == "google":
                google = cimgt.GoogleTiles()
                ax.add_image(google, tile_zoom)
                print("Loaded Google tiles successfully")
            elif map_style in ["google_sat", "google_terrain", "google_road"]:
                google_tiles = create_google_tiles(map_style)
                ax.add_image(google_tiles, tile_zoom)
                print(f"Loaded Google {map_style} tiles successfully")
            elif map_style == "esri":
                esri = cimgt.GoogleTiles(url='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}')
                ax.add_image(esri, tile_zoom)
                print("Loaded Esri tiles successfully")
            else:
                print("Using basic administrative features...")
        except Exception as e:
            print(f"Failed to load {map_style} tiles: {e}")
            print("Using basic administrative features...")
    
    # Add additional features
    ax.add_feature(cfeature.RIVERS, alpha=0.6, linewidth=0.8)
    ax.add_feature(cfeature.LAKES, alpha=0.6, facecolor='lightblue')


def calculate_map_extent(locations_df: pd.DataFrame, buffer: float = 0.02) -> Tuple[float, float, float, float]:
    """Calculate map extent based on station locations with buffer."""
    if locations_df.empty:
        return 127.35, 127.55, 36.295, 36.40
    
    # Calculate coordinate ranges
    lon_range = locations_df['lon'].max() - locations_df['lon'].min()
    lat_range = locations_df['lat'].max() - locations_df['lat'].min()
    
    # Adjust buffer to make the map wider (more longitude range)
    # Target aspect ratio: make longitude range similar to latitude range
    target_lon_range = lat_range * 1.2  # Make longitude range 120% of latitude range
    lon_buffer = (target_lon_range - lon_range) / 2
    
    min_lon = locations_df['lon'].min() - max(buffer, lon_buffer)
    max_lon = locations_df['lon'].max() + max(buffer, lon_buffer)
    min_lat = locations_df['lat'].min() - buffer
    max_lat = locations_df['lat'].max() + buffer
    
    print(f"Station coordinate ranges:")
    print(f"  Longitude: {locations_df['lon'].min():.6f} to {locations_df['lon'].max():.6f} (range: {lon_range:.6f})")
    print(f"  Latitude: {locations_df['lat'].min():.6f} to {locations_df['lat'].max():.6f} (range: {lat_range:.6f})")
    print(f"  Adjusted longitude buffer: {max(buffer, lon_buffer):.6f}")
    print(f"  Map extent: [{min_lon:.3f}, {max_lon:.3f}, {min_lat:.3f}, {max_lat:.3f}]")
    
    return min_lon, max_lon, min_lat, max_lat


def create_static_map(tspulse_df: pd.DataFrame, locations_df: pd.DataFrame, output_dir: Path, map_style: str = "admin", tile_zoom: int = 13, offline: bool = False) -> None:
    """Create a static map showing TSPulse fire events."""
    fig = plt.figure(figsize=(24, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    add_background_map(ax, map_style, tile_zoom, offline)
    
    min_lon, max_lon, min_lat, max_lat = calculate_map_extent(locations_df)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    if not tspulse_df.empty:
        # Use middle time point for static map
        mid_time = tspulse_df.index[len(tspulse_df)//2]
        
        for _, station in locations_df.iterrows():
            color_val, color_hex = get_station_tspulse_status(station['StationID'], mid_time, tspulse_df)
            
            ax.scatter(station['lon'], station['lat'], c=color_hex, s=100, alpha=0.8, 
                      edgecolors='black', linewidth=1, transform=ccrs.PlateCarree())
            
            ax.text(station['lon'] + 0.001, station['lat'] + 0.001, station['StationID'], 
                   fontsize=6, transform=ccrs.PlateCarree())
    else:
        # No data, show all stations in blue
        for _, station in locations_df.iterrows():
            ax.scatter(station['lon'], station['lat'], c='#1f77b4', s=100, alpha=0.8, 
                      edgecolors='black', linewidth=1, transform=ccrs.PlateCarree())
            
            ax.text(station['lon'] + 0.001, station['lat'] + 0.001, station['StationID'], 
                   fontsize=6, transform=ccrs.PlateCarree())
    
    ax.set_title('Fire Events by TSPulse Method', fontsize=14, fontweight='bold')
    ax.gridlines(draw_labels=True, alpha=0.5, xlocs=[127.46, 127.48, 127.50, 127.52], ylocs=[36.30, 36.32, 36.34, 36.36, 36.38])
    
    legend_elements = [
        plt.scatter([], [], c='#1f77b4', s=100, label='No Fire'),
        plt.scatter([], [], c='#d62728', s=100, label='Fire')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / f"tspulse_fire_events_map_static.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved static map: {output_path}")


def create_timeline_visualization(tspulse_df: pd.DataFrame, output_dir: Path) -> None:
    """Create a timeline visualization of TSPulse fire events."""
    if tspulse_df.empty:
        print("No TSPulse data to visualize in timeline")
        return
    
    # Get flag columns
    flag_columns = [col for col in tspulse_df.columns if col.endswith('_tspulse_flag')]
    if not flag_columns:
        print("No TSPulse flag columns found")
        return
    
    # Calculate the actual time span
    time_span_hours = (tspulse_df.index.max() - tspulse_df.index.min()).total_seconds() / 3600
    
    # Adjust figure size based on time span
    fig_width = max(12, min(24, time_span_hours * 0.1))
    fig = plt.figure(figsize=(fig_width, 8))
    
    # Create timeline for each station
    colors = plt.cm.Set3(np.linspace(0, 1, len(flag_columns)))
    
    for i, flag_col in enumerate(flag_columns):
        station_id = flag_col.replace('_tspulse_flag', '')
        station_flags = tspulse_df[flag_col]
        
        # Find fire periods (consecutive 1s)
        fire_periods = []
        in_fire = False
        start_time = None
        
        for time_idx, flag in station_flags.items():
            if flag == 1 and not in_fire:
                # Start of fire period
                in_fire = True
                start_time = time_idx
            elif flag == 0 and in_fire:
                # End of fire period
                in_fire = False
                if start_time is not None:
                    fire_periods.append((start_time, time_idx))
                    start_time = None
        
        # Handle case where fire period extends to end
        if in_fire and start_time is not None:
            fire_periods.append((start_time, tspulse_df.index.max()))
        
        # Plot fire periods
        for start_time, end_time in fire_periods:
            start_offset = (start_time - tspulse_df.index.min()).total_seconds() / 3600
            duration = (end_time - start_time).total_seconds() / 3600
            
            plt.barh(i, duration, left=start_offset, height=0.8, 
                    color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
    
    plt.yticks(range(len(flag_columns)), [col.replace('_tspulse_flag', '') for col in flag_columns])
    plt.xlabel('Time (hours from start)')
    plt.ylabel('Station')
    plt.title('TSPulse Fire Events Timeline by Station', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add time axis labels
    start_time = tspulse_df.index.min()
    time_points = pd.date_range(start_time, tspulse_df.index.max(), freq='24H')
    time_labels = [(t - start_time).total_seconds() / 3600 for t in time_points]
    plt.xticks(time_labels, [t.strftime('%m-%d') for t in time_points], rotation=45)
    
    plt.tight_layout()
    output_path = output_dir / f"tspulse_fire_events_timeline.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved timeline: {output_path}")


def create_animated_map(tspulse_df: pd.DataFrame, locations_df: pd.DataFrame, output_dir: Path, 
                       map_style: str = "admin", tile_zoom: int = 13, fps: int = 2, 
                       time_step_minutes: int = 30, offline: bool = False) -> None:
    """Create an animated map showing TSPulse fire events over time."""
    if tspulse_df.empty:
        print("No TSPulse data to animate")
        return
    
    # Create time range based on TSPulse data
    start_time = tspulse_df.index.min()
    end_time = tspulse_df.index.max()
    time_range = pd.date_range(start_time, end_time, freq=f'{time_step_minutes}min')
    
    fig = plt.figure(figsize=(24, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add background map layers
    add_background_map(ax, map_style, tile_zoom, offline)
    
    # Calculate and set map extent based on station locations
    min_lon, max_lon, min_lat, max_lat = calculate_map_extent(locations_df)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    # Initialize scatter plot
    scatter = ax.scatter([], [], s=100, alpha=0.8, edgecolors='black', linewidth=1,
                        transform=ccrs.PlateCarree())
    
    # Add station labels
    for _, station in locations_df.iterrows():
        ax.text(station['lon'] + 0.001, station['lat'] + 0.001, 
               station['StationID'], fontsize=6, 
               transform=ccrs.PlateCarree())
    
    ax.set_title('Fire Events by TSPulse Method - Dynamic Status', fontsize=14, fontweight='bold')
    ax.gridlines(draw_labels=True, alpha=0.5, xlocs=[127.46, 127.48, 127.50, 127.52], ylocs=[36.30, 36.32, 36.34, 36.36, 36.38])
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], c='#1f77b4', s=100, label='No Fire'),
        plt.scatter([], [], c='#d62728', s=100, label='Fire')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    def animate(frame):
        current_time = time_range[frame]
        
        colors = []
        sizes = []
        lons = []
        lats = []
        
        for _, station in locations_df.iterrows():
            color_val, color_hex = get_station_tspulse_status(station['StationID'], current_time, tspulse_df)
            colors.append(color_hex)
            sizes.append(100 + color_val * 50)  # Size varies with intensity
            lons.append(station['lon'])
            lats.append(station['lat'])
        
        scatter.set_offsets(np.column_stack([lons, lats]))
        scatter.set_color(colors)
        scatter.set_sizes(sizes)
        
        ax.set_title(f'Fire Events by TSPulse Method - {current_time.strftime("%Y-%m-%d %H:%M")}', 
                    fontsize=14, fontweight='bold')
        
        return scatter,
    
    anim = animation.FuncAnimation(fig, animate, frames=len(time_range), 
                                  interval=1000//fps, blit=False, repeat=True)
    
    output_path = output_dir / f"tspulse_fire_events_map_animated.gif"
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()
    print(f"Saved animated map: {output_path}")


def main():
    args = parse_args()
    
    print("🔥 TSPulse Fire Events Map Visualizer")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading station locations...")
    locations_df = load_station_locations(args.location_csv, args.exclude_distant)
    print(f"Loaded {len(locations_df)} active stations")
    
    print("Loading TSPulse data...")
    tspulse_df = load_tspulse_data(args.tspulse_csv)
    if not tspulse_df.empty:
        print(f"Loaded TSPulse data with {len(tspulse_df)} time points")
    
    # Create visualizations
    print(f"Creating static map with {args.map_style} background (zoom={args.tile_zoom})...")
    create_static_map(tspulse_df, locations_df, output_dir, args.map_style, args.tile_zoom, args.offline)
    
    print("Creating timeline visualization...")
    create_timeline_visualization(tspulse_df, output_dir)
    
    if args.animation:
        print("Creating animated map...")
        create_animated_map(tspulse_df, locations_df, output_dir, args.map_style, 
                           args.tile_zoom, args.fps, args.time_step_minutes, args.offline)
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()
