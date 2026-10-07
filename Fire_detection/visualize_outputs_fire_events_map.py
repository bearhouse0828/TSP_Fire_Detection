#!/usr/bin/env python3
"""
Visualize fire events from outputs directory on a map with dynamic station colors.
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
import glob

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize outputs fire events on station map")
    parser.add_argument("--outputs_dir", type=str, default="outputs")
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


def load_outputs_fire_events(outputs_dir: str) -> pd.DataFrame:
    """Load fire events from outputs directory."""
    fire_events = []
    
    # Find all fire_events.csv files in outputs directory
    pattern = f"{outputs_dir}/*/*fire_events.csv"
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} fire events files:")
    
    for file_path in files:
        try:
            # Extract station ID from directory name
            station_id = Path(file_path).parent.name
            print(f"  Loading {station_id}: {Path(file_path).name}")
            
            df = pd.read_csv(file_path)
            df['station'] = station_id
            
            # Convert time columns to datetime
            df['start'] = pd.to_datetime(df['start'])
            df['end'] = pd.to_datetime(df['end'])
            
            fire_events.append(df)
            
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    
    if not fire_events:
        print("No fire events found!")
        return pd.DataFrame()
    
    # Combine all events
    all_events = pd.concat(fire_events, ignore_index=True)
    print(f"Loaded {len(all_events)} total fire events from {len(files)} stations")
    
    return all_events


def get_station_color(station_id: str, current_time: datetime, events_df: pd.DataFrame, stations_list: List[str]) -> Tuple[float, str]:
    """Get station color based on fire events intensity."""
    if events_df.empty:
        return 0.0, '#1f77b4'
    
    # Find events for this station around current time
    station_events = events_df[
        (events_df['station'] == station_id) &
        (events_df['start'] <= current_time) &
        (events_df['end'] >= current_time)
    ]
    
    if station_events.empty:
        return 0.0, '#1f77b4'
    
    # Calculate intensity based on score and duration
    total_intensity = 0
    for _, event in station_events.iterrows():
        # Use score if available, otherwise use 1.0
        score = event.get('score', 1.0)
        duration_hours = (event['end'] - event['start']).total_seconds() / 3600
        intensity = score * min(duration_hours / 2.0, 1.0)  # Normalize by 2 hours
        total_intensity += intensity
    
    # Normalize intensity to 0-1 range
    intensity = min(total_intensity / 3.0, 1.0)
    
    # Color mapping: blue (no fire) to red (fire)
    if intensity == 0:
        return intensity, '#1f77b4'
    else:
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


def create_static_map(events_df: pd.DataFrame, locations_df: pd.DataFrame, output_dir: Path, map_style: str = "admin", tile_zoom: int = 13, offline: bool = False) -> None:
    """Create a static map showing all fire events."""
    fig = plt.figure(figsize=(24, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    add_background_map(ax, map_style, tile_zoom, offline)
    
    min_lon, max_lon, min_lat, max_lat = calculate_map_extent(locations_df)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    stations_list = locations_df['StationID'].tolist()
    
    if not events_df.empty:
        mid_time = events_df['start'].min() + (events_df['end'].max() - events_df['start'].min()) / 2
        
        for _, station in locations_df.iterrows():
            color_val, color_hex = get_station_color(station['StationID'], mid_time, events_df, stations_list)
            
            ax.scatter(station['lon'], station['lat'], c=color_hex, s=100, alpha=0.8, 
                      edgecolors='black', linewidth=1, transform=ccrs.PlateCarree())
            
            ax.text(station['lon'] + 0.001, station['lat'] + 0.001, station['StationID'], 
                   fontsize=6, transform=ccrs.PlateCarree())
    else:
        # No events, show all stations in blue
        for _, station in locations_df.iterrows():
            ax.scatter(station['lon'], station['lat'], c='#1f77b4', s=100, alpha=0.8, 
                      edgecolors='black', linewidth=1, transform=ccrs.PlateCarree())
            
            ax.text(station['lon'] + 0.001, station['lat'] + 0.001, station['StationID'], 
                   fontsize=6, transform=ccrs.PlateCarree())
    
    ax.set_title('Fire Events by Single-Station-Method', fontsize=14, fontweight='bold')
    ax.gridlines(draw_labels=True, alpha=0.5, xlocs=[127.46, 127.48, 127.50, 127.52], ylocs=[36.30, 36.32, 36.34, 36.36, 36.38])
    
    legend_elements = [
        plt.scatter([], [], c='#1f77b4', s=100, label='No Fire'),
        plt.scatter([], [], c='#d62728', s=100, label='Fire')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / f"outputs_fire_events_map_static.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved static map: {output_path}")


def create_timeline_visualization(events_df: pd.DataFrame, output_dir: Path) -> None:
    """Create a timeline visualization of fire events."""
    if events_df.empty:
        print("No events to visualize in timeline")
        return
    
    # Calculate the actual time span of events
    time_span_hours = (events_df['end'].max() - events_df['start'].min()).total_seconds() / 3600
    
    # Adjust figure size based on time span - make it more compact
    fig_width = max(12, min(24, time_span_hours * 0.5))
    fig = plt.figure(figsize=(fig_width, 8))
    
    # Create timeline for each station
    stations = events_df['station'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(stations)))
    
    for i, station in enumerate(stations):
        station_events = events_df[events_df['station'] == station].copy()
        station_events = station_events.sort_values('start')
        
        for _, event in station_events.iterrows():
            # Calculate relative positions
            start_offset = (event['start'] - events_df['start'].min()).total_seconds() / 3600
            duration = (event['end'] - event['start']).total_seconds() / 3600
            
            # Plot horizontal bar for this event
            plt.barh(i, duration, left=start_offset, height=0.8, 
                    color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add event label
            plt.text(start_offset + duration/2, i, 
                    f"{event.get('score', 1.0):.1f}", 
                    ha='center', va='center', fontsize=8, fontweight='bold')
    
    plt.yticks(range(len(stations)), stations)
    plt.xlabel('Time (hours from start)')
    plt.ylabel('Station')
    plt.title('Outputs Fire Events Timeline by Station', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add time axis labels
    start_time = events_df['start'].min()
    time_points = pd.date_range(start_time, events_df['end'].max(), freq='24H')
    time_labels = [(t - start_time).total_seconds() / 3600 for t in time_points]
    plt.xticks(time_labels, [t.strftime('%m-%d') for t in time_points], rotation=45)
    
    plt.tight_layout()
    output_path = output_dir / f"outputs_fire_events_timeline.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved timeline: {output_path}")


def create_animated_map(events_df: pd.DataFrame, locations_df: pd.DataFrame, output_dir: Path, 
                       map_style: str = "admin", tile_zoom: int = 13, fps: int = 2, 
                       time_step_minutes: int = 30, offline: bool = False) -> None:
    """Create an animated map showing fire events over time."""
    if events_df.empty:
        print("No events to animate")
        return
    
    # Create time range
    start_time = events_df['start'].min() - timedelta(hours=1)
    end_time = events_df['end'].max() + timedelta(hours=1)
    time_range = pd.date_range(start_time, end_time, freq=f'{time_step_minutes}min')
    
    fig = plt.figure(figsize=(24, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add background map layers
    add_background_map(ax, map_style, tile_zoom, offline)
    
    # Calculate and set map extent based on station locations
    min_lon, max_lon, min_lat, max_lat = calculate_map_extent(locations_df)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    stations_list = locations_df['StationID'].tolist()
    
    # Initialize scatter plot
    scatter = ax.scatter([], [], s=100, alpha=0.8, edgecolors='black', linewidth=1,
                        transform=ccrs.PlateCarree())
    
    # Add station labels
    for _, station in locations_df.iterrows():
        ax.text(station['lon'] + 0.001, station['lat'] + 0.001, 
               station['StationID'], fontsize=6, 
               transform=ccrs.PlateCarree())
    
    ax.set_title('Fire Events by Single-Station-Method - Dynamic Status', fontsize=14, fontweight='bold')
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
            color_val, color_hex = get_station_color(station['StationID'], current_time, 
                                                   events_df, stations_list)
            colors.append(color_hex)
            sizes.append(100 + color_val * 50)  # Size varies with intensity
            lons.append(station['lon'])
            lats.append(station['lat'])
        
        scatter.set_offsets(np.column_stack([lons, lats]))
        scatter.set_color(colors)
        scatter.set_sizes(sizes)
        
        ax.set_title(f'Fire Events by Single-Station-Method - {current_time.strftime("%Y-%m-%d %H:%M")}', 
                    fontsize=14, fontweight='bold')
        
        return scatter,
    
    anim = animation.FuncAnimation(fig, animate, frames=len(time_range), 
                                  interval=1000//fps, blit=False, repeat=True)
    
    output_path = output_dir / f"outputs_fire_events_map_animated.gif"
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()
    print(f"Saved animated map: {output_path}")


def main():
    args = parse_args()
    
    print("🔥 Outputs Fire Events Map Visualizer")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading station locations...")
    locations_df = load_station_locations(args.location_csv, args.exclude_distant)
    print(f"Loaded {len(locations_df)} active stations")
    
    print("Loading fire events...")
    events_df = load_outputs_fire_events(args.outputs_dir)
    if not events_df.empty:
        print(f"Loaded {len(events_df)} fire events")
    
    # Create visualizations
    print(f"Creating static map with {args.map_style} background (zoom={args.tile_zoom})...")
    create_static_map(events_df, locations_df, output_dir, args.map_style, args.tile_zoom, args.offline)
    
    print("Creating timeline visualization...")
    create_timeline_visualization(events_df, output_dir)
    
    if args.animation:
        print("Creating animated map...")
        create_animated_map(events_df, locations_df, output_dir, args.map_style, 
                           args.tile_zoom, args.fps, args.time_step_minutes, args.offline)
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()
