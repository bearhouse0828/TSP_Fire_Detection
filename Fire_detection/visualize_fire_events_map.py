#!/usr/bin/env python3
"""
Visualize fire events on a map with dynamic station colors.
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
    parser = argparse.ArgumentParser(description="Visualize fire events on station map")
    parser.add_argument("--events_csv", type=str, default="outputs/aggregated/nearby_group_events_voted.csv")
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


def load_fire_events(events_csv: str) -> pd.DataFrame:
    """Load fire events from CSV."""
    df = pd.read_csv(events_csv)
    df['group_start'] = pd.to_datetime(df['group_start'], utc=True)
    df['group_end'] = pd.to_datetime(df['group_end'], utc=True)
    df['first_station_start'] = pd.to_datetime(df['first_station_start'], utc=True)
    return df


def get_station_color(station_id: str, timestamp: pd.Timestamp, events_df: pd.DataFrame, stations_list: List[str]) -> Tuple[float, str]:
    """Determine station color based on fire events at given timestamp."""
    station_events = events_df[
        events_df['stations'].str.contains(station_id, na=False) &
        (events_df['group_start'] <= timestamp) &
        (events_df['group_end'] >= timestamp)
    ]
    
    if station_events.empty:
        return 0.0, '#1f77b4'  # Blue (no fire)
    
    max_votes = station_events['vote_count'].max()
    vote_weight = max_votes / len(stations_list)
    
    recent_bonus = 0.0
    for _, event in station_events.iterrows():
        time_since_start = (timestamp - event['group_start']).total_seconds() / 3600
        if time_since_start < 1:
            recent_bonus += 0.3 * (1 - time_since_start)
    
    intensity = min(1.0, vote_weight + recent_bonus)
    
    if intensity < 0.3:
        return intensity, '#1f77b4'
    elif intensity < 0.7:
        return intensity, '#ff7f0e'
    else:
        return intensity, '#d62728'


def add_background_map(ax, map_style: str = "admin", tile_zoom: int = 13, offline: bool = False) -> None:
    """Add background map layers based on style."""
    ax.add_feature(cfeature.LAND, alpha=0.4, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, alpha=0.4, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5)
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    
    if offline:
        print("Using offline mode - loading bundled stock image only...")
        try:
            ax.stock_img()
        except Exception as e:
            print(f"Failed to load stock image background: {e}")
        return
    
    # Try different tile sources
    try:
        if map_style == "osm":
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
            print("Loaded ESRI tiles successfully")
        elif map_style == "terrain":
            terrain = cimgt.Stamen('terrain')
            ax.add_image(terrain, tile_zoom)
            print("Loaded terrain tiles successfully")
        elif map_style == "satellite":
            satellite = cimgt.Stamen('satellite')
            ax.add_image(satellite, tile_zoom)
            print("Loaded satellite tiles successfully")
        else:  # admin
            print("Using basic administrative features...")
            ax.add_feature(cfeature.COASTLINE, linewidth=2)
            ax.add_feature(cfeature.BORDERS, linewidth=1.5)
    except Exception as e:
        print(f"Failed to load {map_style} tiles: {e}")
        print("Falling back to basic features...")
        ax.add_feature(cfeature.COASTLINE, linewidth=2)
        ax.add_feature(cfeature.BORDERS, linewidth=1.5)
    
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
        mid_time = events_df['group_start'].min() + (events_df['group_end'].max() - events_df['group_start'].min()) / 2
    else:
        mid_time = pd.Timestamp.now(tz='UTC')
    
    for _, station in locations_df.iterrows():
        color_val, color_hex = get_station_color(station['StationID'], mid_time, events_df, stations_list)
        
        ax.scatter(station['lon'], station['lat'], c=color_hex, s=100, alpha=0.8, 
                  edgecolors='black', linewidth=1, transform=ccrs.PlateCarree())
        
        ax.text(station['lon'] + 0.001, station['lat'] + 0.001, station['StationID'], 
               fontsize=6, transform=ccrs.PlateCarree())
    
    ax.set_title('Fire Detection Stations - Event Overview', fontsize=14, fontweight='bold')
    ax.gridlines(draw_labels=True, alpha=0.5)
    
    legend_elements = [
        plt.scatter([], [], c='#1f77b4', s=100, label='Normal (No Fire)'),
        plt.scatter([], [], c='#ff7f0e', s=100, label='Low Fire Risk'),
        plt.scatter([], [], c='#d62728', s=100, label='High Fire Risk')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fire_events_map_static.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved static map: {output_dir / 'fire_events_map_static.png'}")


def create_timeline_visualization(events_df: pd.DataFrame, locations_df: pd.DataFrame, output_dir: Path) -> None:
    """Create a timeline visualization of fire events."""
    if events_df.empty:
        print("No events to visualize in timeline")
        return
    
    # Calculate the actual time span of events
    time_span_hours = (events_df['group_end'].max() - events_df['group_start'].min()).total_seconds() / 3600
    
    # Adjust figure size based on time span - make it more compact
    fig_width = max(12, min(20, time_span_hours * 0.5))  # Scale width with time span
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, 8))
    
    stations_list = locations_df['StationID'].tolist()
    
    for i, (_, event) in enumerate(events_df.iterrows()):
        vote_count = event['vote_count']
        
        if vote_count >= 4:
            color = '#d62728'
        elif vote_count >= 2:
            color = '#ff7f0e'
        else:
            color = '#1f77b4'
        
        # Calculate relative positions
        start_offset = (event['group_start'] - events_df['group_start'].min()).total_seconds() / 3600
        duration = (event['group_end'] - event['group_start']).total_seconds() / 3600
        
        ax1.barh(i, duration, left=start_offset, height=0.8, color=color, alpha=0.7)
        
        # Position text at the start of the event
        ax1.text(start_offset, i, f"Group {event['group_id']} ({vote_count} votes)",
                va='center', ha='left', fontsize=8)
    
    ax1.set_xlabel('Time (hours from first event)')
    ax1.set_ylabel('Event Groups')
    ax1.set_title('Fire Event Timeline')
    ax1.grid(True, alpha=0.3)
    
    # Set x-axis limits to focus on the actual data range
    ax1.set_xlim(0, time_span_hours)
    
    station_participation = {}
    for station in stations_list:
        participation = []
        for _, event in events_df.iterrows():
            if station in event['stations']:
                participation.append(1)
            else:
                participation.append(0)
        station_participation[station] = participation
    
    participation_matrix = np.array(list(station_participation.values()))
    
    im = ax2.imshow(participation_matrix, cmap='RdYlBu_r', aspect='auto')
    ax2.set_xticks(range(len(events_df)))
    ax2.set_xticklabels([f"G{row['group_id']}" for _, row in events_df.iterrows()], rotation=45)
    ax2.set_yticks(range(len(stations_list)))
    ax2.set_yticklabels(stations_list)
    ax2.set_xlabel('Event Groups')
    ax2.set_ylabel('Stations')
    ax2.set_title('Station Participation in Fire Events')
    
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Participation (1=Yes, 0=No)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fire_events_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved timeline: {output_dir / 'fire_events_timeline.png'}")


def create_animated_map(events_df: pd.DataFrame, locations_df: pd.DataFrame,
                       output_dir: Path, fps: int = 2, time_step_minutes: int = 30, 
                       map_style: str = "admin", tile_zoom: int = 13, offline: bool = False) -> None:
    """Create an animated map showing color transitions over time."""
    if events_df.empty:
        print("No events to animate")
        return
    
    # Create time range
    start_time = events_df['group_start'].min() - timedelta(hours=1)
    end_time = events_df['group_end'].max() + timedelta(hours=1)
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
    
    ax.set_title('Fire Detection Stations - Dynamic Status', fontsize=14, fontweight='bold')
    ax.gridlines(draw_labels=True, alpha=0.5)
    
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
        
        ax.set_title(f'Fire Detection Stations - {current_time.strftime("%Y-%m-%d %H:%M UTC")}', 
                    fontsize=14, fontweight='bold')
        
        return scatter,
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(time_range), 
                                 interval=1000//fps, blit=True, repeat=True)
    
    # Save as GIF
    output_path = output_dir / 'fire_events_map_animated.gif'
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()
    print(f"Saved animated map: {output_path}")


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading station locations...")
    locations_df = load_station_locations(args.location_csv, args.exclude_distant)
    print(f"Loaded {len(locations_df)} active stations")
    
    print("Loading fire events...")
    events_df = load_fire_events(args.events_csv)
    print(f"Loaded {len(events_df)} event groups")
    
    if events_df.empty:
        print("No fire events found. Creating map with normal station colors only.")
    
    print(f"Creating static map with {args.map_style} background (zoom={args.tile_zoom})...")
    create_static_map(events_df, locations_df, output_dir, args.map_style, args.tile_zoom, args.offline)
    
    print("Creating timeline visualization...")
    create_timeline_visualization(events_df, locations_df, output_dir)
    
    if args.animation:
        print(f"Creating animated map with {args.map_style} background (zoom={args.tile_zoom})...")
        create_animated_map(events_df, locations_df, output_dir, 
                          args.fps, args.time_step_minutes, args.map_style, args.tile_zoom, args.offline)
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()
