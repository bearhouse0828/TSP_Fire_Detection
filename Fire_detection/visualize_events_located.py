#!/usr/bin/env python3
"""
Visualize events from events_located.csv on a map with dynamic station colors.

This script is adapted to work with the events_located.csv format which has:
- t_start, t_end (instead of group_start, group_end)
- stations_involved (instead of stations)
- event_id, score, level, x0, y0 (location coordinates)
- No vote_count column

Usage:
    python visualize_events_located.py
    python visualize_events_located.py --animation --fps 2
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
    parser = argparse.ArgumentParser(description="Visualize events from events_located.csv on station map")
    parser.add_argument("--events_csv", type=str, default="outputs/events_located.csv")
    parser.add_argument("--location_csv", type=str, default="data/Location.csv")
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--animation", action="store_true")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--time_step_minutes", type=int, default=30)
    parser.add_argument("--exclude_distant", action="store_true", default=True)
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


def load_fire_events(events_csv: str) -> pd.DataFrame:
    """Load fire events from events_located.csv format."""
    df = pd.read_csv(events_csv)
    df['t_start'] = pd.to_datetime(df['t_start'], utc=True)
    df['t_end'] = pd.to_datetime(df['t_end'], utc=True)
    
    # Create a vote_count column based on number of stations involved
    df['vote_count'] = df['stations_involved'].str.count(',') + 1
    
    # Rename columns to match expected format
    df = df.rename(columns={
        't_start': 'group_start',
        't_end': 'group_end', 
        'stations_involved': 'stations'
    })
    
    # Add a dummy first_station_start column
    df['first_station_start'] = df['group_start']
    
    return df


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



def get_station_color(station_id: str, timestamp: pd.Timestamp, 
                     events_df: pd.DataFrame, stations_list: List[str]) -> Tuple[float, str]:
    """
    Determine station color based on fire events at given timestamp.
    For events_located.csv, we use the score and level to determine intensity.
    """
    # Find events involving this station at this time
    station_events = events_df[
        events_df['stations'].str.contains(station_id, na=False) &
        (events_df['group_start'] <= timestamp) &
        (events_df['group_end'] >= timestamp)
    ]
    
    if station_events.empty:
        return 0.0, '#1f77b4'  # Blue (no fire)
    
    # Use score and level to determine intensity
    max_score = station_events['score'].max()
    max_level = station_events['level'].max()
    
    # Convert level to numeric weight
    level_weight = 0.3 if 'Level-1' in max_level else 0.7 if 'Level-2' in max_level else 1.0
    
    # Combine score and level
    intensity = min(1.0, max_score * level_weight)
    
    # Add recency bonus for events that just started
    recent_bonus = 0.0
    for _, event in station_events.iterrows():
        time_since_start = (timestamp - event['group_start']).total_seconds() / 3600
        if time_since_start < 1:  # Within first hour
            recent_bonus += 0.2 * (1 - time_since_start)
    
    intensity = min(1.0, intensity + recent_bonus)
    
    # Convert to color
    if intensity < 0.3:
        return intensity, '#1f77b4'  # Blue
    elif intensity < 0.7:
        return intensity, '#ff7f0e'  # Orange
    else:
        return intensity, '#d62728'  # Red


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


def calculate_map_extent(locations_df: pd.DataFrame, events_df: pd.DataFrame = None, buffer: float = 0.01) -> Tuple[float, float, float, float]:
    """Calculate map extent based on station locations only (like original script)."""
    if locations_df.empty:
        return 127.35, 127.55, 36.30, 36.40
    
    # Focus only on station locations to keep map small and fast
    min_lon = locations_df['lon'].min() - buffer
    max_lon = locations_df['lon'].max() + buffer
    min_lat = locations_df['lat'].min() - buffer
    max_lat = locations_df['lat'].max() + buffer
    
    return min_lon, max_lon, min_lat, max_lat


def create_static_map(events_df: pd.DataFrame, locations_df: pd.DataFrame, output_dir: Path, map_style: str = "admin", tile_zoom: int = 13, offline: bool = False) -> None:
    """Create a static map showing all fire events."""
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    add_background_map(ax, map_style, tile_zoom, offline)
    
    min_lon, max_lon, min_lat, max_lat = calculate_map_extent(locations_df, events_df, buffer=0.005)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    print(f"Map extent: [{min_lon:.3f}, {max_lon:.3f}, {min_lat:.3f}, {max_lat:.3f}]")
    
    stations_list = locations_df['StationID'].tolist()
    
    # Plot stations
    for _, station in locations_df.iterrows():
        if not events_df.empty:
            mid_time = events_df['group_start'].min() + (events_df['group_end'].max() - events_df['group_start'].min()) / 2
        else:
            mid_time = pd.Timestamp.now(tz='UTC')
            
        color_val, color_hex = get_station_color(station['StationID'], mid_time, events_df, stations_list)
        
        ax.scatter(station['lon'], station['lat'], c=color_hex, s=100, alpha=0.8, 
                  edgecolors='black', linewidth=1, transform=ccrs.PlateCarree())
        
        ax.text(station['lon'] + 0.001, station['lat'] + 0.001, station['StationID'], 
               fontsize=8, transform=ccrs.PlateCarree())
    
    # Plot event locations if available (sample only to avoid overcrowding)
    if not events_df.empty:
        sample_events = events_df.head(50)  # Limit to first 50 events
        for _, event in sample_events.iterrows():
            # Color based on score and level
            score = event['score']
            level = event['level']
            if 'Level-2' in level:
                color = '#d62728'  # Red
            elif 'Level-1' in level:
                color = '#ff7f0e'  # Orange
            else:
                color = '#1f77b4'  # Blue
            
            # Note: x0 and y0 are swapped in events_located.csv (x0=lat, y0=lon)
            ax.scatter(event['y0'], event['x0'], c=color, s=50, alpha=0.6, 
                      marker='x', transform=ccrs.PlateCarree())
    
    ax.set_title('Fire Detection Stations and Events - Overview', fontsize=14, fontweight='bold')
    ax.gridlines(draw_labels=True, alpha=0.5)
    
    legend_elements = [
        plt.scatter([], [], c='#1f77b4', s=100, label='Normal (No Fire)'),
        plt.scatter([], [], c='#ff7f0e', s=100, label='Low Fire Risk'),
        plt.scatter([], [], c='#d62728', s=100, label='High Fire Risk'),
        plt.scatter([], [], c='#d62728', s=50, marker='x', label='Event Locations')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'events_located_map_static.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved static map: {output_dir / 'events_located_map_static.png'}")


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
        score = event['score']
        level = event['level']
        
        if 'Level-2' in level:
            color = '#d62728'
        elif 'Level-1' in level:
            color = '#ff7f0e'
        else:
            color = '#1f77b4'
        
        # Calculate relative positions
        start_offset = (event['group_start'] - events_df['group_start'].min()).total_seconds() / 3600
        duration = (event['group_end'] - event['group_start']).total_seconds() / 3600
        
        ax1.barh(i, duration, left=start_offset, height=0.8, color=color, alpha=0.7)
        
        # Position text at the start of the event
        ax1.text(start_offset, i, f"Event {event['event_id']} (Score: {score:.2f}, {level})",
                va='center', ha='left', fontsize=8)
    
    ax1.set_xlabel('Time (hours from first event)')
    ax1.set_ylabel('Events')
    ax1.set_title('Fire Event Timeline (events_located.csv)')
    ax1.grid(True, alpha=0.3)
    
    # Set x-axis limits to focus on the actual data range
    ax1.set_xlim(0, time_span_hours)
    
    # Station participation heatmap
    station_participation = {}
    for station in stations_list:
        participation = []
        for _, event in events_df.iterrows():
            if station in event['stations']:
                participation.append(1)
            else:
                participation.append(0)
        station_participation[station] = participation
    
    if station_participation:
        participation_matrix = np.array(list(station_participation.values()))
        
        im = ax2.imshow(participation_matrix, cmap='RdYlBu_r', aspect='auto')
        ax2.set_xticks(range(len(events_df)))
        ax2.set_xticklabels([f"E{row['event_id']}" for _, row in events_df.iterrows()], rotation=45)
        ax2.set_yticks(range(len(stations_list)))
        ax2.set_yticklabels(stations_list)
        ax2.set_xlabel('Events')
        ax2.set_ylabel('Stations')
        ax2.set_title('Station Participation in Fire Events')
        
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Participation (1=Yes, 0=No)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'events_located_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved timeline: {output_dir / 'events_located_timeline.png'}")


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
    
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add background map layers
    add_background_map(ax, map_style, tile_zoom, offline)
    
    # Calculate and set map extent based on station locations and events
    min_lon, max_lon, min_lat, max_lat = calculate_map_extent(locations_df, events_df, buffer=0.005)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    print(f"Animated map extent: [{min_lon:.3f}, {max_lon:.3f}, {min_lat:.3f}, {max_lat:.3f}]")
    
    stations_list = locations_df['StationID'].tolist()
    
    # Initialize scatter plot
    scatter = ax.scatter([], [], s=100, alpha=0.8, edgecolors='black', linewidth=1,
                        transform=ccrs.PlateCarree())
    
    # Add station labels
    for _, station in locations_df.iterrows():
        ax.text(station['lon'] + 0.001, station['lat'] + 0.001, 
               station['StationID'], fontsize=8, 
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
    output_path = output_dir / 'events_located_map_animated.gif'
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
    
    print("Loading fire events from events_located.csv...")
    events_df = load_fire_events(args.events_csv)
    print(f"Loaded {len(events_df)} events")
    
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
