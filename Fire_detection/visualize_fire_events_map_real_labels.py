#!/usr/bin/env python3
"""
Visualize fire events on a map (restricted to 9/17–9/18 KST) and mark real-fire events
with a fire symbol at involved station locations.

- Reuses styling from visualize_fire_events_map.py
- Filters aggregated events to the date range
- Marks real-fire time windows and overlays a fire emoji (🔥) at stations involved
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
    parser = argparse.ArgumentParser(description="Visualize fire events with real-fire labels (9/11–9/18)")
    parser.add_argument("--location_csv", type=str, default="data/Location.csv")
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--map_style", type=str, default="admin", choices=["admin", "terrain", "satellite", "osm", "google", "google_sat", "google_terrain", "google_road", "esri"])
    parser.add_argument("--tile_zoom", type=int, default=13)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--animation", action="store_true", help="Create animated map")
    parser.add_argument("--fps", type=int, default=2, help="Animation frames per second")
    parser.add_argument("--time_step_minutes", type=int, default=30, help="Time step for animation in minutes")
    return parser.parse_args()


def load_station_locations(location_csv: str, exclude_distant: bool = True) -> pd.DataFrame:
    df = pd.read_csv(location_csv)
    coords = df['Location'].str.extract(r'([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)')
    df['lat'] = pd.to_numeric(coords[0], errors='coerce')
    df['lon'] = pd.to_numeric(coords[1], errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])
    df = df[df['Status'].str.strip().str.lower() == 'active']
    if exclude_distant:
        df = df[df['StationID'] != 'aa-87-15-02']
    return df[['StationID', 'lat', 'lon']].copy()


def load_station_events(locations_df: pd.DataFrame) -> pd.DataFrame:
    """Load fire events from individual station CSV files"""
    all_events = []
    
    for _, station in locations_df.iterrows():
        station_id = station['StationID']
        
        # Try to find fire events file for this station
        event_files = [
            f"outputs/{station_id}/{station_id}_pm_fire_events.csv",
            f"outputs/{station_id}_pm_fire_events.csv"
        ]
        
        for event_file in event_files:
            try:
                import os
                if os.path.exists(event_file):
                    events = pd.read_csv(event_file)
                    if not events.empty:
                        events['station_id'] = station_id
                        events['start'] = pd.to_datetime(events['start'], utc=True)
                        events['end'] = pd.to_datetime(events['end'], utc=True)
                        all_events.append(events)
                        print(f"Loaded {len(events)} events for station {station_id}")
                        break
            except Exception as e:
                print(f"Could not load events for {station_id}: {e}")
                continue
    
    if all_events:
        combined_events = pd.concat(all_events, ignore_index=True)
        print(f"Total events loaded: {len(combined_events)}")
        return combined_events
    else:
        print("No station events found")
        return pd.DataFrame()


def kst_to_utc(ts_str: str) -> pd.Timestamp:
    ts_kst = pd.to_datetime(ts_str)
    ts_utc = (ts_kst - timedelta(hours=9)).tz_localize('UTC') if ts_kst.tz is None else ts_kst.tz_convert('UTC')
    return ts_utc


def get_real_fire_windows_utc() -> List[Dict[str, pd.Timestamp]]:
    # Provided real-fire windows (KST) - extended by 1 hour on each side for visibility
    real_kst = [
        {"start": "2025-09-11 18:00:00", "end": "2025-09-11 20:25:00", "location": "Dae-Cheong Lake"},  # Extended
        {"start": "2025-09-12 07:15:00", "end": "2025-09-12 09:45:00", "location": "Dae-Cheong Lake"},  # Extended
        {"start": "2025-09-16 18:10:00", "end": "2025-09-16 21:00:00", "location": "Dae-Cheong Lake"},  # Extended
        {"start": "2025-09-17 08:20:00", "end": "2025-09-17 11:25:00", "location": "Dae-Cheong Lake"},  # Extended
        {"start": "2025-09-11 18:40:00", "end": "2025-09-11 21:00:00", "location": "Cherry Blossom Road"},  # Extended
        {"start": "2025-09-16 17:00:00", "end": "2025-09-16 19:30:00", "location": "Cherry Blossom Road"},  # Extended
    ]
    out = []
    for win in real_kst:
        out.append({
            "start": kst_to_utc(win["start"]),
            "end": kst_to_utc(win["end"]),
            "location": win["location"],
        })
    return out


def filter_events_by_timerange(df: pd.DataFrame, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    # Keep events overlapping [start_utc, end_utc]
    mask = (df['start'] <= end_utc) & (df['end'] >= start_utc)
    return df.loc[mask].copy()


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
            ax.add_feature(cfeature.BORDERS, linewidth=1.5)
    except Exception:
        ax.add_feature(cfeature.COASTLINE, linewidth=2)
        ax.add_feature(cfeature.BORDERS, linewidth=1.5)
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


def create_static_map_for_range(events_df: pd.DataFrame, locations_df: pd.DataFrame,
                               real_windows: List[Dict[str, pd.Timestamp]],
                               start_utc: pd.Timestamp, end_utc: pd.Timestamp,
                               output_dir: Path, map_style: str = "admin",
                               tile_zoom: int = 13, offline: bool = False) -> None:
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    add_background_map(ax, map_style, tile_zoom, offline)

    min_lon, max_lon, min_lat, max_lat = calculate_map_extent(locations_df)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # Filter station events to date range
    df_range = filter_events_by_timerange(events_df, start_utc, end_utc)
    
    # Determine which stations detected fire events during the time range
    stations_with_detections = set()
    for _, row in df_range.iterrows():
        stations_with_detections.add(row['station_id'])
    
    print(f"Stations with fire detections: {sorted(stations_with_detections)}")
    
    # Scatter all active stations with colors based on fire detection
    for _, st in locations_df.iterrows():
        if st['StationID'] in stations_with_detections:
            # Red for stations that detected fire
            color = '#d62728'
            edge_color = 'darkred'
        else:
            # Blue for stations with no fire detection
            color = '#1f77b4'
            edge_color = 'black'
        
        ax.scatter(st['lon'], st['lat'], c=color, s=80, alpha=0.9,
                   edgecolors=edge_color, linewidth=1, transform=ccrs.PlateCarree())
        ax.text(st['lon'] + 0.001, st['lat'] + 0.001, st['StationID'], fontsize=7,
                transform=ccrs.PlateCarree())

    # Build set of stations involved during the real-fire windows within the range
    stations_to_mark = set()
    real_fire_locations = []
    
    for win in real_windows:
        # Only consider windows that intersect the map time range
        if (win['start'] <= end_utc) and (win['end'] >= start_utc):
            real_fire_locations.append(win['location'])
            # Find station events overlapping this real-fire window
            overlap_mask = (df_range['start'] <= win['end']) & (df_range['end'] >= win['start'])
            overlapping = df_range.loc[overlap_mask]
            print(f"Real fire at {win['location']} ({win['start']} - {win['end']})")
            print(f"  Overlapping detected events: {len(overlapping)}")
            for _, row in overlapping.iterrows():
                print(f"    Station {row['station_id']}: {row['start']} - {row['end']}")
                stations_to_mark.add(row['station_id'])
    
    print(f"\nStations to mark with fire symbols: {sorted(stations_to_mark)}")
    print(f"Real fire locations in map range: {real_fire_locations}")

    # Place real fire event markers as triangles at approximate locations
    # Define approximate coordinates for real fire locations
    # Note: coordinates are (lon, lat) where lon ~ 127.5, lat ~ 36.35
    real_fire_coords = {
        'Dae-Cheong Lake': (127.495, 36.355),  # Near station area
        'Cherry Blossom Road': (127.500, 36.360)  # Near station area
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
    ax.set_title(f"Fire Events Map: Station Detections vs Real Fire Events\nTime Window: {start_kst.strftime('%Y-%m-%d %H:%M KST')} – {end_kst.strftime('%Y-%m-%d %H:%M KST')}",
                 fontsize=12, fontweight='bold')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, label='Station (No Fire Detected)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=10, label='Station (Fire Detected)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=12, label='Real Fire Event Location')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'real_fire_events_map_917_918.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved static map with real-fire markers: {output_path}")


def get_station_color_at_time(station_id: str, timestamp: pd.Timestamp, events_df: pd.DataFrame) -> str:
    """Determine station color based on fire events at given timestamp"""
    station_events = events_df[
        (events_df['station_id'] == station_id) &
        (events_df['start'] <= timestamp) &
        (events_df['end'] >= timestamp)
    ]
    
    if station_events.empty:
        return '#1f77b4'  # Blue (no fire)
    else:
        return '#d62728'  # Red (fire detected)


def create_animated_map(events_df: pd.DataFrame, locations_df: pd.DataFrame,
                       real_windows: List[Dict[str, pd.Timestamp]],
                       start_utc: pd.Timestamp, end_utc: pd.Timestamp,
                       output_dir: Path, map_style: str = "admin",
                       tile_zoom: int = 13, offline: bool = False,
                       fps: int = 2, time_step_minutes: int = 30) -> None:
    """Create an animated map showing station color changes over time"""
    
    # Create time range for animation
    time_range = pd.date_range(start_utc, end_utc, freq=f'{time_step_minutes}min')
    
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    add_background_map(ax, map_style, tile_zoom, offline)
    
    min_lon, max_lon, min_lat, max_lat = calculate_map_extent(locations_df)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    # Filter events to our time range
    df_range = filter_events_by_timerange(events_df, start_utc, end_utc)
    
    # Initialize scatter plot for stations
    scatter = ax.scatter([], [], s=80, alpha=0.9, edgecolors='black', linewidth=1,
                        transform=ccrs.PlateCarree())
    
    # Add station labels (static)
    for _, st in locations_df.iterrows():
        ax.text(st['lon'] + 0.001, st['lat'] + 0.001, st['StationID'], fontsize=7,
                transform=ccrs.PlateCarree())
    
    # Initialize real fire markers (will be animated)
    real_fire_coords = {
        'Dae-Cheong Lake': (127.495, 36.355),  # Near station area
        'Cherry Blossom Road': (127.500, 36.360)  # Near station area
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
    
    # No text labels for real fire locations (cleaner look)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, label='Station (No Fire Detected)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=10, label='Station (Fire Detected)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=12, label='Real Fire Event Location')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
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
        
        for win in real_windows:
            if (win['start'] <= end_utc) and (win['end'] >= start_utc):
                # Check if current time is within this real fire window
                if (current_time >= win['start']) and (current_time <= win['end']):
                    if win['location'] in real_fire_coords:
                        lon, lat = real_fire_coords[win['location']]
                        fire_lons.append(lon)
                        fire_lats.append(lat)
                        fire_colors.append('orange')
        
        if fire_lons:
            real_fire_scatter.set_offsets(np.column_stack([fire_lons, fire_lats]))
            real_fire_scatter.set_color(fire_colors)
        else:
            real_fire_scatter.set_offsets(np.column_stack([[], []]))
        
        # Update title with current time
        start_kst = (current_time.tz_convert('Asia/Seoul') if current_time.tz else current_time.tz_localize('UTC').tz_convert('Asia/Seoul'))
        ax.set_title(f"Fire Events Map: Station Detections vs Real Fire Events\nTime: {start_kst.strftime('%Y-%m-%d %H:%M KST')}", 
                    fontsize=12, fontweight='bold')
        
        return scatter, real_fire_scatter,
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(time_range), 
                                 interval=1000//fps, blit=True, repeat=True)
    
    # Save as GIF
    output_path = output_dir / 'real_fire_events_map_917_918_animated.gif'
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()
    print(f"Saved animated map: {output_path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading station locations...")
    locations_df = load_station_locations(args.location_csv, exclude_distant=True)
    print(f"Loaded {len(locations_df)} active stations")

    print("Loading station events...")
    events_df = load_station_events(locations_df)
    print(f"Loaded {len(events_df)} station events")

    # Date range: 9/11–9/18 in KST => UTC: 2025-09-11 00:00 to 2025-09-18 15:00
    start_utc = pd.Timestamp('2025-09-11 00:00:00', tz='UTC')
    end_utc = pd.Timestamp('2025-09-18 15:00:00', tz='UTC')

    # Real-fire windows in UTC
    real_windows = get_real_fire_windows_utc()

    print("Creating static map...")
    create_static_map_for_range(events_df, locations_df, real_windows, start_utc, end_utc,
                                output_dir, args.map_style, args.tile_zoom, args.offline)

    if args.animation:
        print("Creating animated map...")
        create_animated_map(events_df, locations_df, real_windows, start_utc, end_utc,
                           output_dir, args.map_style, args.tile_zoom, args.offline,
                           args.fps, args.time_step_minutes)

    print("Done.")


if __name__ == "__main__":
    main()
