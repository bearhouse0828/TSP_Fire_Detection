#!/usr/bin/env python3
"""
Example usage of the Fire Detection Data Loader

This script demonstrates how to use the FireDetectionDataLoader class
to load and visualize fire detection sensor data.
"""

from fire_detection_data_loader import FireDetectionDataLoader
import matplotlib.pyplot as plt

def main():
    """Main example function"""
    print("Fire Detection Data Analysis Example")
    print("=" * 40)
    
    # Initialize the data loader
    loader = FireDetectionDataLoader('data')
    
    # Load all sensor data
    print("Loading sensor data...")
    data = loader.load_all_data()
    
    if not data:
        print("No data loaded. Please check your data folder.")
        return
    
    # Display basic information about the loaded data
    print("\nLoaded Data Summary:")
    print("-" * 25)
    summary = loader.get_data_summary()
    
    for sensor_name, info in summary.items():
        print(f"\n{sensor_name}:")
        print(f"  Data shape: {info['shape']}")
        print(f"  Time range: {info['time_range'][0]} to {info['time_range'][1]}")
        print(f"  Missing data: {info['missing_data_percentage']:.2f}%")
        print(f"  Stations: {len(info['columns'])}")
    
    # Plot specific sensor data
    print("\nGenerating visualizations...")
    
    # Plot PM1 data (particulate matter)
    if 'PM1' in data:
        print("Plotting PM1 (Particulate Matter) data...")
        loader.plot_sensor_data('PM1')
    
    # Plot Temperature data
    if 'Temperature' in data:
        print("Plotting Temperature data...")
        loader.plot_sensor_data('Temperature')
    
    # Plot Relative Humidity data
    if 'Relative_Humidity' in data:
        print("Plotting Relative Humidity data...")
        loader.plot_sensor_data('Relative_Humidity')
    
    # Plot overview of all sensors
    print("Plotting overview of all sensors...")
    loader.plot_all_sensors_overview()
    
    # Plot correlation matrix
    print("Plotting correlation matrix...")
    loader.plot_correlation_matrix()
    
    # Plot station comparison for PM1
    if 'PM1' in data:
        print("Plotting station comparison for PM1...")
        loader.plot_station_comparison('PM1')
    
    print("\nAnalysis complete!")
    print("You can now use this data to develop your fire detection algorithm.")

if __name__ == "__main__":
    main()
