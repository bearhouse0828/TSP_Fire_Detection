#!/usr/bin/env python3
"""
Test script for visualization functionality
"""

from fire_detection_data_loader import FireDetectionDataLoader
import matplotlib.pyplot as plt

def test_visualization():
    """Test visualization functionality"""
    print("Testing Fire Detection Data Visualization...")
    print("=" * 45)
    
    # Initialize data loader
    loader = FireDetectionDataLoader('data')
    
    # Load all data
    print("Loading data...")
    data = loader.load_all_data()
    
    if not data:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Loaded {len(data)} sensor datasets")
    
    # Test plotting individual sensors
    print("\nTesting individual sensor plotting...")
    
    # Plot PM1 data
    if 'PM1' in data:
        print("Plotting PM1 data...")
        try:
            loader.plot_sensor_data('PM1')
            print("✓ PM1 plot generated successfully")
        except Exception as e:
            print(f"✗ Error plotting PM1: {e}")
    
    # Plot Temperature data
    if 'Temperature' in data:
        print("Plotting Temperature data...")
        try:
            loader.plot_sensor_data('Temperature')
            print("✓ Temperature plot generated successfully")
        except Exception as e:
            print(f"✗ Error plotting Temperature: {e}")
    
    # Test overview plotting
    print("\nTesting overview plotting...")
    try:
        loader.plot_all_sensors_overview()
        print("✓ Overview plot generated successfully")
    except Exception as e:
        print(f"✗ Error generating overview plot: {e}")
    
    # Test correlation matrix
    print("\nTesting correlation matrix...")
    try:
        loader.plot_correlation_matrix()
        print("✓ Correlation matrix generated successfully")
    except Exception as e:
        print(f"✗ Error generating correlation matrix: {e}")
    
    print("\nVisualization test completed!")
    print("Note: Plots should have appeared in separate windows.")

if __name__ == "__main__":
    test_visualization()
