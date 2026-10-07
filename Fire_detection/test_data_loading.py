#!/usr/bin/env python3
"""
Test script for data loading functionality
"""

from fire_detection_data_loader import FireDetectionDataLoader

def test_data_loading():
    """Test basic data loading functionality"""
    print("Testing Fire Detection Data Loading...")
    print("=" * 40)
    
    # Initialize data loader
    loader = FireDetectionDataLoader('data')
    
    # Load location data
    print("1. Loading location data...")
    location_data = loader.load_location_data()
    if location_data is not None:
        print(f"   ✓ Loaded {len(location_data)} stations")
        print(f"   Station IDs: {loader.station_ids}")
    else:
        print("   ✗ Failed to load location data")
    
    # Test loading individual sensors
    print("\n2. Testing individual sensor loading...")
    test_sensors = ['PM1', 'T', 'RH', 'S1']
    
    for sensor in test_sensors:
        print(f"   Loading {sensor}...")
        data = loader.load_sensor_data(sensor)
        if data is not None:
            print(f"   ✓ {sensor}: {data.shape[0]} rows, {data.shape[1]} columns")
        else:
            print(f"   ✗ Failed to load {sensor}")
    
    # Load all data
    print("\n3. Loading all sensor data...")
    all_data = loader.load_all_data()
    
    if all_data:
        print(f"   ✓ Successfully loaded {len(all_data)} sensor datasets")
        
        # Print summary
        summary = loader.get_data_summary()
        print("\n4. Data Summary:")
        print("-" * 20)
        for sensor_name, info in summary.items():
            print(f"   {sensor_name}: {info['shape']} shape, {info['missing_data_percentage']:.1f}% missing")
    else:
        print("   ✗ Failed to load sensor data")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_data_loading()
