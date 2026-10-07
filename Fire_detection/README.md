# Fire Detection Data Analysis System

This project provides tools for loading, visualizing, and analyzing fire detection sensor data from multiple monitoring stations.

## Data Structure

The system processes data from the following sensors:
- **PM1**: Particulate matter (µg/m³)
- **PM4**: Particulate matter (µg/m³) 
- **Temperature (T)**: Temperature readings (°C)
- **Relative Humidity (RH)**: Humidity percentage
- **Pressure (P)**: Atmospheric pressure
- **SPH Sensors (S1-S12)**: Specialized fire detection sensors

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from fire_detection_data_loader import FireDetectionDataLoader

# Initialize the data loader
loader = FireDetectionDataLoader('data')

# Load all sensor data
data = loader.load_all_data()

# Plot specific sensor data
loader.plot_sensor_data('PM1')
loader.plot_sensor_data('Temperature')
```

### Running the Example

```bash
python example_usage.py
```

## Features

### Data Loading
- Automatic loading of all CSV files in the data folder
- Handles missing data and data type conversion
- Loads location information for sensor stations

### Visualization
- **Individual Sensor Plots**: Time series plots for specific sensors
- **Overview Plots**: Multi-panel overview of all sensors
- **Correlation Matrix**: Heatmap showing correlations between sensors
- **Station Comparison**: Side-by-side comparison of data from different stations

### Data Analysis
- Summary statistics for all loaded data
- Missing data percentage calculation
- Time range analysis
- Station information

## File Structure

```
Fire_detection/
├── data/                          # Sensor data files
│   ├── Location.csv              # Station location information
│   ├── PM1.csv                   # PM1 sensor data
│   ├── PM4.csv                   # PM4 sensor data
│   ├── T.csv                     # Temperature data
│   ├── RH.csv                    # Relative humidity data
│   ├── P.csv                     # Pressure data
│   └── S1.csv - S12.csv          # SPH sensor data
├── fire_detection_data_loader.py  # Main data loader class
├── example_usage.py              # Example usage script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Data Format

Each sensor CSV file contains:
- `timestamp_utc`: UTC timestamp
- Station ID columns: Data from different monitoring stations
- Missing values are represented as empty cells

## Next Steps for Fire Detection Algorithm

This data loading system provides the foundation for developing a fire detection algorithm. Consider:

1. **Feature Engineering**: Create derived features from raw sensor data
2. **Anomaly Detection**: Identify unusual patterns in sensor readings
3. **Multi-sensor Fusion**: Combine data from different sensor types
4. **Machine Learning**: Train models to classify fire vs. non-fire conditions
5. **Real-time Processing**: Implement real-time data processing pipeline

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- matplotlib: Plotting and visualization
- seaborn: Statistical data visualization
- scikit-learn: Machine learning tools
