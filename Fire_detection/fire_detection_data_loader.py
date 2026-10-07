import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class FireDetectionDataLoader:
    """
    Fire Detection Data Loader and Visualizer
    
    This class handles loading and visualization of fire detection sensor data
    including PM1, PM4, Temperature, Relative Humidity, and SPH sensors (S1-S12)
    """
    
    def __init__(self, data_folder='data'):
        """
        Initialize the data loader
        
        Args:
            data_folder (str): Path to the data folder containing CSV files
        """
        self.data_folder = data_folder
        self.data = {}
        self.location_info = None
        self.station_ids = None
        
    def load_location_data(self):
        """Load location information for sensor stations"""
        location_file = os.path.join(self.data_folder, 'Location.csv')
        if os.path.exists(location_file):
            self.location_info = pd.read_csv(location_file)
            self.station_ids = self.location_info['StationID'].tolist()
            print(f"Loaded location data for {len(self.station_ids)} stations")
            return self.location_info
        else:
            print("Location.csv not found")
            return None
    
    def load_sensor_data(self, sensor_name, skip_rows=1):
        """
        Load data from a specific sensor CSV file
        
        Args:
            sensor_name (str): Name of the sensor (e.g., 'PM1', 'T', 'RH', 'S1')
            skip_rows (int): Number of rows to skip (default 1 for header)
        
        Returns:
            pd.DataFrame: Loaded sensor data
        """
        file_path = os.path.join(self.data_folder, f'{sensor_name}.csv')
        
        if not os.path.exists(file_path):
            print(f"File {file_path} not found")
            return None
        
        try:
            # Load data, skipping the metric comment row
            df = pd.read_csv(file_path, skiprows=skip_rows)
            
            # Convert timestamp to datetime
            df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
            
            # Set timestamp as index
            df.set_index('timestamp_utc', inplace=True)
            
            # Convert all columns to numeric, replacing non-numeric values with NaN
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"Loaded {sensor_name} data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            print(f"Error loading {sensor_name} data: {e}")
            return None
    
    def load_all_data(self):
        """Load all available sensor data"""
        print("Loading all sensor data...")
        
        # Load location data first
        self.load_location_data()
        
        # Define sensor types and their names
        sensors = {
            'PM1': 'PM1',
            'PM4': 'PM4', 
            'Temperature': 'T',
            'Relative_Humidity': 'RH',
            'Pressure': 'P'
        }
        
        # Load SPH sensors (S1-S12)
        for i in range(1, 13):
            sensors[f'SPH_S{i}'] = f'S{i}'
        
        # Load each sensor
        for sensor_type, file_name in sensors.items():
            df = self.load_sensor_data(file_name)
            if df is not None:
                self.data[sensor_type] = df
        
        print(f"Successfully loaded {len(self.data)} sensor datasets")
        return self.data
    
    def get_data_summary(self):
        """Get summary statistics for all loaded data"""
        summary = {}
        
        for sensor_name, df in self.data.items():
            summary[sensor_name] = {
                'shape': df.shape,
                'time_range': (df.index.min(), df.index.max()),
                'columns': df.columns.tolist(),
                'missing_data_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            }
        
        return summary
    
    def plot_sensor_data(self, sensor_name, stations=None, figsize=(15, 8)):
        """
        Plot data for a specific sensor
        
        Args:
            sensor_name (str): Name of the sensor to plot
            stations (list): List of station IDs to plot (if None, plot all)
            figsize (tuple): Figure size
        """
        if sensor_name not in self.data:
            print(f"Sensor {sensor_name} not found in loaded data")
            return
        
        df = self.data[sensor_name]
        
        if stations is None:
            stations = df.columns.tolist()
        
        # Filter stations that exist in the data
        available_stations = [s for s in stations if s in df.columns]
        
        if not available_stations:
            print(f"No available stations found for {sensor_name}")
            return
        
        plt.figure(figsize=figsize)
        
        for station in available_stations:
            if not df[station].isna().all():  # Only plot if station has data
                plt.plot(df.index, df[station], label=station, alpha=0.7)
        
        plt.title(f'{sensor_name} Sensor Data Over Time', fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel(f'{sensor_name}', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_all_sensors_overview(self, figsize=(20, 15)):
        """Create an overview plot of all sensor data"""
        n_sensors = len(self.data)
        n_cols = 3
        n_rows = (n_sensors + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_sensors > 1 else [axes]
        
        for idx, (sensor_name, df) in enumerate(self.data.items()):
            ax = axes[idx]
            
            # Plot first few stations for overview
            stations_to_plot = df.columns[:3]  # Plot first 3 stations
            
            for station in stations_to_plot:
                if not df[station].isna().all():
                    ax.plot(df.index, df[station], label=station, alpha=0.7)
            
            ax.set_title(f'{sensor_name}', fontsize=12)
            ax.set_xlabel('Time')
            ax.set_ylabel(sensor_name)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            if len(stations_to_plot) <= 3:
                ax.legend()
        
        # Hide unused subplots
        for idx in range(n_sensors, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, sensor_names=None, figsize=(12, 10)):
        """
        Plot correlation matrix between different sensors
        
        Args:
            sensor_names (list): List of sensor names to include in correlation
            figsize (tuple): Figure size
        """
        if sensor_names is None:
            sensor_names = list(self.data.keys())
        
        # Create a combined dataframe with all sensors
        combined_data = pd.DataFrame()
        
        for sensor_name in sensor_names:
            if sensor_name in self.data:
                df = self.data[sensor_name]
                # Add sensor prefix to column names
                for col in df.columns:
                    combined_data[f'{sensor_name}_{col}'] = df[col]
        
        # Calculate correlation matrix
        corr_matrix = combined_data.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, 
                    annot=True, 
                    cmap='coolwarm', 
                    center=0,
                    square=True,
                    fmt='.2f')
        plt.title('Sensor Data Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_station_comparison(self, sensor_name, figsize=(15, 10)):
        """
        Compare data from different stations for a specific sensor
        
        Args:
            sensor_name (str): Name of the sensor
            figsize (tuple): Figure size
        """
        if sensor_name not in self.data:
            print(f"Sensor {sensor_name} not found")
            return
        
        df = self.data[sensor_name]
        
        # Create subplots for each station
        n_stations = len(df.columns)
        n_cols = 3
        n_rows = (n_stations + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_stations > 1 else [axes]
        
        for idx, station in enumerate(df.columns):
            ax = axes[idx]
            
            if not df[station].isna().all():
                ax.plot(df.index, df[station], color='blue', alpha=0.7)
                ax.set_title(f'Station: {station}', fontsize=10)
                ax.set_xlabel('Time')
                ax.set_ylabel(sensor_name)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Station: {station} (No Data)', fontsize=10)
        
        # Hide unused subplots
        for idx in range(n_stations, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{sensor_name} Data by Station', fontsize=16)
        plt.tight_layout()
        plt.show()

def main():
    """Main function to demonstrate data loading and visualization"""
    print("Fire Detection Data Loader and Visualizer")
    print("=" * 50)
    
    # Initialize data loader
    loader = FireDetectionDataLoader('data')
    
    # Load all data
    data = loader.load_all_data()
    
    # Print summary
    print("\nData Summary:")
    print("-" * 30)
    summary = loader.get_data_summary()
    for sensor_name, info in summary.items():
        print(f"{sensor_name}:")
        print(f"  Shape: {info['shape']}")
        print(f"  Time range: {info['time_range'][0]} to {info['time_range'][1]}")
        print(f"  Missing data: {info['missing_data_percentage']:.2f}%")
        print()
    
    # Plot individual sensor data
    print("Plotting individual sensor data...")
    for sensor_name in ['PM1', 'Temperature', 'Relative_Humidity']:
        if sensor_name in data:
            loader.plot_sensor_data(sensor_name)
    
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

if __name__ == "__main__":
    main()
