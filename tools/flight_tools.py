import os
import pandas as pd
import pyarrow.parquet as pq
import json
import torch
import numpy as np

class FlightSegmentToolbox:
    def __init__(self, stride, target_hz, stats_path, manifest_path, shard_dir, model_dir, target_phase):
        self.stride = stride
        self.target_hz = target_hz
        self.shard_dir = shard_dir
        
        # 1. Load Stats (JSON)
        with open(stats_path, 'r') as f:
            self.master_stats = json.load(f)
            
        # 2. Load Manifest Data (Pandas)
        self.manifest_df = pd.read_parquet(manifest_path)
        
        # 3. Extract Sensor List from Parquet Metadata Header
        # We use read_schema to get metadata WITHOUT loading the full table rows
        schema = pq.read_schema(manifest_path)
        if schema.metadata and b'sensor_list' in schema.metadata:
            # Decode the bytes and then the JSON
            self.sensors = json.loads(schema.metadata[b'sensor_list'].decode('utf-8'))
        else:
            # Fallback: Pull from master_stats if parquet is missing metadata
            self.sensors = list(self.master_stats.get('metadata', {}).keys())
        
        # 4. Cache Phase-Specific Stats
        self.phase_key = str(target_phase + 1)
        self.phase_stats = self.master_stats['stats'].get(self.phase_key, {})

    def get_sensor_list(self):
        return self.sensors
    
    def get_sensor_count(self):
        return len(self.sensors)

    def get_shard_path(self, flight_id, shard_w_idx):
        """
        Efficiently finds the shard path using Pandas filtering.
        """
        # Search for the shard filename containing the 'w' index
        # Format in manifest: "tail/flight_pX_wY.pt"
        match_str = f"_w{shard_w_idx}.pt"
        mask = (self.manifest_df['flight'] == flight_id) & \
               (self.manifest_df['shard'].str.contains(match_str))
        
        res = self.manifest_df[mask]
        if not res.empty:
            return os.path.join(self.shard_dir, res.iloc[0]['shard'])
        return None

    # ... Include previous helper methods (to_engineering_units, get_time_axis, etc.) ...

    def get_shard_metadata(self, exact_sec):
        """
        Calculates the shard bucket and padding offsets for a specific timestamp.
        """
        # 1. Global Index based on 4Hz
        global_idx = int(exact_sec * self.target_hz)
        
        # 2. Find the shard 'w' index (matching kinematic_sharder.py logic)
        shard_w_idx = (global_idx // self.stride) * self.stride
        window_start_sec = shard_w_idx / self.target_hz
        
        return {
            "shard_w_idx": shard_w_idx,
            "window_start_sec": window_start_sec,
            "global_idx": global_idx
        }

    def get_sensor_units(self, sensor_name):
        return self.master_stats.get('metadata', {}).get(sensor_name, {}).get('units', 'N/A')

    def to_engineering_units(self, sensor_name, normalized_data):
        """
        Converts normalized [-4, 4] shard data back to physical units.
        """
        stats = self.phase_stats.get(sensor_name, {})
        mean = stats.get('mean', 0)
        std = stats.get('std', 1)
        
        # Handle the VRTG offset logic from kinematic_sharder.py
        if sensor_name == 'VRTG':
            return (normalized_data * std) + 1.0
        return (normalized_data * std) + mean

    def get_time_axis(self, window_start_sec, mask):
        """
        Generates an absolute time axis accounting for the kinematic_sharder's centering padding.
        """
        # Find where data actually starts in the padded 128-sample window
        valid_indices = torch.where(mask == 1.0)[0]
        if len(valid_indices) == 0:
            return np.linspace(window_start_sec, window_start_sec + 32, 128)
            
        start_pad = valid_indices[0].item()
        
        # The first valid sample is at window_start_sec
        # Therefore, index 0 of the tensor is (window_start_sec - padding_time)
        padding_offset_sec = start_pad / self.target_hz
        base_time = window_start_sec - padding_offset_sec
        
        return np.array([base_time + (i / self.target_hz) for i in range(128)])

    def locate_anomaly_in_window(self, exact_sec, window_start_sec, mask):
        """
        Calculates the exact index inside a 128-sample tensor for a given timestamp.
        """
        valid_indices = torch.where(mask == 1.0)[0]
        start_pad = valid_indices[0].item() if len(valid_indices) > 0 else 0
        
        # Relative seconds from the start of valid data
        rel_sec = exact_sec - window_start_sec
        return int(rel_sec * self.target_hz) + start_pad
    
    def get_valid_range(self, mask):
        """Returns the start and end indices of the non-padded data."""
        valid_indices = torch.where(mask == 1.0)[0]
        if len(valid_indices) == 0:
            return 0, 127
        return valid_indices[0].item(), valid_indices[-1].item()

    def get_visual_peak_time(self, error_data, mask, time_axis):
        """Finds the timestamp of the maximum error within the valid mask range."""
        s, e = self.get_valid_range(mask)
        peak_idx = np.argmax(error_data[s : e + 1]) + s
        return time_axis[peak_idx]

