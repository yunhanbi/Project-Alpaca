"""
Market Data Gateway - Simulates live market data feed from historical files
Part 2, Step 1: Gateway for Data Ingestion
"""

import pandas as pd
import numpy as np
import time
from typing import Iterator, Dict, Any
from datetime import datetime
import queue
import threading


class MarketDataGateway:
    """Simulates live market data feed from historical files"""
    
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.data = None
        self.current_index = 0
        self.is_streaming = False
        self.data_queue = queue.Queue()
        
    def load_data(self):
        """Load historical data from CSV file"""
        try:
            self.data = pd.read_csv(self.csv_file_path, index_col='Datetime', parse_dates=True)
            
            # Convert price/volume columns to numeric (fixes string formatting errors)
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Drop rows with invalid data
            self.data.dropna(inplace=True)
            
            print(f"Loaded {len(self.data)} records from {self.csv_file_path}")
        except Exception as e:
            print(f"Error loading data: {e}")
            
    def stream_data(self, delay: float = 0.1) -> Iterator[Dict[str, Any]]:
        """Stream data row-by-row to mimic real-time updates"""
        if self.data is None:
            self.load_data()
            
        self.is_streaming = True
        
        for timestamp, row in self.data.iterrows():
            if not self.is_streaming:
                break
                
            market_tick = {
                'timestamp': timestamp,
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close'],
                'volume': row['Volume']
            }
            
            yield market_tick
            time.sleep(delay)  # Simulate real-time delay
            
    def stop_streaming(self):
        """Stop the data stream"""
        self.is_streaming = False
        
    def get_latest_tick(self) -> Dict[str, Any]:
        """Get the latest market tick"""
        if self.data is None or self.current_index >= len(self.data):
            return None
            
        row = self.data.iloc[self.current_index]
        timestamp = self.data.index[self.current_index]
        
        self.current_index += 1
        
        return {
            'timestamp': timestamp,
            'open': row['Open'],
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'volume': row['Volume']
        }
        
    def reset_stream(self):
        """Reset stream to beginning"""
        self.current_index = 0
        self.is_streaming = False
        
    def get_current_position(self) -> int:
        """Get current position in the data stream"""
        return self.current_index
        
    def get_total_records(self) -> int:
        """Get total number of records"""
        if self.data is None:
            self.load_data()
        return len(self.data) if self.data is not None else 0