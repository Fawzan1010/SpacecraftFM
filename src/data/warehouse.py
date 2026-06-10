"""Data warehouse and storage management."""

from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger
import pandas as pd
import json
from datetime import datetime


class DataWarehouse:
    """Manage data warehouse storage and retrieval."""
    
    def __init__(self, warehouse_dir: Path):
        """Initialize data warehouse.
        
        Args:
            warehouse_dir: Root directory for warehouse data
        """
        self.warehouse_dir = Path(warehouse_dir)
        self.warehouse_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standard directory structure
        self._create_structure()
        
        logger.info(f"Initialized DataWarehouse at {self.warehouse_dir}")
    
    def _create_structure(self) -> None:
        """Create standard warehouse directory structure."""
        dirs = [
            'missions/GRACEFO_C',
            'missions/GRACEFO_D',
            'missions/SWARM_A',
            'missions/SWARM_B',
            'missions/SWARM_C',
            'space_weather',
            'geomagnetic',
            'atmospheric',
            'processed',
            'features',
            'timeseries',
        ]
        
        for dir_path in dirs:
            (self.warehouse_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    def store_timeseries(
        self,
        data: pd.DataFrame,
        mission_id: str,
        data_type: str,
        filename: str = None,
    ) -> Path:
        """Store time series data.
        
        Args:
            data: DataFrame with time series data
            mission_id: Mission identifier
            data_type: Data type (e.g., 'attitude', 'orbit')
            filename: Custom filename (optional)
        
        Returns:
            Path to stored data file
        """
        # Determine storage path
        if mission_id.startswith('GRACEFO'):
            storage_dir = self.warehouse_dir / 'missions' / mission_id
        elif mission_id.startswith('SWARM'):
            storage_dir = self.warehouse_dir / 'missions' / mission_id
        else:
            storage_dir = self.warehouse_dir / 'timeseries'
        
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{mission_id}_{data_type}_{timestamp}.parquet"
        
        filepath = storage_dir / filename
        
        # Store as Parquet for efficiency
        data.to_parquet(filepath)
        logger.info(f"Stored {len(data)} records to {filepath}")
        
        return filepath
    
    def store_space_weather(
        self,
        data: Dict[str, Any],
        parameter: str,
    ) -> Path:
        """Store space weather data.
        
        Args:
            data: Space weather data dictionary
            parameter: Parameter name (e.g., 'kp', 'dst')
        
        Returns:
            Path to stored data file
        """
        storage_dir = self.warehouse_dir / 'space_weather'
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{parameter}_{timestamp}.json"
        filepath = storage_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Stored space weather data to {filepath}")
        return filepath
    
    def retrieve_timeseries(
        self,
        mission_id: str,
        data_type: str,
    ) -> Optional[pd.DataFrame]:
        """Retrieve time series data.
        
        Args:
            mission_id: Mission identifier
            data_type: Data type
        
        Returns:
            DataFrame with time series data or None if not found
        """
        # Find latest file matching criteria
        if mission_id.startswith('GRACEFO') or mission_id.startswith('SWARM'):
            search_dir = self.warehouse_dir / 'missions' / mission_id
        else:
            search_dir = self.warehouse_dir / 'timeseries'
        
        if not search_dir.exists():
            logger.warning(f"Directory not found: {search_dir}")
            return None
        
        # Find latest file
        matching_files = list(search_dir.glob(f"*{data_type}*.parquet"))
        if not matching_files:
            logger.warning(f"No files found for {mission_id} {data_type}")
            return None
        
        latest_file = sorted(matching_files)[-1]
        logger.info(f"Retrieved data from {latest_file}")
        
        return pd.read_parquet(latest_file)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get warehouse storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'total_size_bytes': 0,
            'file_count': 0,
            'mission_stats': {},
        }
        
        for mission_dir in (self.warehouse_dir / 'missions').iterdir():
            if mission_dir.is_dir():
                mission_id = mission_dir.name
                files = list(mission_dir.rglob('*'))
                size = sum(f.stat().st_size for f in files if f.is_file())
                
                stats['mission_stats'][mission_id] = {
                    'size_bytes': size,
                    'file_count': len([f for f in files if f.is_file()]),
                }
                
                stats['total_size_bytes'] += size
                stats['file_count'] += len([f for f in files if f.is_file()])
        
        return stats
