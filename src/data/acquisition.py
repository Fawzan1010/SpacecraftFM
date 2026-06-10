"""Data acquisition and download orchestration."""

import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import json

from ..api.nasa import CMRClient, GRACEFOClient
from ..api.esa import SWARMClient
from ..api.noaa import NOAASWPCClient
from ..api.igrf import IGRFClient
from ..api.atmosphere import NRLMSISE00Client, JB2008Client


class DataAcquisitionManager:
    """Manage data acquisition from all sources."""
    
    def __init__(
        self,
        data_dir: Path,
        username: str = None,
        password: str = None,
    ):
        """Initialize data acquisition manager.
        
        Args:
            data_dir: Root directory for downloaded data
            username: NASA/ESA EarthData username
            password: NASA/ESA EarthData password
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API clients
        self.cmr_client = CMRClient(username, password)
        self.grace_fo_client = GRACEFOClient(username, password)
        self.swarm_client = SWARMClient(username, password)
        self.swpc_client = NOAASWPCClient()
        self.igrf_client = IGRFClient()
        self.nrlmsise_client = NRLMSISE00Client()
        self.jb2008_client = JB2008Client()
        
        self.username = username
        self.password = password
        
        logger.info(f"Initialized DataAcquisitionManager with root dir: {self.data_dir}")
    
    def acquire_grace_fo_data(
        self,
        spacecraft: str = 'C',
        product_types: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> Dict[str, List[Path]]:
        """Acquire GRACE-FO mission data.
        
        Args:
            spacecraft: Spacecraft ID ('C' or 'D')
            product_types: List of product types to download
            start_date: Start date for acquisition
            end_date: End date for acquisition
        
        Returns:
            Dictionary mapping product types to list of downloaded file paths
        """
        if product_types is None:
            product_types = ['SCA1B', 'ACC1B', 'ORB1B']
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        downloaded_files = {}
        
        for product_type in product_types:
            logger.info(f"Acquiring GRACE-FO {spacecraft} {product_type} data from {start_date.date()} to {end_date.date()}")
            
            # Search for available files
            available_files = self.grace_fo_client.search_data_availability(
                spacecraft=spacecraft,
                product_type=product_type,
                start_date=start_date,
                end_date=end_date,
            )
            
            # Create product directory
            product_dir = self.data_dir / 'GRACEFO' / f'GRACEFO_{spacecraft}' / product_type
            product_dir.mkdir(parents=True, exist_ok=True)
            
            # Download files
            downloaded_files[product_type] = []
            for file_info in available_files:
                try:
                    file_path = self.grace_fo_client.download_data(
                        url=file_info['url'],
                        output_dir=product_dir,
                    )
                    if file_path:
                        downloaded_files[product_type].append(file_path)
                        logger.info(f"Downloaded: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error downloading file: {e}")
        
        logger.info(f"GRACE-FO data acquisition complete")
        return downloaded_files
    
    def acquire_swarm_data(
        self,
        satellites: List[str] = None,
        product_types: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> Dict[str, Dict[str, List[Path]]]:
        """Acquire SWARM mission data.
        
        Args:
            satellites: Satellite IDs to acquire ('A', 'B', 'C')
            product_types: List of product types
            start_date: Start date for acquisition
            end_date: End date for acquisition
        
        Returns:
            Nested dictionary mapping satellites -> product types -> file paths
        """
        if satellites is None:
            satellites = ['A', 'B', 'C']
        if product_types is None:
            product_types = ['MAG', 'ACC', 'GPS']
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        downloaded_files = {}
        
        for satellite in satellites:
            downloaded_files[satellite] = {}
            
            for product_type in product_types:
                logger.info(f"Acquiring SWARM {satellite} {product_type} data")
                
                # Search for available files
                available_files = self.swarm_client.search_data(
                    satellite=satellite,
                    product_type=product_type,
                    start_date=start_date,
                    end_date=end_date,
                )
                
                # Create product directory
                product_dir = self.data_dir / 'SWARM' / f'SWARM_{satellite}' / product_type
                product_dir.mkdir(parents=True, exist_ok=True)
                
                # Download files
                downloaded_files[satellite][product_type] = []
                for file_info in available_files:
                    try:
                        file_path = self.swarm_client.download_data(
                            file_id=file_info['id'],
                            output_dir=product_dir,
                        )
                        if file_path:
                            downloaded_files[satellite][product_type].append(file_path)
                    except Exception as e:
                        logger.error(f"Error downloading SWARM data: {e}")
        
        logger.info(f"SWARM data acquisition complete")
        return downloaded_files
    
    def acquire_space_weather_data(
        self,
        parameters: List[str] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Acquire space weather data from NOAA SWPC.
        
        Args:
            parameters: Space weather parameters to acquire
            days: Number of days to retrieve
        
        Returns:
            Dictionary of space weather data
        """
        if parameters is None:
            parameters = ['kp', 'dst', 'solar_wind', 'f107']
        
        logger.info(f"Acquiring space weather data for {days} days")
        
        space_weather_data = {}
        
        if 'kp' in parameters:
            space_weather_data['kp'] = self.swpc_client.get_kp_index(days=days)
        
        if 'dst' in parameters:
            space_weather_data['dst'] = self.swpc_client.get_dst_index(days=days)
        
        if 'solar_wind' in parameters:
            space_weather_data['solar_wind'] = self.swpc_client.get_solar_wind()
        
        if 'f107' in parameters:
            space_weather_data['f107'] = self.swpc_client.get_f107_index(days=days)
        
        logger.info(f"Space weather data acquisition complete")
        return space_weather_data
    
    def compute_geomagnetic_environment(
        self,
        latitude: float,
        longitude: float,
        altitude_km: float,
    ) -> Dict[str, float]:
        """Compute geomagnetic environment at location.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            altitude_km: Altitude in kilometers
        
        Returns:
            Geomagnetic field parameters
        """
        return self.igrf_client.compute_magnetic_field(
            latitude, longitude, altitude_km
        )
    
    def compute_atmospheric_environment(
        self,
        latitude: float,
        longitude: float,
        altitude_km: float,
        datetime_utc: datetime,
        f107: float = 100.0,
        kp: float = 2.0,
    ) -> Dict[str, float]:
        """Compute atmospheric environment at location.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            altitude_km: Altitude in kilometers
            datetime_utc: UTC datetime
            f107: Solar flux
            kp: Kp index
        
        Returns:
            Atmospheric parameters from NRLMSISE-00
        """
        return self.nrlmsise_client.get_density(
            latitude, longitude, altitude_km, datetime_utc, f107, f107, kp
        )
    
    def save_metadata(
        self,
        metadata: Dict[str, Any],
        filename: str = 'acquisition_metadata.json',
    ) -> Path:
        """Save acquisition metadata.
        
        Args:
            metadata: Metadata dictionary
            filename: Output filename
        
        Returns:
            Path to saved metadata file
        """
        metadata_path = self.data_dir / filename
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to {metadata_path}")
        return metadata_path
