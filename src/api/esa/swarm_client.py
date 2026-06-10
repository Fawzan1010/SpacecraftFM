"""ESA SWARM mission data client."""

import requests
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
from pathlib import Path


class SWARMClient:
    """Client for SWARM mission data retrieval."""
    
    BASE_URL = "https://swarm-diss.eo.esa.int"
    
    PRODUCT_TYPES = {
        'MAG': {
            'name': 'Magnetometer',
            'description': 'Magnetometer measurements',
            'frequency': '1 Hz',
            'data_type': 'magnetic',
        },
        'ACC': {
            'name': 'Accelerometer',
            'description': 'Accelerometer measurements',
            'frequency': '10 Hz',
            'data_type': 'acceleration',
        },
        'GPS': {
            'name': 'GPS Position',
            'description': 'GPS position and velocity',
            'frequency': '1 Hz',
            'data_type': 'orbit',
        },
        'VFM': {
            'name': 'Vector Field',
            'description': 'Vector magnetic field data',
            'frequency': '1 Hz',
            'data_type': 'magnetic',
        },
        'TEC': {
            'name': 'Total Electron Content',
            'description': 'Ionospheric TEC measurements',
            'frequency': 'variable',
            'data_type': 'plasma',
        },
    }
    
    def __init__(self, username: str = None, password: str = None):
        """Initialize SWARM client.
        
        Args:
            username: ESA EarthData username
            password: ESA EarthData password
        """
        self.username = username
        self.password = password
        self.session = requests.Session()
        if username and password:
            self.session.auth = (username, password)
    
    def get_product_info(self, product_type: str) -> Optional[Dict[str, Any]]:
        """Get information about a SWARM product type.
        
        Args:
            product_type: Product type code
        
        Returns:
            Product metadata dictionary
        """
        return self.PRODUCT_TYPES.get(product_type)
    
    def list_product_types(self) -> List[str]:
        """List all available SWARM product types.
        
        Returns:
            List of product type codes
        """
        return list(self.PRODUCT_TYPES.keys())
    
    def search_data(
        self,
        satellite: str,
        product_type: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Search for available SWARM data files.
        
        Args:
            satellite: Satellite ID ('A', 'B', or 'C')
            product_type: Product type
            start_date: Start date
            end_date: End date
        
        Returns:
            List of available data files with metadata
        """
        params = {
            'satellite': satellite,
            'productType': product_type,
            'startTime': start_date.isoformat(),
            'endTime': end_date.isoformat(),
        }
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/search",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            files = data.get('results', [])
            
            logger.info(f"Found {len(files)} SWARM {satellite} {product_type} files")
            return files
        
        except Exception as e:
            logger.error(f"Error searching SWARM data: {e}")
            return []
    
    def download_data(
        self,
        file_id: str,
        output_dir: Path,
    ) -> Optional[Path]:
        """Download SWARM data file.
        
        Args:
            file_id: File ID from search results
            output_dir: Output directory
        
        Returns:
            Path to downloaded file or None if failed
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/download/{file_id}",
                stream=True,
                timeout=300
            )
            response.raise_for_status()
            
            filename = response.headers.get('content-disposition', 'swarm_data.nc')
            output_path = output_dir / filename
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Downloaded {output_path.name}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error downloading SWARM data: {e}")
            return None
