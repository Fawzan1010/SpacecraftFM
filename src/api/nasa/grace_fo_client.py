"""GRACE-FO specific data retrieval client."""

import requests
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
from pathlib import Path
import os


class GRACEFOClient:
    """Client for GRACE-FO data retrieval."""
    
    # GRACE-FO product types and their metadata
    PRODUCT_TYPES = {
        'SCA1B': {
            'name': 'Star Camera Attitude',
            'description': 'Attitude products from star camera',
            'frequency': 'daily',
            'data_type': 'attitude',
        },
        'ACC1B': {
            'name': 'Accelerometer',
            'description': 'Accelerometer products',
            'frequency': 'daily',
            'data_type': 'dynamics',
        },
        'ORB1B': {
            'name': 'Orbit',
            'description': 'Orbit products',
            'frequency': 'daily',
            'data_type': 'orbit',
        },
        'CLK1B': {
            'name': 'Clock',
            'description': 'Clock offset products',
            'frequency': 'daily',
            'data_type': 'clock',
        },
        'KBR1B': {
            'name': 'K-Band Ranging',
            'description': 'Inter-satellite K-band ranging data',
            'frequency': 'daily',
            'data_type': 'ranging',
        },
    }
    
    # PO.DAAC dataset IDs for GRACE-FO products
    PODAAC_DATASET_IDS = {
        'GRACEFO_C_SCA1B': 'GRACEFO_C_SCA1B_V02',
        'GRACEFO_D_SCA1B': 'GRACEFO_D_SCA1B_V02',
        'GRACEFO_C_ACC1B': 'GRACEFO_C_ACC1B_V02',
        'GRACEFO_D_ACC1B': 'GRACEFO_D_ACC1B_V02',
        'GRACEFO_C_ORB1B': 'GRACEFO_C_ORB1B_V02',
        'GRACEFO_D_ORB1B': 'GRACEFO_D_ORB1B_V02',
    }
    
    # ESA GRACE-FO FTP base directory
    ESA_FTP_BASE = "ftp://isdcftp.gfz-potsdam.de/grace-fo"
    
    def __init__(self, username: str = None, password: str = None):
        """Initialize GRACE-FO client.
        
        Args:
            username: NASA EarthData username
            password: NASA EarthData password
        """
        self.username = username
        self.password = password
        self.session = requests.Session()
        if username and password:
            self.session.auth = (username, password)
    
    def get_product_info(self, product_type: str) -> Optional[Dict[str, Any]]:
        """Get information about a product type.
        
        Args:
            product_type: Product type code (e.g., 'SCA1B')
        
        Returns:
            Product metadata dictionary
        """
        return self.PRODUCT_TYPES.get(product_type)
    
    def list_product_types(self) -> List[str]:
        """List all available GRACE-FO product types.
        
        Returns:
            List of product type codes
        """
        return list(self.PRODUCT_TYPES.keys())
    
    def get_data_url(
        self,
        spacecraft: str,
        product_type: str,
        date: datetime,
    ) -> Optional[str]:
        """Get download URL for GRACE-FO data file.
        
        Args:
            spacecraft: Spacecraft ID ('C' or 'D')
            product_type: Product type (e.g., 'SCA1B')
            date: Date for data file
        
        Returns:
            Download URL or None if not found
        """
        if product_type not in self.PRODUCT_TYPES:
            logger.error(f"Unknown product type: {product_type}")
            return None
        
        # Construct filename following GRACE-FO naming convention
        year = date.strftime('%Y')
        doy = date.strftime('%j')
        
        filename = f"GRACEFO_{spacecraft}_{product_type}_V02_0_0000.binex"
        
        # URL pattern for PO.DAAC OPeNDAP server
        base_url = f"https://podaac-opendap.jpl.nasa.gov/opendap/hyrax/allData/grace_fo/L1B"
        url = f"{base_url}/{year}/{doy}/{filename}"
        
        return url
    
    def search_data_availability(
        self,
        spacecraft: str,
        product_type: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Search for available GRACE-FO data files.
        
        Args:
            spacecraft: Spacecraft ID ('C' or 'D')
            product_type: Product type
            start_date: Start date
            end_date: End date
        
        Returns:
            List of available data files with metadata
        """
        available_files = []
        current_date = start_date
        
        while current_date <= end_date:
            url = self.get_data_url(spacecraft, product_type, current_date)
            
            if url:
                # Check if file exists
                try:
                    response = self.session.head(url, timeout=5)
                    if response.status_code == 200:
                        available_files.append({
                            'date': current_date.isoformat(),
                            'spacecraft': spacecraft,
                            'product_type': product_type,
                            'url': url,
                            'size_bytes': int(response.headers.get('content-length', 0)),
                        })
                except Exception as e:
                    logger.debug(f"File not found or error checking {url}: {e}")
            
            current_date += timedelta(days=1)
        
        logger.info(f"Found {len(available_files)} available files for {spacecraft} {product_type}")
        return available_files
    
    def download_data(
        self,
        url: str,
        output_dir: Path,
        filename: str = None,
    ) -> Optional[Path]:
        """Download GRACE-FO data file.
        
        Args:
            url: Download URL
            output_dir: Output directory
            filename: Custom filename (optional)
        
        Returns:
            Path to downloaded file or None if failed
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = url.split('/')[-1]
        
        output_path = output_dir / filename
        
        try:
            logger.info(f"Downloading {url}...")
            response = self.session.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Progress: {progress:.1f}%", end='\r')
            
            logger.info(f"Downloaded {output_path.name}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            if output_path.exists():
                output_path.unlink()
            return None
