"""Copernicus Data Space API client."""

import requests
from typing import List, Dict, Optional, Any
from datetime import datetime
from loguru import logger
from pathlib import Path


class CopernicusClient:
    """Client for Copernicus Data Space Ecosystem."""
    
    BASE_URL = "https://dataspace.copernicus.eu/api"
    
    def __init__(self, username: str = None, password: str = None):
        """Initialize Copernicus client.
        
        Args:
            username: Copernicus username
            password: Copernicus password
        """
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.token = None
        
        if username and password:
            self._authenticate()
    
    def _authenticate(self) -> bool:
        """Authenticate with Copernicus.
        
        Returns:
            True if successful
        """
        try:
            response = requests.post(
                f"{self.BASE_URL}/auth/token",
                json={
                    'username': self.username,
                    'password': self.password,
                },
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            self.token = data.get('access_token')
            
            if self.token:
                self.session.headers.update({
                    'Authorization': f'Bearer {self.token}'
                })
                logger.info("Authenticated with Copernicus")
                return True
        
        except Exception as e:
            logger.error(f"Copernicus authentication failed: {e}")
        
        return False
    
    def search_products(
        self,
        collection: str,
        geometry: Dict[str, Any] = None,
        temporal: tuple = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search for products in Copernicus.
        
        Args:
            collection: Collection name (e.g., 'SENTINEL-1')
            geometry: GeoJSON geometry for spatial filtering
            temporal: Tuple of (start_date, end_date)
            limit: Maximum number of results
        
        Returns:
            List of product metadata
        """
        params = {
            'collection': collection,
            'limit': limit,
        }
        
        if temporal:
            start_date, end_date = temporal
            params['temporal'] = f"[{start_date.isoformat()},{end_date.isoformat()}]"
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/search",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            products = data.get('features', [])
            
            logger.info(f"Found {len(products)} products in {collection}")
            return products
        
        except Exception as e:
            logger.error(f"Error searching Copernicus products: {e}")
            return []
    
    def download_product(
        self,
        product_id: str,
        output_dir: Path,
    ) -> Optional[Path]:
        """Download product from Copernicus.
        
        Args:
            product_id: Product ID
            output_dir: Output directory
        
        Returns:
            Path to downloaded file or None if failed
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/download/{product_id}",
                stream=True,
                timeout=3600  # 1 hour timeout for large files
            )
            response.raise_for_status()
            
            filename = product_id.split('/')[-1] + '.zip'
            output_path = output_dir / filename
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
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
            logger.error(f"Error downloading product: {e}")
            return None
