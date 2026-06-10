"""NASA CMR (Common Metadata Repository) API client."""

import requests
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
import json


class CMRClient:
    """Client for NASA CMR API."""
    
    BASE_URL = "https://cmr.earthdata.nasa.gov/search"
    
    def __init__(self, username: str = None, password: str = None):
        """Initialize CMR client.
        
        Args:
            username: NASA EarthData username
            password: NASA EarthData password
        """
        self.username = username
        self.password = password
        self.session = requests.Session()
        if username and password:
            self.session.auth = (username, password)
    
    def search_collections(
        self,
        keyword: str,
        platform: str = None,
        instrument: str = None,
        temporal: tuple = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search for collections in CMR.
        
        Args:
            keyword: Search keyword (e.g., 'GRACE-FO')
            platform: Platform name (e.g., 'GRACE-FO')
            instrument: Instrument name
            temporal: Tuple of (start_date, end_date) as datetime objects
            limit: Maximum number of results
        
        Returns:
            List of collection metadata dictionaries
        """
        params = {
            'keyword': keyword,
            'page_size': min(limit, 2000),
            'sort_key': '-start_date',
        }
        
        if platform:
            params['platform'] = platform
        if instrument:
            params['instrument'] = instrument
        
        if temporal:
            start_date, end_date = temporal
            params['temporal'] = f"{start_date.isoformat()},{end_date.isoformat()}"
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/collections.json",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            collections = data.get('feed', {}).get('entry', [])
            
            logger.info(f"Found {len(collections)} collections matching '{keyword}'")
            return collections
        
        except Exception as e:
            logger.error(f"Error searching CMR collections: {e}")
            return []
    
    def search_granules(
        self,
        collection_id: str,
        temporal: tuple = None,
        bounding_box: tuple = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search for granules in a collection.
        
        Args:
            collection_id: Collection ID (e.g., 'G1000000-NASA_OBDAAC')
            temporal: Tuple of (start_date, end_date) as datetime objects
            bounding_box: Tuple of (west, south, east, north)
            limit: Maximum number of results
        
        Returns:
            List of granule metadata dictionaries
        """
        params = {
            'collection_concept_id': collection_id,
            'page_size': min(limit, 2000),
            'sort_key': '-start_date',
        }
        
        if temporal:
            start_date, end_date = temporal
            params['temporal'] = f"{start_date.isoformat()},{end_date.isoformat()}"
        
        if bounding_box:
            west, south, east, north = bounding_box
            params['bounding_box'] = f"{west},{south},{east},{north}"
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/granules.json",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            granules = data.get('feed', {}).get('entry', [])
            
            logger.info(f"Found {len(granules)} granules in collection {collection_id}")
            return granules
        
        except Exception as e:
            logger.error(f"Error searching CMR granules: {e}")
            return []
    
    def get_collection_metadata(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed metadata for a collection.
        
        Args:
            collection_id: Collection ID
        
        Returns:
            Collection metadata dictionary
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/collections/{collection_id}.json",
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get('Collection', {})
        
        except Exception as e:
            logger.error(f"Error getting collection metadata: {e}")
            return None
    
    def list_grace_fo_collections(self) -> List[Dict[str, Any]]:
        """List all available GRACE-FO collections.
        
        Returns:
            List of GRACE-FO collection metadata
        """
        return self.search_collections(
            keyword='GRACE-FO',
            platform='GRACE-FO',
            limit=100
        )
    
    def search_grace_fo_granules(
        self,
        product_type: str,
        spacecraft: str = 'C',
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> List[Dict[str, Any]]:
        """Search for GRACE-FO granules of specific type.
        
        Args:
            product_type: Product type (e.g., 'SCA1B', 'ACC1B', 'ORB1B')
            spacecraft: Spacecraft ID ('C' or 'D')
            start_date: Start date for search
            end_date: End date for search
        
        Returns:
            List of granule metadata
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        keyword = f"GRACE-FO {spacecraft} {product_type}"
        
        return self.search_granules(
            collection_id=None,  # Will use keyword search instead
            temporal=(start_date, end_date),
            limit=1000
        )
