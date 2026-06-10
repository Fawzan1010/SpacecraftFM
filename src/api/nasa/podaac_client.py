"""NASA PO.DAAC (Physical Oceanography Data) API client."""

import requests
from typing import List, Dict, Optional, Any
from datetime import datetime
from loguru import logger
import json


class PODAACClient:
    """Client for NASA PO.DAAC API."""
    
    BASE_URL = "https://podaac.jpl.nasa.gov/api"
    
    def __init__(self, api_key: str = None):
        """Initialize PO.DAAC client.
        
        Args:
            api_key: PO.DAAC API key (optional)
        """
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def search_datasets(
        self,
        keyword: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Search for datasets in PO.DAAC.
        
        Args:
            keyword: Search keyword
            limit: Maximum number of results
            offset: Offset for pagination
        
        Returns:
            List of dataset metadata dictionaries
        """
        params = {
            'search': keyword,
            'limit': limit,
            'offset': offset,
            'sort': '-score',
        }
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/search",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            datasets = data.get('results', [])
            
            logger.info(f"Found {len(datasets)} datasets matching '{keyword}'")
            return datasets
        
        except Exception as e:
            logger.error(f"Error searching PO.DAAC datasets: {e}")
            return []
    
    def get_dataset_info(
        self,
        dataset_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get detailed information about a dataset.
        
        Args:
            dataset_id: Dataset ID
        
        Returns:
            Dataset metadata dictionary
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/datasets/{dataset_id}",
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Retrieved metadata for dataset {dataset_id}")
            return data
        
        except Exception as e:
            logger.error(f"Error getting dataset info: {e}")
            return None
    
    def list_granules(
        self,
        dataset_id: str,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List granules in a dataset.
        
        Args:
            dataset_id: Dataset ID
            start_date: Start date for search
            end_date: End date for search
            limit: Maximum number of results
        
        Returns:
            List of granule metadata
        """
        params = {
            'limit': limit,
        }
        
        if start_date:
            params['startTime'] = start_date.isoformat()
        if end_date:
            params['endTime'] = end_date.isoformat()
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/datasets/{dataset_id}/granules",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            granules = data.get('results', [])
            
            logger.info(f"Found {len(granules)} granules in dataset {dataset_id}")
            return granules
        
        except Exception as e:
            logger.error(f"Error listing granules: {e}")
            return []
