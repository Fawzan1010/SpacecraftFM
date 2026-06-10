"""Data catalog and metadata management."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger


class DataCatalog:
    """Manage data catalog and metadata."""
    
    def __init__(self, catalog_file: Path = None):
        """Initialize data catalog.
        
        Args:
            catalog_file: Path to catalog JSON file
        """
        self.catalog_file = catalog_file or Path('data_catalog.json')
        self.catalog = self._load_catalog()
    
    def _load_catalog(self) -> Dict[str, Any]:
        """Load catalog from file.
        
        Returns:
            Catalog dictionary
        """
        if self.catalog_file.exists():
            with open(self.catalog_file, 'r') as f:
                return json.load(f)
        return {'missions': {}, 'products': {}}
    
    def save_catalog(self) -> None:
        """Save catalog to file."""
        with open(self.catalog_file, 'w') as f:
            json.dump(self.catalog, f, indent=2, default=str)
        logger.info(f"Saved catalog to {self.catalog_file}")
    
    def add_mission(
        self,
        mission_id: str,
        mission_name: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Add mission to catalog.
        
        Args:
            mission_id: Unique mission ID
            mission_name: Human-readable mission name
            metadata: Mission metadata dictionary
        """
        if 'missions' not in self.catalog:
            self.catalog['missions'] = {}
        
        self.catalog['missions'][mission_id] = {
            'name': mission_name,
            'metadata': metadata,
            'added_at': datetime.now().isoformat(),
        }
        
        logger.info(f"Added mission {mission_id} to catalog")
    
    def add_product(
        self,
        mission_id: str,
        product_type: str,
        product_data: Dict[str, Any],
    ) -> None:
        """Add product to catalog.
        
        Args:
            mission_id: Mission ID
            product_type: Product type (e.g., 'SCA1B', 'MAG')
            product_data: Product metadata
        """
        if 'products' not in self.catalog:
            self.catalog['products'] = {}
        
        if mission_id not in self.catalog['products']:
            self.catalog['products'][mission_id] = {}
        
        if product_type not in self.catalog['products'][mission_id]:
            self.catalog['products'][mission_id][product_type] = []
        
        self.catalog['products'][mission_id][product_type].append({
            'metadata': product_data,
            'added_at': datetime.now().isoformat(),
        })
        
        logger.info(f"Added product {product_type} for mission {mission_id}")
    
    def list_missions(self) -> List[str]:
        """List all missions in catalog.
        
        Returns:
            List of mission IDs
        """
        return list(self.catalog.get('missions', {}).keys())
    
    def list_products(
        self,
        mission_id: str = None,
    ) -> Dict[str, List[str]]:
        """List products in catalog.
        
        Args:
            mission_id: Filter by mission (optional)
        
        Returns:
            Dictionary of mission -> product types
        """
        if mission_id:
            return {mission_id: self.catalog.get('products', {}).get(mission_id, {})}
        return self.catalog.get('products', {})
    
    def get_mission_metadata(
        self,
        mission_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get mission metadata.
        
        Args:
            mission_id: Mission ID
        
        Returns:
            Mission metadata dictionary
        """
        return self.catalog.get('missions', {}).get(mission_id)
    
    def get_product_metadata(
        self,
        mission_id: str,
        product_type: str,
    ) -> List[Dict[str, Any]]:
        """Get product metadata.
        
        Args:
            mission_id: Mission ID
            product_type: Product type
        
        Returns:
            List of product metadata dictionaries
        """
        return self.catalog.get('products', {}).get(mission_id, {}).get(product_type, [])
    
    def export_catalog(
        self,
        output_path: Path,
    ) -> None:
        """Export catalog to file.
        
        Args:
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(self.catalog, f, indent=2, default=str)
        logger.info(f"Exported catalog to {output_path}")
