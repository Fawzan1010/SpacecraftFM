"""NASA EarthData authentication handler."""

import os
import json
from pathlib import Path
from typing import Optional
from loguru import logger
import requests
from requests.auth import HTTPBasicAuth


class EarthdataAuth:
    """Handle NASA EarthData authentication."""
    
    CREDS_FILE = Path.home() / ".earthdata"
    AUTH_URL = "https://urs.earthdata.nasa.gov"
    
    def __init__(self):
        """Initialize EarthData auth handler."""
        self.username = None
        self.password = None
        self.token = None
        self.load_credentials()
    
    def load_credentials(self) -> None:
        """Load credentials from file or environment."""
        # Try from file first
        if self.CREDS_FILE.exists():
            try:
                with open(self.CREDS_FILE, 'r') as f:
                    creds = json.load(f)
                    self.username = creds.get('username')
                    self.password = creds.get('password')
                    logger.info(f"Loaded EarthData credentials for {self.username}")
                    return
            except Exception as e:
                logger.warning(f"Failed to load credentials from file: {e}")
        
        # Try from environment
        self.username = os.getenv('EARTHDATA_USERNAME')
        self.password = os.getenv('EARTHDATA_PASSWORD')
        
        if self.username and self.password:
            logger.info(f"Loaded EarthData credentials from environment")
        else:
            logger.warning("No EarthData credentials found")
    
    def setup_interactive(self) -> None:
        """Interactively setup credentials."""
        logger.info("Setting up NASA EarthData credentials...")
        logger.info(f"Register at: {self.AUTH_URL}")
        
        self.username = input("Enter NASA EarthData username: ").strip()
        self.password = input("Enter NASA EarthData password: ").strip()
        
        if self.username and self.password:
            self.save_credentials()
            logger.info("Credentials saved to ~/.earthdata")
        else:
            logger.error("Invalid credentials")
    
    def save_credentials(self) -> None:
        """Save credentials to file."""
        creds = {
            'username': self.username,
            'password': self.password,
        }
        
        self.CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.CREDS_FILE, 'w') as f:
            json.dump(creds, f)
        
        # Restrict file permissions
        os.chmod(self.CREDS_FILE, 0o600)
    
    def get_auth(self) -> Optional[HTTPBasicAuth]:
        """Get authentication object for requests.
        
        Returns:
            HTTPBasicAuth object or None if credentials not available
        """
        if self.username and self.password:
            return HTTPBasicAuth(self.username, self.password)
        return None
    
    def test_connection(self) -> bool:
        """Test connection to EarthData.
        
        Returns:
            True if connection successful
        """
        try:
            auth = self.get_auth()
            if not auth:
                logger.error("No credentials available")
                return False
            
            response = requests.get(
                f"{self.AUTH_URL}/api/users/profile",
                auth=auth,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("EarthData connection successful")
                return True
            else:
                logger.error(f"EarthData connection failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"EarthData connection test failed: {e}")
            return False
