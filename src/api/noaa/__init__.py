"""NOAA Space Weather Prediction Center (SWPC) API client."""

import requests
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
import json


class NOAASWPCClient:
    """Client for NOAA SWPC space weather data."""
    
    BASE_URL = "https://services.swpc.noaa.gov/json"
    
    def __init__(self):
        """Initialize NOAA SWPC client."""
        self.session = requests.Session()
    
    def get_kp_index(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get Kp index data.
        
        Args:
            days: Number of days to retrieve (default: 30)
        
        Returns:
            List of Kp index data points
        """
        try:
            forecast_url = f"{self.BASE_URL}/planetary_k_index_forecast.json"
            response = self.session.get(forecast_url, timeout=30)
            response.raise_for_status()
            
            forecast_data = response.json()
            historical_data = self._get_kp_historical(days)
            
            logger.info(f"Retrieved Kp index data for {days} days")
            return historical_data + forecast_data
        
        except Exception as e:
            logger.error(f"Error retrieving Kp index: {e}")
            return []
    
    def get_dst_index(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get Dst (Disturbance Storm Time) index data.
        
        Args:
            days: Number of days to retrieve
        
        Returns:
            List of Dst index data points
        """
        try:
            url = f"{self.BASE_URL}/dst_30min.json"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Retrieved Dst index data")
            return data
        
        except Exception as e:
            logger.error(f"Error retrieving Dst index: {e}")
            return []
    
    def get_solar_wind(
        self,
        parameter: str = 'speed',
    ) -> List[Dict[str, Any]]:
        """Get solar wind data.
        
        Args:
            parameter: Parameter to retrieve ('speed', 'density', 'temperature')
        
        Returns:
            List of solar wind data points
        """
        try:
            url = f"{self.BASE_URL}/ace_swepam_solar_wind.json"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Retrieved solar wind {parameter} data")
            return data
        
        except Exception as e:
            logger.error(f"Error retrieving solar wind data: {e}")
            return []
    
    def get_geomagnetic_alerts(
        self,
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """Get recent geomagnetic storm alerts.
        
        Args:
            hours: Number of hours to look back
        
        Returns:
            List of alert data
        """
        try:
            url = f"{self.BASE_URL}/alerts_and_watches.json"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            cutoff_time = datetime.now() - timedelta(hours=hours)
            filtered_alerts = [
                alert for alert in data
                if 'time' in alert and alert['time'] > cutoff_time.isoformat()
            ]
            
            logger.info(f"Retrieved {len(filtered_alerts)} geomagnetic alerts from last {hours} hours")
            return filtered_alerts
        
        except Exception as e:
            logger.error(f"Error retrieving geomagnetic alerts: {e}")
            return []
    
    def _get_kp_historical(
        self,
        days: int,
    ) -> List[Dict[str, Any]]:
        """Get historical Kp index data."""
        try:
            logger.warning("Historical Kp data retrieval not fully implemented")
            return []
        except Exception as e:
            logger.error(f"Error retrieving historical Kp: {e}")
            return []
