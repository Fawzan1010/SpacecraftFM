"""Physical constants and mission parameters for SpacecraftFM."""

import numpy as np
from dataclasses import dataclass
from typing import Dict

# Physical Constants (SI units)
EARTH_RADIUS_KM = 6371.0
EARTH_RADIUS_M = 6.371e6
GM_EARTH = 3.986004418e14  # m^3/s^2
EARTH_OMEGA = 7.2921150e-5  # rad/s
EARTH_J2 = 1.08262668e-3

# Solar Constants
SOLAR_CONSTANT = 1361.0  # W/m^2

# Magnetic Constants
MU_0 = 4 * np.pi * 1e-7  # H/m
EARTH_MAG_MOMENT = 7.91e15  # A*m^2

# Atmospheric Model Constants
ATMOS_SCALE_HEIGHT = 8500.0  # m

# Time Constants
GPS_WEEK_SECONDS = 604800  # s
GPS_EPOCH_OFFSET = 315964800  # s (1980-01-06 to 1970-01-01)


@dataclass
class MissionConfig:
    """Configuration for spacecraft missions."""
    
    name: str
    norad_id: int
    launch_date: str
    mass_kg: float
    inertia_tensor: np.ndarray  # 3x3 matrix
    drag_area_m2: float
    drag_coefficient: float
    srp_area_m2: float
    srp_coefficient: float
    orbit_altitude_km: float
    orbit_inclination_deg: float
    

# Mission Configurations
MISSION_CONFIGS: Dict[str, MissionConfig] = {
    "GRACEFO_C": MissionConfig(
        name="GRACE-FO C",
        norad_id=43013,
        launch_date="2018-05-22",
        mass_kg=2009.0,
        inertia_tensor=np.diag([500.0, 500.0, 300.0]),
        drag_area_m2=15.0,
        drag_coefficient=2.5,
        srp_area_m2=10.0,
        srp_coefficient=1.25,
        orbit_altitude_km=490.0,
        orbit_inclination_deg=89.0,
    ),
    "GRACEFO_D": MissionConfig(
        name="GRACE-FO D",
        norad_id=43014,
        launch_date="2018-05-22",
        mass_kg=2009.0,
        inertia_tensor=np.diag([500.0, 500.0, 300.0]),
        drag_area_m2=15.0,
        drag_coefficient=2.5,
        srp_area_m2=10.0,
        srp_coefficient=1.25,
        orbit_altitude_km=490.0,
        orbit_inclination_deg=89.0,
    ),
    "SWARM_A": MissionConfig(
        name="SWARM A",
        norad_id=39444,
        launch_date="2013-11-22",
        mass_kg=1440.0,
        inertia_tensor=np.diag([900.0, 900.0, 400.0]),
        drag_area_m2=12.0,
        drag_coefficient=2.5,
        srp_area_m2=8.0,
        srp_coefficient=1.25,
        orbit_altitude_km=460.0,
        orbit_inclination_deg=87.4,
    ),
    "SWARM_B": MissionConfig(
        name="SWARM B",
        norad_id=39445,
        launch_date="2013-11-22",
        mass_kg=1440.0,
        inertia_tensor=np.diag([900.0, 900.0, 400.0]),
        drag_area_m2=12.0,
        drag_coefficient=2.5,
        srp_area_m2=8.0,
        srp_coefficient=1.25,
        orbit_altitude_km=520.0,
        orbit_inclination_deg=87.4,
    ),
    "SWARM_C": MissionConfig(
        name="SWARM C",
        norad_id=39446,
        launch_date="2013-11-22",
        mass_kg=1440.0,
        inertia_tensor=np.diag([900.0, 900.0, 400.0]),
        drag_area_m2=12.0,
        drag_coefficient=2.5,
        srp_area_m2=8.0,
        srp_coefficient=1.25,
        orbit_altitude_km=520.0,
        orbit_inclination_deg=87.4,
    ),
}

# Feature Names for Data Pipeline
ATTITUDE_FEATURES = [
    "quaternion_w",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "euler_roll",
    "euler_pitch",
    "euler_yaw",
]

DYNAMICS_FEATURES = [
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "angular_acceleration_x",
    "angular_acceleration_y",
    "angular_acceleration_z",
]

ORBIT_FEATURES = [
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "altitude_km",
    "inclination_deg",
    "true_anomaly_deg",
    "semi_major_axis_km",
]

ENVIRONMENT_FEATURES = [
    "kp_index",
    "dst_index",
    "solar_flux_f107",
    "atmospheric_density",
    "atmospheric_temperature",
    "magnetic_field_strength",
    "magnetic_inclination",
    "magnetic_declination",
    "solar_wind_speed",
    "solar_wind_density",
]

EVENT_FEATURES = [
    "is_maneuver",
    "is_safe_mode",
    "is_sensor_anomaly",
    "is_thruster_event",
    "is_geomagnetic_storm",
]

# Data Catalog Hierarchy
DATA_CATEGORIES = {
    "attitude": ATTITUDE_FEATURES,
    "dynamics": DYNAMICS_FEATURES,
    "orbit": ORBIT_FEATURES,
    "environment": ENVIRONMENT_FEATURES,
    "events": EVENT_FEATURES,
}
