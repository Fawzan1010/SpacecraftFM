"""Configuration loader using Hydra."""

from pathlib import Path
from typing import Any, Dict
import yaml
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf, DictConfig


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_hydra_config(config_dir: str, config_name: str) -> DictConfig:
    """Load configuration using Hydra.
    
    Args:
        config_dir: Directory containing config files
        config_name: Name of config file (without .yaml)
    
    Returns:
        OmegaConf DictConfig object
    """
    config_dir = str(Path(config_dir).absolute())
    
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name=config_name)
    
    return cfg


def save_yaml_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save config file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries (override_config takes precedence).
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
    
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
