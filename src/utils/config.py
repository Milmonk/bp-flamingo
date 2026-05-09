"""
BP-FLAMINGO: Configuration loader utility.

Loads the central YAML config and provides easy access to all parameters.
Usage:
    from src.utils.config import load_config
    cfg = load_config()
    print(cfg["openflamingo"]["model_name"])
"""

import os
import yaml
from pathlib import Path


def find_project_root() -> Path:
    """Find the project root by looking for configs/config.yaml."""
    current = Path.cwd()
    # Walk up from current directory
    for parent in [current] + list(current.parents):
        if (parent / "configs" / "config.yaml").exists():
            return parent
    raise FileNotFoundError(
        "Could not find project root (looking for configs/config.yaml). "
        "Make sure you run from within the bp-flamingo project."
    )


def load_config(config_path: str = None) -> dict:
    """
    Load the YAML configuration file.

    Args:
        config_path: Optional explicit path to config file.
                     If None, auto-detects from project root.

    Returns:
        Dictionary with all configuration parameters.
    """
    if config_path is None:
        root = find_project_root()
        config_path = root / "configs" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve relative paths to absolute
    root = config_path.parent.parent  # configs/ -> project root
    for key, value in config.get("paths", {}).items():
        if isinstance(value, str) and not os.path.isabs(value):
            config["paths"][key] = str(root / value)

    return config


def get_output_path(config: dict, subdir: str, filename: str) -> Path:
    """
    Build an output file path and ensure the directory exists.

    Args:
        config: Loaded config dictionary.
        subdir: Subdirectory key from config["paths"] (e.g., "captions_dir").
        filename: Output filename.

    Returns:
        Full Path object.
    """
    dir_path = Path(config["paths"][subdir])
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path / filename


if __name__ == "__main__":
    # Quick test
    cfg = load_config()
    print("Config loaded successfully!")
    print(f"  OpenFlamingo model: {cfg['openflamingo']['model_name']}")
    print(f"  Languages: {cfg['dataset']['languages']}")
    print(f"  Metrics: {cfg['evaluation']['metrics']}")
