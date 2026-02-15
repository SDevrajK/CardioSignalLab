"""Configuration management for CardioSignalLab.

Uses attrs with validators for type-safe, validated configuration.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import attrs
from attrs import define, field


def positive_float(instance, attribute, value):
    """Validator: ensure value is a positive float."""
    if value <= 0:
        raise ValueError(f"{attribute.name} must be positive, got {value}")


def positive_int(instance, attribute, value):
    """Validator: ensure value is a positive integer."""
    if value <= 0:
        raise ValueError(f"{attribute.name} must be positive, got {value}")


def nonnegative_float(instance, attribute, value):
    """Validator: ensure value is non-negative."""
    if value < 0:
        raise ValueError(f"{attribute.name} must be non-negative, got {value}")


@define
class ProcessingConfig:
    """Signal processing configuration with scientific validation."""

    # Reproducibility
    random_seed: int = field(default=1234, validator=attrs.validators.instance_of(int))

    # ECG Processing (0.5-40 Hz bandpass typical for R-peak detection)
    ecg_lowcut: float = field(default=0.5, validator=[attrs.validators.instance_of(float), positive_float])
    ecg_highcut: float = field(default=40.0, validator=[attrs.validators.instance_of(float), positive_float])
    ecg_filter_order: int = field(default=4, validator=[attrs.validators.instance_of(int), positive_int])

    # PPG Processing (0.5-8 Hz bandpass typical for pulse detection)
    ppg_lowcut: float = field(default=0.5, validator=[attrs.validators.instance_of(float), positive_float])
    ppg_highcut: float = field(default=8.0, validator=[attrs.validators.instance_of(float), positive_float])
    ppg_filter_order: int = field(default=2, validator=[attrs.validators.instance_of(int), positive_int])

    # EDA Processing (typically 0.05-5 Hz for SCR detection)
    eda_lowcut: float = field(default=0.05, validator=[attrs.validators.instance_of(float), nonnegative_float])
    eda_highcut: float = field(default=5.0, validator=[attrs.validators.instance_of(float), positive_float])

    # EEMD Parameters (Ensemble Empirical Mode Decomposition)
    eemd_ensemble_size: int = field(default=500, validator=[attrs.validators.instance_of(int), positive_int])
    eemd_noise_width: float = field(default=0.2, validator=[attrs.validators.instance_of(float), positive_float])

    # Peak Detection Methods
    ecg_peak_method: str = field(default="neurokit", validator=attrs.validators.in_(["neurokit", "pantompkins", "hamilton"]))
    ppg_peak_method: str = field(default="neurokit", validator=attrs.validators.in_(["neurokit", "elgendi"]))
    eda_scr_method: str = field(default="neurokit", validator=attrs.validators.instance_of(str))

    @ecg_highcut.validator
    def _check_ecg_frequency_order(self, attribute, value):
        """Ensure highcut > lowcut for ECG."""
        if value <= self.ecg_lowcut:
            raise ValueError(f"ecg_highcut ({value}) must be greater than ecg_lowcut ({self.ecg_lowcut})")

    @ppg_highcut.validator
    def _check_ppg_frequency_order(self, attribute, value):
        """Ensure highcut > lowcut for PPG."""
        if value <= self.ppg_lowcut:
            raise ValueError(f"ppg_highcut ({value}) must be greater than ppg_lowcut ({self.ppg_lowcut})")

    @eda_highcut.validator
    def _check_eda_frequency_order(self, attribute, value):
        """Ensure highcut > lowcut for EDA."""
        if value <= self.eda_lowcut:
            raise ValueError(f"eda_highcut ({value}) must be greater than eda_lowcut ({self.eda_lowcut})")


@define
class PathConfig:
    """File path configuration."""

    # Data directories (store as strings for JSON serialization)
    data_root: str = field(default="Data", validator=attrs.validators.instance_of(str))
    raw_data_dir: str = field(default="raw", validator=attrs.validators.instance_of(str))
    processed_data_dir: str = field(default="processed", validator=attrs.validators.instance_of(str))
    exports_dir: str = field(default="exports", validator=attrs.validators.instance_of(str))
    sessions_dir: str = field(default="sessions", validator=attrs.validators.instance_of(str))

    # Supported file extensions
    supported_extensions: list[str] = field(factory=lambda: [".xdf", ".csv"])

    def get_data_root_path(self) -> Path:
        """Get data_root as a Path object."""
        return Path(self.data_root)


@define
class GUIConfig:
    """GUI configuration with color schemes and window settings."""

    # Window settings
    window_width: int = field(default=1400, validator=[attrs.validators.instance_of(int), positive_int])
    window_height: int = field(default=900, validator=[attrs.validators.instance_of(int), positive_int])
    theme: str = field(default="light", validator=attrs.validators.in_(["light", "dark"]))

    # Plot settings
    plot_background: str = field(default="white", validator=attrs.validators.instance_of(str))
    grid_alpha: float = field(default=0.3, validator=[attrs.validators.instance_of(float), nonnegative_float])

    # Peak marker colors (classification-based)
    peak_color_auto: str = field(default="blue", validator=attrs.validators.instance_of(str))  # Auto-detected
    peak_color_manual: str = field(default="green", validator=attrs.validators.instance_of(str))  # User-added
    peak_color_ectopic: str = field(default="orange", validator=attrs.validators.instance_of(str))  # Ectopic beat
    peak_color_bad: str = field(default="red", validator=attrs.validators.instance_of(str))  # Bad/artifact
    peak_color_selected: str = field(default="yellow", validator=attrs.validators.instance_of(str))  # Selected highlight

    # Signal colors (high-contrast for accessibility)
    signal_color_ecg: str = field(default="#1f77b4", validator=attrs.validators.instance_of(str))  # Blue
    signal_color_ppg: str = field(default="#d62728", validator=attrs.validators.instance_of(str))  # Red
    signal_color_eda: str = field(default="#2ca02c", validator=attrs.validators.instance_of(str))  # Green


@define
class AppConfig:
    """Main application configuration combining all sub-configs."""

    processing: ProcessingConfig = field(factory=ProcessingConfig)
    paths: PathConfig = field(factory=PathConfig)
    gui: GUIConfig = field(factory=GUIConfig)

    @classmethod
    def default(cls) -> AppConfig:
        """Create default configuration."""
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        """Create configuration from dictionary."""
        return cls(
            processing=ProcessingConfig(**data.get("processing", {})),
            paths=PathConfig(**data.get("paths", {})),
            gui=GUIConfig(**data.get("gui", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return attrs.asdict(self)

    def save(self, filepath: str | Path) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> AppConfig:
        """Load configuration from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        return cls.from_dict(data)


class ConfigManager:
    """Manages application configuration with environment variable overrides."""

    def __init__(self, config_dir: str | Path | None = None):
        """Initialize config manager.

        Args:
            config_dir: Directory for config files. Defaults to ~/.cardio_signal_lab/
        """
        if config_dir is None:
            config_dir = Path.home() / ".cardio_signal_lab"
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.user_config_path = self.config_dir / "user_config.json"
        self.default_config_path = self.config_dir / "default_config.json"

        self._config: AppConfig | None = None

    def get_config(self) -> AppConfig:
        """Get current configuration with environment variable overrides."""
        if self._config is None:
            self._config = self._load_config()
            self._apply_env_overrides()
        return self._config

    def _load_config(self) -> AppConfig:
        """Load configuration from user or default file."""
        # Try user config first
        if self.user_config_path.exists():
            return AppConfig.load(self.user_config_path)

        # Fall back to default
        if self.default_config_path.exists():
            return AppConfig.load(self.default_config_path)

        # Create new default config
        config = AppConfig.default()
        config.save(self.default_config_path)
        return config

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to config.

        Environment variables like CSL_ECG_LOWCUT=0.7 override config.processing.ecg_lowcut
        """
        if self._config is None:
            return

        env_prefix = "CSL_"

        # Processing overrides
        for attr in ["ecg_lowcut", "ecg_highcut", "ppg_lowcut", "ppg_highcut"]:
            env_var = f"{env_prefix}{attr.upper()}"
            if env_var in os.environ:
                setattr(self._config.processing, attr, float(os.environ[env_var]))

        # GUI overrides
        for attr in ["window_width", "window_height"]:
            env_var = f"{env_prefix}{attr.upper()}"
            if env_var in os.environ:
                setattr(self._config.gui, attr, int(os.environ[env_var]))

    def save_user_config(self) -> None:
        """Save current configuration as user config."""
        if self._config is not None:
            self._config.save(self.user_config_path)

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self._config = AppConfig.default()
        if self.user_config_path.exists():
            self.user_config_path.unlink()


# Global singleton instance
_config_manager: ConfigManager | None = None


def get_config() -> AppConfig:
    """Get global configuration singleton.

    Returns:
        AppConfig instance with current settings.

    Example:
        >>> from cardio_signal_lab.config.settings import get_config
        >>> config = get_config()
        >>> print(config.processing.ecg_lowcut)
        0.5
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.get_config()


def get_config_manager() -> ConfigManager:
    """Get global config manager singleton.

    Returns:
        ConfigManager instance for advanced config management.
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
