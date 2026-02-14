"""Configuration management for CardioSignalLab."""

from .keybindings import (
    KEYBINDINGS,
    get_description,
    get_help_text,
    get_keysequence,
    get_shortcut_text,
)
from .settings import AppConfig, ConfigManager, get_config, get_config_manager

__all__ = [
    "AppConfig",
    "ConfigManager",
    "get_config",
    "get_config_manager",
    "KEYBINDINGS",
    "get_keysequence",
    "get_description",
    "get_shortcut_text",
    "get_help_text",
]
