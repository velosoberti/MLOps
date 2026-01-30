"""Configuration module for ML Pipeline.

This module provides centralized configuration management using Pydantic settings.
All configuration values can be overridden via environment variables with the ML_ prefix.
"""

from config.settings import Settings, settings

__all__ = ["Settings", "settings"]
