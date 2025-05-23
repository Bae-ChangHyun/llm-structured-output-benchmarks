"""
Framework compatibility checker module.

This module provides functionality to check if a framework is compatible with
a specified model host based on the framework_compatibility.yaml configuration.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Any


class FrameworkCompatibilityError(Exception):
    """Exception raised when a framework is not compatible with a model host."""
    pass


class ConfigChecker:
    """
    Checks compatibility between frameworks and model hosts.
    """

    def __init__(self):
        """Initialize the compatibility checker with data from YAML file."""
        yaml_path = os.path.join(
            Path(__file__).parent, "framework_compatibility.yaml"
        )
        with open(yaml_path, "r") as file:
            self.compatibility_data = yaml.safe_load(file)

    def is_compatible(self, framework_name: str, model_host: str) -> bool:
        """
        Check if a framework is compatible with a model host.

        Args:
            framework_name: Name of the framework class
            model_host: Model host to check (openai, google, ollama, transformers)

        Returns:
            bool: True if compatible, False otherwise
        """
        if framework_name not in self.compatibility_data:
            # If framework is not in the compatibility list, assume it's unrestricted
            return True

        hosts = self.compatibility_data[framework_name].get("hosts", [])
        return model_host in hosts

    def check_compatibility(self, framework_name: str, model_host: str) -> None:
        """
        Check if a framework is compatible with a model host and raise an error if not.

        Args:
            framework_name: Name of the framework class
            model_host: Model host to check

        Raises:
            FrameworkCompatibilityError: If the framework is not compatible with the model host
        """
        if not self.is_compatible(framework_name, model_host):
            supported = self.compatibility_data[framework_name].get("hosts", [])
            supported_str = ", ".join(supported)
            raise FrameworkCompatibilityError(
                f"Framework '{framework_name}' is not compatible with model host '{model_host}'. "
                f"Supported model hosts: {supported_str}"
            )

    def get_hosts(self, framework_name: str) -> List[str]:
        """
        Get the list of supported model hosts for a framework.

        Args:
            framework_name: Name of the framework class

        Returns:
            List of supported model host names
        """
        if framework_name not in self.compatibility_data:
            return ["openai", "google", "ollama", "transformers"]  # Default to all
            
        return self.compatibility_data[framework_name].get("hosts", [])


# Singleton instance for easy import
compatibility_checker = ConfigChecker()